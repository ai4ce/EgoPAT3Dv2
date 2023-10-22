import torch as t

'''Helper Functions'''
def generatepred(x):
        resultlist=t.linspace(-1,1,1024*5).cuda()
        x=x/x.max(1)[0].unsqueeze(-1)
        for i in range(3):
            x[i][t.where(x[i]<0.5)]=0
        return (x*resultlist).sum(1)/x.sum(1)

def rgb_generatepred_manual(x):
    '''
    rgb loss that's manual during inference
    '''
    resultlist=t.linspace(-1,1,1024*5).cuda()
    x=x/x.max(1)[0].unsqueeze(-1)
    for i in range(4):
            x[i][t.where(x[i]<0.5)]=0
    return (x[:3]*resultlist).sum(1)/x[:3].sum(1),(x[3]*resultlist).sum(0)/x[3].sum(0),(x[4]*resultlist).sum(0)/x[4].sum(0),(x[5]*resultlist).sum(0)/x[5].sum(0)

def rgb_generatepred(x):
    '''
    rgb loss that's incorporated into the training
    '''
    resultlist=t.linspace(-1,1,1024*5).cuda()
    x=x/x.max(1)[0].unsqueeze(-1)
    for i in range(6):
            x[i][t.where(x[i]<0.5)]=0
    return (x[:3]*resultlist).sum(1)/x[:3].sum(1),(x[3]*resultlist).sum(0)/x[3].sum(0),(x[4]*resultlist).sum(0)/x[4].sum(0),(x[5]*resultlist).sum(0)/x[5].sum(0)


def calculate(x,y):

        pred=generatepred(x)
        loss=((pred-y)**2).sum()

        return loss

def rgb_calculate_manual(x,y,hand,time,train):

        pred,handx,handy,pred_time=rgb_generatepred_manual(x)
        loss=((pred[0]-y[0])**2+(pred[1]-y[1])**2+(pred[2]-y[2])**2).sum()
        if train:
            loss+=0.1*((pred_time-time)**2).sum()
            if hand[0] != 0 and hand[1] != 0:
                loss += 0.1*((handx-hand[0])**2+(handy-hand[1])**2).sum()

        return loss


def rgb_calculate(x,y,hand,time,train,pre,id):

        pred,handx,handy,pred_time=rgb_generatepred(x)
        if id >= 10:
            if hand[id][0] != 0 and hand[id][1] != 0 and hand[id-1][0] != 0 and hand[id-1][1] != 0:
                ma = 0
                for i in range(id-1):
                    diff = hand[i+1]-hand[i]
                    diff = diff[0]**2+diff[1]**2
                    if diff>ma:
                        ma = diff

        loss=((pred[0]-y[0])**2+(pred[1]-y[1])**2+(pred[2]-y[2])**2).sum()
        if train:
            loss+=0.1*((pred_time-time)**2).sum()
            if hand[id][0] != 0 and hand[id][1] != 0:
                loss += 0.1*((handx-hand[id][0])**2+(handy-hand[id][1])**2).sum()

        return loss,pred,pred_time,handx,handy

'''Loss Functions'''
def oriloss(pred,gt,length):
    # original loss
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            loss.append((calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i])))
    return sum(loss)/batch

def last_oriloss(pred,gt,length):
    # experimental loss. Not used in the paper
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            if pred_xyz==length[i]-1: # only calculate the loss for the last frame
                loss.append((calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i])))
    return sum(loss)/batch

def scaled_oriloss(pred,gt,length):
    # experimental loss. Not used in the paper
    batch=gt.size()[0]
    loss=[]
    max_length = max(length) # the maximum sequence length in the batch
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            ori_loss = (calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i]))
            scaled_loss = ori_loss*(max_length/length[i]) # scale the loss by the maximum sequence length
            loss.append(scaled_loss)
    return sum(loss)/batch

def last_frame_loss(pred,gt):
    # experimental loss. Not used in the paper
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):

        loss.append(calculate(pred[0][i],gt[i,:]))
    return sum(loss)/batch

def last_frame_dist(pred,gt):
    # experimental loss. Not used in the paper
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):

        loss.append(t.sqrt(calculate(pred[0][i],gt[i,:])))
    return sum(loss)/batch

def own_l2_loss(pred,gt,length):
    # experimental loss. Not used in the paper
    '''
    pred: (seq_len, batch, 1024)
    gt: (seq_len, batch, 1024)
    length: (batch)
    '''
    batch=gt[0].shape[0] # batch size
    loss=[]
    for i in range(batch):
        
        # the rest after length[i] are 0-padded
        sequence_loss = []
        for j in range(length[i]):
            adjusted_loss = t.nn.functional.mse_loss(pred[j][i], gt[j][i], reduction='mean')*(2-j/length[i]) # adjust the loss by the sequence length
            sequence_loss.append(adjusted_loss)
        sequence_loss = t.as_tensor(sequence_loss) # cast to tensor to preserve the autograd graph
        sequence_loss = t.sum(sequence_loss)/length[i] # average the loss over the sequence
        loss.append(sequence_loss)
    loss = t.as_tensor(loss) # cast to tensor to preserve the autograd graph
    return t.sum(loss)/batch

def rgbloss_manual(pred,gt,hand,length,train=True):
    # experimental loss. Not used in the paper
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            loss.append(25*(rgb_calculate_manual(pred[pred_xyz][i],gt[i][pred_xyz],hand[i][pred_xyz],pred_xyz/length[i],train)*(2-pred_xyz/length[i]))/length[i])
            if loss[-1]>100:
                print("large",length[i],loss[-1],gt[i][pred_xyz])
    return sum(loss)/(batch)

def rgbloss(pred,gt,hand,length,train=True):
    # loss used in the paper
    batch=gt.size()[0]
    loss=[]
    pre = []
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            single,pres,pred_time,handx,handy = rgb_calculate(pred[pred_xyz][i],gt[i][pred_xyz],hand[i],pred_xyz/length[i],train,pre,pred_xyz)
            loss.append(25*(single*(2-pred_xyz/length[i]))/length[i])
            if loss[-1]>100:
                print("large",length[i],loss[-1],gt[i][pred_xyz],hand[i][pred_xyz],handx,handy,pred_time,pred_xyz)
            pre.append(pres)
    return sum(loss)/(batch)

