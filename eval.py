import os
import matplotlib.pyplot as plt
import argparse
import re
import numpy as np
import open3d as o3d
from glob import glob
import pathlib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



def plott(x,y,modelname,savepath): # visulization of each clips
    
    colorlist=['red','lime','blue','fuchsia','black']

    fig1 = plt.figure(figsize=(20,8))
    plt.plot(x, y,color=colorlist[0])

    titlesize=15
    xysize=12
    
    plt.xlim(0, 1)
    plt.ylim(0, y.max()+y.min())
    font1 = {
    'weight' : 'normal',
    'size' : titlesize,
    }
    font2 = {
    'weight' : 'normal',
    'size' : xysize,
    }
    plt.legend(modelname, loc='upper left')
    
    plt.grid(which='major',color='gray',linestyle='--')
    plt.title('Prediction plot',font1)     
    plt.xlabel('Uniformed times',font2)         
    plt.ylabel('Error',font2)
    
    plt.savefig(savepath)
    plt.close(fig1)
    
def plotbar(xxx,ylabelname,modelname,savepath): # visulization of each scenes
        


    colorlist=['indianred','red','orangered','tomato','lightcoral','coral','lightsalmon','peachpuff','navajowhite','papayawhip']

    total_width, n = 0.6, 1
    width = total_width / n
    y=[]
    x=[]

    for timesthreshold in range(len(xxx.keys())):
        y.append(np.array(xxx[str(timesthreshold)]).mean()*100) #transfer m --> cm
        x.append(str(timesthreshold))

    weight=np.linspace(2,1,10)
    numm=round((np.array(y)*weight).sum()/weight.sum(),2)

    fig1 = plt.figure(figsize=(30,15))
    
    plt.bar(x, y, width=width, label=modelname+' ['+str(numm)+'] ',tick_label = ylabelname,color=colorlist)

    titlesize=40
    xysize=35
    plt.legend(loc='upper left',fontsize=20)
    font1 = {
    'weight' : 'normal',
    'size' : titlesize,
    }
    font2 = {
    'weight' : 'normal',
    'size' : xysize,
    }
    plt.grid(which='major',color='gray',linestyle='--')
    plt.title('Prediction plot',font1)       
    plt.xlabel('Uniformed times range',font2)           
    plt.ylabel('Error (cm)',font2)
    
    for a,b in zip(x,y):  
        plt.text(a, b+1, '%5.2f' % b, ha='center', va= 'bottom',fontsize=20) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 30)
    plt.savefig(savepath)
    plt.close(fig1)
    return y

def plotallbar(xxx,ylabelname,modelname,savepath):  # visulization of models on validation set or test set or unseen set
    total_width, n = 0.6, 1
    width = total_width / n
    x=np.array(list(range(len(ylabelname))))
    yy=[]

    colorlist=['indianred','red','orangered','tomato','lightcoral','coral','lightsalmon','peachpuff','navajowhite','papayawhip']

    
    yy=np.array(xxx).mean(0)
    print(xxx)
    weight=np.linspace(2,1,10)
    numm=round((np.array(yy)*weight).sum()/weight.sum(),2)



    
    fig1 = plt.figure(figsize=(30,15))

    plt.bar(x, yy, width=width,tick_label = ylabelname,color=colorlist,label=modelname+' ['+str(numm)+'] ')

    titlesize=40
    xysize=35
    plt.legend(loc='upper left',fontsize=20)
    font1 = {
    'weight' : 'normal',
    'size' : titlesize,
    }
    font2 = {
    'weight' : 'normal',
    'size' : xysize,
    }
    plt.grid(which='major',color='gray',linestyle='--')
    plt.title('Prediction plot',font1)       
    plt.xlabel('Uniformed times range',font2)          
    plt.ylabel('Error (cm)',font2)
    
    for a,b in zip(x,yy):  
        plt.text(a, b+1, '%5.2f' % b, ha='center', va= 'bottom',fontsize=20) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 30)
    plt.savefig(savepath)
    plt.close(fig1)

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()



def plotcompbar(xxx,ylabelname,modelname,savepath,mode): # comparison among different models
    result={}
    for num in range(len(modelname)):
        aa=[]
        for inum in xxx.keys():
            if inum.split('||')[0]==modelname[num]:
                record=[]
                for numclip in xxx[inum].keys():
                    record.append(np.array(xxx[inum][numclip]).mean())

                aa.append(record)

        result[modelname[num]]=np.array(aa).mean(0)*100

    total_width, n = 0.8, len(modelname)
    width = total_width / n
    x=[]
    for xlabel in np.linspace(0,0.9,10):
        x.append(str(round(xlabel,2))+'~'+str(round(xlabel+0.1,2)))

    delta=np.array(list(range(len(x))))

    fig1 = plt.figure(figsize=(250,50))
    colorlist=['limegreen','deepskyblue','red','orange','mediumpurple','slateblue','lawngreen','blue',\
        'fuchsia','hotpink','lightgray','olive','gold','turquoise']



    weight=np.linspace(2,1,10)

    for num in range(len(modelname)):
        
        
        numm=round((np.array(result[modelname[num]])*weight).sum()/weight.sum(),2)
           
        if len(modelname)%2==1:

            if num==(len(modelname)-1)//2:
                plt.bar(delta, result[modelname[num]], width=width, label=modelname[num]+' ['+str(numm)+'] ',tick_label = x,fc = colorlist[num])

            else:
                plt.bar(delta+(num-(len(modelname)-1)//2)*width, result[modelname[num]], width=width, label=modelname[num]+' ['+str(numm)+'] ',fc = colorlist[num])
        else:
            if num==(len(modelname))//2:
                plt.bar(delta, result[modelname[num]], width=width, label=modelname[num]+' ['+str(numm)+'] ',tick_label = x,fc = colorlist[num])

            else:
                plt.bar(delta+(num-(len(modelname))//2)*width, result[modelname[num]], width=width, label=modelname[num]+' ['+str(numm)+'] ',fc = colorlist[num])
        
        if os.path.exists('./results/metric/')==0:
                os.mkdir('./results/metric/')
        pathlib.Path('./results/metric/'+modelname[num]+'.txt').parent.mkdir(parents=True, exist_ok=True)
        np.savetxt('./results/metric/'+modelname[num]+'.txt', \
            np.array(result[modelname[num]]),fmt='%f',delimiter=',')

    
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.88, top=0.9,hspace=0.1,wspace=0.1)
    titlesize=300
    xysize=200
    plt.legend( loc='best',fontsize=150,bbox_to_anchor=(1, 1.05)) 
    font1 = {
    'weight' : 'normal',
    'size' : titlesize,
    }
    font2 = {
    'weight' : 'normal',
    'size' : xysize,
    }
    plt.grid(which='major',color='gray',linestyle='--',linewidth =10)
    plt.title('Prediction plot ('+mode+') ',font1)     
    plt.xlabel('Uniformed times range',font2)           
    plt.ylabel('Error (cm)',font2)
    plt.xticks(fontsize=150)
    plt.yticks(fontsize=150)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(10)
    ax.spines['left'].set_linewidth(10)
    ax.spines['right'].set_linewidth(10)
    ax.spines['top'].set_linewidth(10)


    plt.ylim(10, 30)
    plt.savefig(savepath)

    plt.close(fig1)


def extract_gtxyz(path):
    with open(path,'r') as f:
        alldata=f.readlines()
        reference={}
        for line in alldata:
            eachxyz=line.strip('\n').split(',')
            if len(eachxyz)==9:
                reference[eachxyz[0]+'-'+eachxyz[1]]=float(eachxyz[3]),float(eachxyz[4]),float(eachxyz[5])
                reference[eachxyz[1]+'-'+eachxyz[2]]=float(eachxyz[6]),float(eachxyz[7]),float(eachxyz[8])
            elif len(eachxyz)==5:
                reference[eachxyz[0]+'-'+eachxyz[1]]=float(eachxyz[2]),float(eachxyz[3]),float(eachxyz[4])
            elif len(eachxyz)==13:
                reference[eachxyz[0]+'-'+eachxyz[1]]=float(eachxyz[4]),float(eachxyz[5]),float(eachxyz[6])
                reference[eachxyz[1]+'-'+eachxyz[2]]=float(eachxyz[7]),float(eachxyz[8]),float(eachxyz[9])
                reference[eachxyz[2]+'-'+eachxyz[3]]=float(eachxyz[10]),float(eachxyz[11]),float(eachxyz[12])
    return reference
    
def extract_predictxyz(path):
    with open (path,'r') as f:
        alldata=f.readlines()
        data=np.zeros((len(alldata),3))

        extra_line = 0 # somehow some results get 1 extra number at the end
        for line in range(len(alldata)):
            eachxyz=alldata[line].strip('\n').split(',')
            if len(eachxyz) < 3:
                extra_line += 1
                continue
            data[line,:]=float(eachxyz[0]),float(eachxyz[1]),float(eachxyz[2])
        
        data = data[:line+1-extra_line, :]

    return data

def tryint(s):                    
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):            
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--data_path', default='./data/benchmark', help='The path of the dataset')
    parser.add_argument('--model_name', default='baseline_rgb_convnext_t', help='Model name')
    parser.add_argument('--mode', action='store_true', default='test', help='Evaluation on validate set or test set or unseen set')
    return parser.parse_args()

def main(args):
    path=os.getcwd()
    benchmarkbasepath=args.data_path
    if args.mode=='validate':
        annopath=os.path.join(benchmarkbasepath,'annovalidate')
    elif args.mode=='novel':
        annopath=os.path.join(benchmarkbasepath,'annonoveltest')
    else:
        annopath=os.path.join(benchmarkbasepath,'annotest')

    
    clip_value=[]
    scene_value=[]
    
    all_conpre={}
    all_models_2b_eval = [args.model_name + '/' + i.split('/')[-1] for i in glob(f'./experiment/{args.model_name}/result/*')] # grab all the folder names that contain the test results
    print(all_models_2b_eval)                                                                                     # note that here I did the trick to get the folder name, not the (relative path)
    for num in tqdm(range(len(all_models_2b_eval))):
        model_name, epoch_name = all_models_2b_eval[num].split('/')

        resultpath=os.path.join(path, f'./experiment/{model_name}/result/{epoch_name}')
        print(resultpath)
        all_value=[]
        scene_list=os.listdir(annopath)    
        for each_scene in scene_list:
        
            clip_path=os.path.join(annopath,each_scene)
            clip_list=os.listdir(clip_path)
            
            for each_record11 in clip_list:
                each_record=each_record11[:-4]

                gt_recordpath=os.path.join(annopath,each_scene,each_record+'.txt')
                gt_xyz=extract_gtxyz(gt_recordpath)
                pre_recordpath=os.path.join(resultpath,each_scene,each_record)
                cliplist=os.listdir(pre_recordpath)
                cliplist.sort(key=str2int)

                eachclipresult={}
                plotrange=np.around(np.arange(0,1.1,0.1), decimals=3)
                for eachbin in range(len(plotrange)-1):
                    eachclipresult[str(eachbin)]=[]

                for each_clip in cliplist:
                    
                    if '_gt' in each_clip:
                        continue
                    predict=extract_predictxyz(os.path.join(pre_recordpath,each_clip)) #n,3
                    gt=np.loadtxt(os.path.join(pre_recordpath,each_clip)[:-4]+"_gt.txt")
                    eachdistance=np.sqrt(((predict-gt)**2).sum(1))
                    times=1-np.linspace(0,1,len(predict))
                    base=each_clip.split('-')

                    xlist=np.arange(0,len(predict))/(len(predict)-1)

                    savepath=os.path.join(resultpath,'newfigure') 
                    if os.path.exists(os.path.join(savepath,each_scene,each_record))==0:
                        if os.path.exists(savepath)==0:
                            os.mkdir(savepath)
                        if os.path.exists(os.path.join(savepath,each_scene))==0:
                            os.mkdir(os.path.join(savepath,each_scene))
                        
                        os.mkdir(os.path.join(savepath,each_scene,each_record))
                    
                    for eachbin in range(len(plotrange)-1):


                        if np.isnan(eachdistance[np.where((xlist>=plotrange[eachbin])&(xlist<=plotrange[eachbin+1]))].mean())==0:
                            eachclipresult[str(eachbin)].append(\
                            eachdistance[np.where((xlist>=plotrange[eachbin])&(xlist<=plotrange[eachbin+1]))].mean())

                ylabelname=[]
                for iinum in range(len(plotrange)-1):
                    ylabelname.append(str(plotrange[iinum])+'~'+str(plotrange[iinum+1]))
                
                all_conpre[all_models_2b_eval[num]+'||'+each_record]=eachclipresult
                
                ress=plotbar(eachclipresult,ylabelname,all_models_2b_eval[num],os.path.join(savepath,each_scene,each_record+'.jpg'))
                all_value.append(ress)
                
        
        plotallbar(all_value,ylabelname,all_models_2b_eval[num],os.path.join(savepath,args.mode+'overall.jpg'))
        
    #plotcompbar(all_conpre,ylabelname,all_models_2b_eval,os.path.join(path,'results',args.mode+'compare.jpg'),args.mode)


            
            
if __name__ == '__main__':
    args = parse_args()
    main(args)
        
        
    
