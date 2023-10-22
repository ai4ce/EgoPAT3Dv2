import argparse
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import cv2
from data_utils.testDataset import testDataset
from data_utils.Dataset_RGBD import EgoPAT3DDataset as RGBDDataset
from data_utils.Dataset_RGBD_enhanced import EgoPAT3DDataset_Enhanced as RGBDDataset_Enhanced
from model.baseline import *
import logging
from pathlib import Path
from tqdm import tqdm
from configs.cfg_utils import load_cfg
from loss import rgb_generatepred


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
    parser.add_argument('--model_epoch', default='', help='model name and epoch number')
    parser.add_argument('--config_file', default='', type=str, help='path to yaml config file')
    return parser.parse_args()


def main(cfg, model_epoch, checkpoint_path):
    '''HYPER PARAMETER'''
    model_name = model_epoch.split('/')[0]
    epoch_number = model_epoch.split('/')[1] # because model names are usually model_name/epoch_number


    
    '''CREATE DIR'''
    result_folder_path = f'./experiment/{model_name}/result/{epoch_number}' 
    experiment_dir = Path(f'./experiment/{model_name}/eval/{epoch_number}')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = experiment_dir
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(cfg.MODEL.MODEL_NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%model_epoch.replace('/', '_'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')


    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = cfg.DATA.DATA_ROOT
    
    if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
        # Pointcloud
        TEST_DATASET = testDataset(cfg=cfg)
    else:
        # RGB
        if cfg.TESTING.ENHANCED == False:
            if cfg.TESTING.SEEN == True:
                TEST_DATASET = RGBDDataset(cfg, mode="annotest")
            else:
                # d1_unseen
                TEST_DATASET = RGBDDataset_Enhanced(cfg, mode="annotest_unseen_v1")
        else:
            if cfg.TESTING.SEEN == True:
                TEST_DATASET = RGBDDataset_Enhanced(cfg, mode="annotest")
            else:
                TEST_DATASET = RGBDDataset_Enhanced(cfg, mode="annotest_unseen")
    
    finaltestDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=cfg.TESTING.BATCH_SIZE,shuffle=False)
    logger.info("The number of test data is: %d", len(TEST_DATASET))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''MODEL LOADING'''
    if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
        # Pointcloud
        classifier = Baseline(cfg=cfg).train()
    else:
        # RGB
        classifier = Baseline_RGB(cfg=cfg).train()

    
    classifier = classifier.to(device).eval()
  
    print('Load CheckPoint...')
    logger.info('Load CheckPoint')
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])


    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')
    
    # RGB related data
    err = 0
    total = 0
    ttotal = 0

    with torch.no_grad():
        
        for data in tqdm(finaltestDataLoader, total=len(finaltestDataLoader), smoothing=0.9):
            if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
                # Pointcloud
                gt_xyz, pointcloud, motion , LENGTH,clipsource= data
                pointcloud=pointcloud.transpose(3,2)
                gt_xyz, pointcloud, motion = gt_xyz.to(device), pointcloud.to(device), motion.to(device)
                
                
                tic=cv2.getTickCount()

                pred = classifier(pointcloud[:,:,:3,:],
                                pointcloud[:,:,3:,:],
                                motion,
                                LENGTH.max().repeat(torch.cuda.device_count()).to(device),
                                cfg
                                )


                toc=cv2.getTickCount()-tic    
                toc /= cv2.getTickFrequency()
                print('speed:',LENGTH/toc,'FPS')
                scene_path = os.path.join(result_folder_path, clipsource[0][0],clipsource[1][0]) # path to save the results for each scene
                if not os.path.isdir(scene_path):
                    os.makedirs(scene_path)
                result_path=os.path.join(scene_path,clipsource[2][0]+'-'+clipsource[3][0]+'.txt') # path to save the predicted results for each clip
                gt_path=os.path.join(scene_path,clipsource[2][0]+'-'+clipsource[3][0]+'_gt.txt') # path to save the ground truth for each clip
                np.savetxt(gt_path,gt_xyz[0][:len(pred)].cpu().numpy())

                with open(result_path, 'w') as f:
                    for xx in pred:
                        
                        def dcon(x):
                            resultlist=torch.linspace(-1,1,1024*5).cuda()
                            x=x/x.max()

                            x[torch.where(x<=0.5)]=0

                            return (x*resultlist).sum()/x.sum()
    

                        data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                        f.write(data+'\n')
            
            else:
                # RGB
                gt_xyz, rgb, rangenum, finalsource, hand, handLM = data

                rgb = rgb.transpose(3,4)
                rgb = rgb.transpose(2,3)
        
                gt_xyz, rgb,hand,handLM=gt_xyz.to(device), rgb.to(device), hand.to(device), handLM.to(device)
                
                tic=cv2.getTickCount()
                pred = classifier(img=rgb,
                                  hand=handLM,
                                  LEGHTN=rangenum.max().repeat(torch.cuda.device_count()).to(device),
                                  cfg=cfg)



                toc=cv2.getTickCount()-tic    
                toc /= cv2.getTickFrequency()
                print('speed:',rangenum/toc,'FPS')
                scene_path = os.path.join(result_folder_path, finalsource[0][0],finalsource[1][0]) # path to save the results for each scene
                if not os.path.isdir(scene_path):
                    os.makedirs(scene_path)
                result_path=os.path.join(scene_path,str(int(finalsource[2][0]))+'-'+str(int(finalsource[3][0]))+'.txt') # path to save the predicted results for each clip
                gt_path=os.path.join(scene_path,str(int(finalsource[2][0]))+'-'+str(int(finalsource[3][0]))+'_gt.txt') # path to save the ground truth for each clip
                np.savetxt(gt_path,gt_xyz[0][:len(pred)].cpu().numpy())
                
                if cfg.TRAINING.LOSS == 'RGB_Ori':
                    # currently, we need to hardcode the camera intrinsic parameters and resolution. We will update this part in the future.
                    with open(result_path, 'w') as f:
                        id = 0
                        ma = 0
                        prev = 0
                        for xx in pred:
                            xx = xx.detach()
                            pos,handx,handy,tim = rgb_generatepred(xx[0])
                            u=(pos[0]*1.80820276e+03/pos[2]+1.94228662e+03)/3840
                            v=(pos[1]*1.80794556e+03/pos[2]+1.12382178e+03)/2160
                            if hand[0][id][0]!=0 and hand[0][id][1]!=0:
                                err += np.sqrt(float((handx-hand[0][id][0])**2+(handy-hand[0][id][1])**2))
                                total += 1
                            if id > 0 and hand[0][id-1][0]!=0 and hand[0][id-1][1]!=0 and hand[0][id][0]!=0 and hand[0][id][1]!=0:
                                diff = hand[0][id]-prev
                                diff = diff[0]**2+diff[1]**2
                                if diff >ma:
                                    ma = diff
                                if id >=10:
                                    u = u*diff/ma+hand[0][id][0]*(1-diff/ma)
                                    v = v*diff/ma+hand[0][id][1]*(1-diff/ma)

                                    z1 = pos[2]
                                    x1 = (int(hand[0][id][0]*3840)-1.94228662e+03)*z/1.80820276e+03
                                    y1 = (int(hand[0][id][1]*2160)-1.12382178e+03)*z/1.80794556e+03
                                    pos[0] = pos[0]*diff/ma+x1*(1-diff/ma)
                                    pos[1] = pos[1]*diff/ma+y1*(1-diff/ma)
                                    pos[2] = pos[2]*diff/ma+z1*(1-diff/ma)
                            ttotal += 1
                            u = int(u*3840)
                            v = int(v*2160)
                            if u>=3840:
                                u=3840
                            if u<0:
                                u=0
                            if v>=2160:
                                u=2160
                            if v<0:
                                v=0

                            z = pos[2]
                            x = (u-1.94228662e+03)*z/1.80820276e+03
                            y = (v-1.12382178e+03)*z/1.80794556e+03

                            data=str(float(x))+','+str(float(y))+','+str(float(z))

                            f.write(data+'\n')
                            if hand[0][id][0]!=0 and hand[0][id][1]!=0:
                                prev = hand[0][id]
                            id+=1 

                elif cfg.TRAINING.LOSS == 'Ori':
                    with open(result_path, 'w') as f:
                        for xx in pred:
                            
                            def dcon(x):
                                resultlist=torch.linspace(-1,1,1024*5).cuda()
                                x=x/x.max()

                                x[torch.where(x<=0.5)]=0

                                return (x*resultlist).sum()/x.sum()
    
                            data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                            f.write(data+'\n')
                
    logger.info('End of evaluation...')

if __name__ == '__main__':
    arg = parse_args()
    cfg = load_cfg(arg.config_file)
    model_epoch = arg.model_epoch
    checkpoint_path = arg.checkpoint
    main(cfg, model_epoch, checkpoint_path)
