import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

# these two are used for pointcloud
from data_utils.trainDataset import trainDataset
from data_utils.validateDataset import validateDataset

# these two are used for rgb
from data_utils.Dataset_RGBD import EgoPAT3DDataset as RGBDDataset
from data_utils.Dataset_RGBD_enhanced import EgoPAT3DDataset_Enhanced as RGBDDataset_Enhanced


from model.baseline import *
from model.baseline_streaming import *
from loss import oriloss, last_oriloss, rgbloss, rgbloss_manual
from utils.utils import save_checkpoint
from configs.cfg_utils import load_cfg





def blockprint():
    # block printing for all processes except the first one since we are doing distributed training
    sys.stdout = open(os.devnull, 'w')    

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    
    parser.add_argument(
    '--config_file',
    default='',
    type=str,
    help='path to yaml config file',)
    return parser.parse_args()

def get_my_loss(cfg, **kwargs):
    '''
    Wrapper for the loss function. 
    So now when I modify loss function I only need to modify this function instead of in both train and val function
    '''
    if cfg.TRAINING.LOSS == 'Ori':
        # if we are using the old loss function
        criterion = kwargs['criterion']
        loss = criterion(pred=kwargs['pred'],
                        gt=kwargs['gt_xyz'],
                        length=kwargs['LENGTH'])

    elif cfg.TRAINING.LOSS == 'RGB_Ori':
        # if we are using the new loss function
        criterion = kwargs['criterion']
        loss = criterion(pred=kwargs['pred'],
                        gt=kwargs['gt_xyz'],
                        hand=kwargs['hand'],
                        length=kwargs['LENGTH'],
                        train=kwargs['train'])
    else:
        raise NotImplementedError('Not implemented loss')
    return loss

def get_my_pred(cfg, **kwargs):
    '''
    Wrapper for the prediction function. 
    So now when I modify prediction function I only need to modify this function instead of in both train and val function
    '''
    if cfg.MODEL.STREAMING == True:
        # Streaming
        pred_list = []
        hout, cout = 0, 0
        classifier = kwargs['classifier']

        # the first frame
        pred, hout, cout = classifier(img = kwargs['rgb'][:,0,:,:],
                          hand = kwargs['handLM'][:,0,:],
                          start = True,
                          hout = hout,
                          cout = cout,
                          cfg = cfg
                          )
        pred_list.append(pred)

        # the rest of the frames
        for idx in range(1, int(kwargs['LENGTH'][0])): # the first frame has been processed
            pred, hout, cout = classifier(img = kwargs['rgb'][:,idx,:,:],
                          hand = kwargs['handLM'][:,idx,:],
                          start = False,
                          hout = hout,
                          cout = cout,
                          cfg = cfg
                          )
            pred_list.append(pred)


    else:
        if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
            # pointcloud
            classifier = kwargs['classifier']
            pred_list = classifier(pointxyz = kwargs['pointcloud'][:,:,:3,:], 
                            pointfeat = kwargs['pointcloud'][:,:,3:,:],
                            motion = kwargs['motion'],
                            LEGHTN = kwargs['LENGTH'].max().repeat(torch.cuda.device_count()).to(kwargs['device']),
                            cfg = cfg
                            )
        else:
            # rgb
            classifier = kwargs['classifier']
            pred_list = classifier(img = kwargs['rgb'],
                            hand = kwargs['handLM'],
                            LEGHTN=kwargs['LENGTH'],
                            cfg = cfg
                            )

    return pred_list

def train(classifier, dataloader, optimizer, criterion, scheduler, scaler, device, global_rank, cfg):
    classifier.train()
    total_loss = 0
    scheduler.step()
    for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9, disable=global_rank!=0):
        # disable the progress bar for all processes except the first one
        
        if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
            # pointcloud
            gt_xyz,pointcloud, motion, LENGTH, _ = data
            pointcloud=pointcloud.transpose(3,2)
            gt_xyz, pointcloud, motion = gt_xyz.to(device),pointcloud.to(device), motion.to(device)

            optimizer.zero_grad()
            pred = get_my_pred(cfg=cfg,
                                classifier=classifier, 
                                pointcloud=pointcloud, 
                                motion=motion, 
                                LENGTH=LENGTH, 
                                device=device,
                                )

            loss = get_my_loss(cfg=cfg, 
                                pred=pred, 
                                gt_xyz=gt_xyz,
                                LENGTH=LENGTH,
                                criterion=criterion)
        else:
            # rgb
            gt_xyz, rgb, rangenum, finalsource, hand, handLM = data
            # some preprocessing needed to make the rgb channel first for PyTorch
            rgb = rgb.transpose(3,4)
            rgb = rgb.transpose(2,3)
            rgb, gt_xyz, handLM = rgb.to(device), gt_xyz.to(device), handLM.to(device)
            
            optimizer.zero_grad()

            pred = get_my_pred(cfg=cfg,
                                classifier=classifier,
                                rgb=rgb,
                                handLM=handLM,
                                LENGTH=[25],
                                device=device,
                                )
            
            loss = get_my_loss(cfg=cfg,
                                pred=pred,
                                gt_xyz=gt_xyz,
                                hand=hand,
                                LENGTH=rangenum,
                                criterion=criterion,
                                train=True,
                                )

        total_loss = loss + total_loss

        loss.backward()
        optimizer.step()

    
    # it's not appropriate to average the loss, because the different stages should have different loss
    return total_loss


def validate(classifier, dataloader, criterion, scaler, device, global_rank, cfg):
    classifier.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9, disable=global_rank!=0):
            # disable the progress bar for all processes except the first one
            
            if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
                # pointcloud
                gt_xyz,pointcloud, motion, LENGTH, _ = data
                pointcloud=pointcloud.transpose(3,2)
                gt_xyz, pointcloud, motion = gt_xyz.to(device),pointcloud.to(device), motion.to(device)

                pred = get_my_pred(cfg=cfg,
                                    classifier=classifier, 
                                    pointcloud=pointcloud, 
                                    motion=motion, 
                                    LENGTH=LENGTH, 
                                    device=device,
                                    )

                loss = get_my_loss(cfg=cfg, 
                                    pred=pred, 
                                    gt_xyz=gt_xyz,
                                    LENGTH=LENGTH,
                                    criterion=criterion)
            else:
                # rgb
                gt_xyz, rgb, rangenum, finalsource, hand, handLM = data

                rgb = rgb.transpose(3,4)
                rgb = rgb.transpose(2,3)
                rgb, gt_xyz, handLM = rgb.to(device), gt_xyz.to(device), handLM.to(device)
                
                pred = get_my_pred(cfg=cfg,
                                    classifier=classifier,
                                    rgb=rgb,
                                    handLM=handLM,
                                    LENGTH=rangenum.max().repeat(torch.cuda.device_count()).to(device),
                                    device=device,
                                    )
                
                loss = get_my_loss(cfg=cfg,
                                    pred=pred,
                                    gt_xyz=gt_xyz,
                                    hand=hand,
                                    LENGTH=rangenum,
                                    criterion=criterion,
                                    train=False
                                    )

            total_loss = loss + total_loss


    
    # it's not appropriate to average the loss, because the different stages should have different loss
    return total_loss


def main(cfg):
    '''DDP SETUP'''
    dist.init_process_group(backend="nccl") # init distributed
    slurm_proc_id = os.environ.get("SLURM_PROCID", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    print(f'process started with local rank: {local_rank}, global rank: {global_rank}, world size: {world_size}')

    '''CREATE DIR'''
    basepath=os.getcwd()
    experiment_dir = Path(os.path.join(basepath,'experiment'))
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s'%cfg.MODEL.MODEL_NAME)
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    # a place to save the evaluation results and sbatch stdouts
    eval_dir = file_dir.joinpath('eval/')
    eval_dir.mkdir(exist_ok=True)
    output_logs_dir = eval_dir.joinpath('output_logs/')
    output_logs_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(cfg.MODEL.MODEL_NAME)
    if global_rank == 0: # only log on the first process
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(file_dir.joinpath('train_%s_cls.txt'%cfg.MODEL.MODEL_NAME))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('---------------------------------------------------TRANING---------------------------------------------------')



    '''DATA LOADING'''
    if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
        # Pointcloud
        TRAIN_DATASET = trainDataset(cfg) 
        VAL_DATASET = validateDataset(cfg) 
    else:
        # RGB
        if cfg.DATA.ENHANCED == True:
            # traning with EgoPAT3Dv2
            TRAIN_DATASET = RGBDDataset_Enhanced(cfg, mode="annotrain")
            VAL_DATASET = RGBDDataset_Enhanced(cfg, mode="annovalidate")
        else:
            # training with EgoPAT3Dv1
            TRAIN_DATASET = RGBDDataset(cfg, mode="annotrain")
            VAL_DATASET = RGBDDataset(cfg, mode="annovalidate")

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(TRAIN_DATASET, shuffle=True)
        val_sampler = DistributedSampler(VAL_DATASET, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_iterator = DataLoader(TRAIN_DATASET, 
                                batch_size=cfg.DATA.DATA_LOADER.BATCH_SIZE,
                                num_workers=cfg.DATA.DATA_LOADER.NUM_WORKERS,
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=cfg.DATA.DATA_LOADER.PIN_MEMORY)
    val_iterator = DataLoader(VAL_DATASET, 
                                batch_size=int(cfg.DATA.DATA_LOADER.BATCH_SIZE*2), 
                                num_workers=cfg.DATA.DATA_LOADER.NUM_WORKERS,
                                sampler=val_sampler,
                                drop_last=True,
                                pin_memory=cfg.DATA.DATA_LOADER.PIN_MEMORY)
    
    if global_rank == 0: # only log on the first process
        logger.info("The number of training data is: %d", len(TRAIN_DATASET))


    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    
    if cfg.MODEL.STREAMING != True:
        # For the deployment of the streaming model, we need to have the best possible model
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    '''MODEL LOADING'''
    if cfg.MODEL.STREAMING == True:
        # Streaming
        classifier = Baseline_RGB_Streaming(cfg=cfg).train()

    else:
        # Non-streaming
        if cfg.MODEL.ARCH.POINTCLOUD == True and cfg.MODEL.ARCH.RGB == False:
            # Pointcloud
            classifier = Baseline(cfg=cfg).train()
        else:
            # RGB
            classifier = Baseline_RGB(cfg=cfg).train()

    if dist.is_available() and dist.is_initialized():
        device = f"cuda:{local_rank}"
        classifier = classifier.to(device)
        classifier = DDP(classifier, device_ids=None)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = classifier.to(device)

    classifier = torch.compile(classifier)

    if cfg.MODEL.CHECKPOINT != '':
        if global_rank == 0: # only log on the first process
            print('Use pretrain model...')
            logger.info('Use pretrain model')
        start_epoch = torch.load(cfg.MODEL.CHECKPOINT)['epoch']
        classifier.module.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT)['model_state_dict'])
    else:
        if global_rank == 0: # only log on the first process
            print('No existing model, starting training from scratch...')
        start_epoch = 0

    '''OPTIMIZER, LOSS, SCHEDULER'''
    # Optimizer
    if cfg.TRAINING.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.TRAINING.LEARNING_RATE, momentum=0.9, weight_decay=cfg.TRAINING.DECAY_RATE)
    elif cfg.TRAINING.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.TRAINING.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=cfg.TRAINING.DECAY_RATE)
    else:
        raise NotImplementedError('Not implemented optimizer')
    
    # Loss
    if cfg.TRAINING.LOSS == 'Ori':
        criterion = oriloss
    elif cfg.TRAINING.LOSS == 'Last_Ori':
        criterion = last_oriloss
    elif cfg.TRAINING.LOSS == 'RGB_Ori_Manual':
        criterion = rgbloss_manual
    elif cfg.TRAINING.LOSS == 'RGB_Ori':
        criterion = rgbloss

    else:
        raise NotImplementedError('Not implemented loss')
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


    '''TRANING'''
    if global_rank == 0: # only log on the first process
        logger.info('Start training...')

    scaler = None # not doing mixed precision training for now

    for epoch in range(start_epoch, cfg.TRAINING.NUM_EPOCHS):
        if global_rank != 0: # only log on the first process
            blockprint() 

        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, cfg.TRAINING.NUM_EPOCHS))
        logger.info('Epoch %d (%d/%s):' ,epoch + 1, epoch + 1, cfg.TRAINING.NUM_EPOCHS)
        print('lr=',optimizer.state_dict()['param_groups'][0]['lr'])

        train_total_loss = train(
            classifier=classifier,
            dataloader=train_iterator,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            global_rank=global_rank,
            cfg=cfg
        )

        torch.cuda.empty_cache() # clear cache between train and val

        val_total_loss = validate(
            classifier=classifier,
            dataloader=val_iterator,
            criterion=criterion,
            scaler=scaler,
            device=device,
            global_rank=global_rank,
            cfg=cfg
        )

        if global_rank == 0: # only log on the first process
            save_checkpoint(
                epoch + 1,
                classifier.module,
                optimizer,
                str(checkpoints_dir),
                cfg.MODEL.MODEL_NAME)
            print('Saving model....')
            logger.info(f'Training Loss: Total: {train_total_loss:.2f}')
            logger.info(f'Validation Loss: Total: {val_total_loss:.2f}')


    if global_rank == 0: # only log on the first process
        logger.info('End of training...')
    

if __name__ == '__main__':
    arg = parse_args()
    cfg = load_cfg(arg.config_file)
    main(cfg)
