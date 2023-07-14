import numpy as np
import warnings
import os
import open3d as o3d
import cv2
# cv2.setNumThreads(0) 
# import multiprocessing
# multiprocessing.set_start_method('spawn') 
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import re
from tqdm import tqdm
from functools import reduce
import h5py
import time
import sys


class trainDataLoader(Dataset):
    def __init__(self, root, num):  # root="/scratch/yw5458/EgoPAT3D/dataset.hdf5"
        self.root = root
        self.rgb_video_path = "/scratch/yw5458/EgoPAT3D/videos"
        self.num = num
        self.indexlist = []
        self.cliplength = []
        self.mode = "annotrain"

        dataset_file = h5py.File(self.root, "r")
        gt_grp = dataset_file[self.mode]
        for scene_name in gt_grp:
            scene_grp = gt_grp[scene_name]
            for video_name in scene_grp:
                video = scene_grp[video_name]
                video = np.asarray(video)
                for line in video:
                    self.indexlist.append([scene_name, video_name, line[0], line[1], line[3:6]])        # scene name(nightstand), video name(nightstand_3), start frame, end frame, ground truth position at last frame
                    self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    self.indexlist.append([scene_name, video_name, line[1], line[2], line[6:]])
                    self.cliplength.append(int( int(line[2])-int(line[1]) ))
        dataset_file.close()
        self.maxclip=25
        self.indexoff = np.where((np.array(self.cliplength)<=25)==1)[0]
        self.length = len(self.indexoff)
 
    def __len__(self):
        return len(self.indexoff)
    
    def __getitem__(self, index):
        finalsource = self.indexlist[self.indexoff[index]]
        dataset_file = h5py.File(self.root, "r")
        video_path = f"sequences/{finalsource[0]}/{finalsource[1]}"
        video = dataset_file[f"{video_path}"]
        
        #
        pointcloud_grp = video[f"pointcloud"]
        
        odometry_grp = video[f"transformation/odometry"]
        
        imu_file = np.array(video[f"imu"])
        
        rgb_path = self.rgb_video_path
        rgb_path = os.path.join(rgb_path, finalsource[0], finalsource[1]+".mp4")
        cap = cv2.VideoCapture(rgb_path)
        if not cap.isOpened():
            print(rgb_path)
            return -1
        pointcloud=np.zeros((self.maxclip,self.num,6))
        geometry=np.zeros((self.maxclip,18))  
        gt_xyz=np.zeros((self.maxclip,3)) 
        image=np.zeros((self.maxclip, 3, 224, 224))
        
        #
        first=np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist=[]
        rangenum=int(finalsource[3])-int(finalsource[2])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(finalsource[2]-1))
        for idx in range(rangenum):
            pointxyz = np.array(pointcloud_grp[f"pointxyz{idx+1+int(finalsource[2])}"])
            pointcolor = np.array(pointcloud_grp[f"pointcolor{idx+1+int(finalsource[2])}"])
            randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))        # [8192, ]
            pointcloud[idx,:,:3] = pointxyz[randomlist]                           # [8192, 3]
            pointcloud[idx,:,3:] = pointcolor[randomlist]                         # [8192, 3]
            
            frame = idx+int(finalsource[2])
            
            ret, rgb_frame = cap.read()
            # rgb_height, rgb_weight, _ = rgb_frame.shape   # [2160, 3840, 3]
            # rgb_height, rgb_weight = int((1/8)*rgb_height), int((1/8)*rgb_weight) # [270, 480, 3]
            # rgb_frame = cv2.resize(np.array(rgb_frame), (rgb_weight, rgb_height))
            rgb_frame = cv2.resize(np.array(rgb_frame), (224, 224))
            rgb_frame = rgb_frame.transpose((2,0,1)) # [3, 270, 480]
            image[idx] = np.array(rgb_frame, dtype=np.byte)
            # if idx!=0:
            odometrylist.append(np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"]))
            odometry=reduce(np.dot, odometrylist)
            # if idx ==0:
            #     gt_xyz[idx,:]=first
            # else:
            gt_xyz[idx,:]=np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
            transformationsource = np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"])[:3].reshape(-1)
                                                                                        
            imudata=imu_file[0][1:].reshape(-1)                                 
            imu_sign = 0                                                        
            for imu_data in imu_file:                                           
                if int(imu_data[0])==frame:                                          
                    imudata = imu_data[1:].reshape(-1)                          
                    imu_sign = 1                                                
                else:                                                           
                    if imu_sign == 1:                                           
                        break                                               

            # transformationsource=np.load(os.path.join(transformationsourcepath,str(idx+int(finalsource[2]))+'.npy'))[:3].reshape(-1)
            # # transformation source: /Dataset/sequences/scene/$/transformation/odometry/$.npy       # shape (12,)
            # imudata=self.getimudata(imupath,1+idx+int(finalsource[2])).reshape(-1)
            # imu data path: /Dataset/sequences/scene/$/data.txt                                    # shape (6,)
            geometry[idx]=np.concatenate((transformationsource,imudata),0)
        dataset_file.close()
        cap.release()
        return gt_xyz,pointcloud,geometry,image,rangenum,finalsource

class trainDataLoader_without_pc(Dataset):
    def __init__(self, root, num):  # root="/scratch/yw5458/EgoPAT3D/dataset.hdf5"
        self.root = root
        self.rgb_video_path = "/scratch/yw5458/EgoPAT3D/videos"
        self.num = num
        self.indexlist = []
        self.cliplength = []
        self.mode = "annotrain"

        dataset_file = h5py.File(self.root, "r")
        gt_grp = dataset_file[self.mode]
        for scene_name in gt_grp:
            scene_grp = gt_grp[scene_name]
            for video_name in scene_grp:
                video = scene_grp[video_name]
                video = np.asarray(video)
                for line in video:
                    self.indexlist.append([scene_name, video_name, line[0], line[1], line[3:6]])        # scene name(nightstand), video name(nightstand_3), start frame, end frame, ground truth position at last frame
                    self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    self.indexlist.append([scene_name, video_name, line[1], line[2], line[6:]])
                    self.cliplength.append(int( int(line[2])-int(line[1]) ))
        dataset_file.close()
        self.maxclip=25
        self.indexoff = np.where((np.array(self.cliplength)<=25)==1)[0]
        self.length = len(self.indexoff)
 
    def __len__(self):
        return len(self.indexoff)
    
    def __getitem__(self, index):
        finalsource = self.indexlist[self.indexoff[index]]
        dataset_file = h5py.File(self.root, "r")
        video_path = f"sequences/{finalsource[0]}/{finalsource[1]}"
        video = dataset_file[f"{video_path}"]
        
        #
        # pointcloud_grp = video[f"pointcloud"]
        
        odometry_grp = video[f"transformation/odometry"]
        
        imu_file = np.array(video[f"imu"])
        
        rgb_path = self.rgb_video_path
        rgb_path = os.path.join(rgb_path, finalsource[0], finalsource[1]+".mp4")
        cap = cv2.VideoCapture(rgb_path)
        if not cap.isOpened():
            print(rgb_path)
            return -1
        # pointcloud=np.zeros((self.maxclip,self.num,6))
        geometry=np.zeros((self.maxclip,18))  
        gt_xyz=np.zeros((self.maxclip,3)) 
        image=np.zeros((self.maxclip, 3, 224, 224))
        
        #
        first=np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist=[]
        rangenum=int(finalsource[3])-int(finalsource[2])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(finalsource[2]-1))
        for idx in range(rangenum):
            # pointxyz = np.array(pointcloud_grp[f"pointxyz{idx+1+int(finalsource[2])}"])
            # pointcolor = np.array(pointcloud_grp[f"pointcolor{idx+1+int(finalsource[2])}"])
            # randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))        # [8192, ]
            # pointcloud[idx,:,:3] = pointxyz[randomlist]                           # [8192, 3]
            # pointcloud[idx,:,3:] = pointcolor[randomlist]                         # [8192, 3]
            
            frame = idx+int(finalsource[2])
            
            ret, rgb_frame = cap.read()
            rgb_frame = cv2.resize(np.array(rgb_frame), (224, 224))
            rgb_frame = rgb_frame.transpose((2,0,1)) # [3, 270, 480]
            image[idx] = np.array(rgb_frame, dtype=np.byte)
            # if idx!=0:
            odometrylist.append(np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"]))
            odometry=reduce(np.dot, odometrylist)
            # if idx ==0:
            #     gt_xyz[idx,:]=first
            # else:
            gt_xyz[idx,:]=np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
            transformationsource = np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"])[:3].reshape(-1)
                                                                                        
            imudata=imu_file[0][1:].reshape(-1)                                 
            imu_sign = 0                                                        
            for imu_data in imu_file:                                           
                if int(imu_data[0])==frame:                                          
                    imudata = imu_data[1:].reshape(-1)                          
                    imu_sign = 1                                                
                else:                                                           
                    if imu_sign == 1:                                           
                        break                                               

            # transformationsource=np.load(os.path.join(transformationsourcepath,str(idx+int(finalsource[2]))+'.npy'))[:3].reshape(-1)
            # # transformation source: /Dataset/sequences/scene/$/transformation/odometry/$.npy       # shape (12,)
            # imudata=self.getimudata(imupath,1+idx+int(finalsource[2])).reshape(-1)
            # imu data path: /Dataset/sequences/scene/$/data.txt                                    # shape (6,)
            geometry[idx]=np.concatenate((transformationsource,imudata),0)
        dataset_file.close()
        cap.release()
        return gt_xyz,geometry,image,rangenum,finalsource


class trainDataLoader_without_image(Dataset):
    def __init__(self, root, num):  # root="/scratch/yw5458/EgoPAT3D/dataset.hdf5"
        self.root = root
        self.rgb_video_path = "/scratch/yw5458/EgoPAT3D/videos"
        self.num = num
        self.indexlist = []
        self.cliplength = []
        self.mode = "annotrain"

        dataset_file = h5py.File(self.root, "r")
        gt_grp = dataset_file[self.mode]
        for scene_name in gt_grp:
            scene_grp = gt_grp[scene_name]
            for video_name in scene_grp:
                video = scene_grp[video_name]
                video = np.asarray(video)
                for line in video:
                    self.indexlist.append([scene_name, video_name, line[0], line[1], line[3:6]])        # scene name(nightstand), video name(nightstand_3), start frame, end frame, ground truth position at last frame
                    self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    self.indexlist.append([scene_name, video_name, line[1], line[2], line[6:]])
                    self.cliplength.append(int( int(line[2])-int(line[1]) ))
        dataset_file.close()
        self.maxclip=25
        self.indexoff = np.where((np.array(self.cliplength)<=25)==1)[0]
        self.length = len(self.indexoff)
 
    def __len__(self):
        return len(self.indexoff)
    
    def __getitem__(self, index):
        finalsource = self.indexlist[self.indexoff[index]]
        dataset_file = h5py.File(self.root, "r")
        video_path = f"sequences/{finalsource[0]}/{finalsource[1]}"
        video = dataset_file[f"{video_path}"]
        
        #
        pointcloud_grp = video[f"pointcloud"]
        
        odometry_grp = video[f"transformation/odometry"]
        
        imu_file = np.array(video[f"imu"])
        
        # rgb_path = self.rgb_video_path
        # rgb_path = os.path.join(rgb_path, finalsource[0], finalsource[1]+".mp4")
        # cap = cv2.VideoCapture(rgb_path)
        # if not cap.isOpened():
        #     print(rgb_path)
        #     return -1
        pointcloud=np.zeros((self.maxclip,self.num,6))
        geometry=np.zeros((self.maxclip,18))  
        gt_xyz=np.zeros((self.maxclip,3)) 
        # image=np.zeros((self.maxclip, 3, 224, 224))
        
        #
        first=np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist=[]
        rangenum=int(finalsource[3])-int(finalsource[2])
        
        # cap.set(cv2.CAP_PROP_POS_FRAMES, int(finalsource[2]-1))
        for idx in range(rangenum):
            pointxyz = np.array(pointcloud_grp[f"pointxyz{idx+1+int(finalsource[2])}"])
            pointcolor = np.array(pointcloud_grp[f"pointcolor{idx+1+int(finalsource[2])}"])
            randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))        # [8192, ]
            pointcloud[idx,:,:3] = pointxyz[randomlist]                           # [8192, 3]
            pointcloud[idx,:,3:] = pointcolor[randomlist]                         # [8192, 3]
            
            frame = idx+int(finalsource[2])
            

            # if idx!=0:
            odometrylist.append(np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"]))
            odometry=reduce(np.dot, odometrylist)
            # if idx ==0:
            #     gt_xyz[idx,:]=first
            # else:
            gt_xyz[idx,:]=np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
            transformationsource = np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"])[:3].reshape(-1)
                                                                                        
            imudata=imu_file[0][1:].reshape(-1)                                 
            imu_sign = 0                                                        
            for imu_data in imu_file:                                           
                if int(imu_data[0])==frame:                                          
                    imudata = imu_data[1:].reshape(-1)                          
                    imu_sign = 1                                                
                else:                                                           
                    if imu_sign == 1:                                           
                        break                                               

            # transformationsource=np.load(os.path.join(transformationsourcepath,str(idx+int(finalsource[2]))+'.npy'))[:3].reshape(-1)
            # # transformation source: /Dataset/sequences/scene/$/transformation/odometry/$.npy       # shape (12,)
            # imudata=self.getimudata(imupath,1+idx+int(finalsource[2])).reshape(-1)
            # imu data path: /Dataset/sequences/scene/$/data.txt                                    # shape (6,)
            geometry[idx]=np.concatenate((transformationsource,imudata),0)
        dataset_file.close()
        # cap.release()
        return gt_xyz,pointcloud,geometry,rangenum,finalsource



        
if __name__ == '__main__':
    import torch
    DATA_PATH = './Benchmark/'
    root="/scratch/yw5458/EgoPAT3D/dataset.hdf5"
    TRAIN_DATASET = trainDataLoader(root=root,num=8192)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=True, num_workers=16)
    for epoch in range(5):
        # time1 = time.time()
        # for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        for data in tqdm(trainDataLoader): 
            gt_xyz,pointcloud,geometry,image,rangenum,finalsource = data
            print(max(pointcloud))
            # print("gt_xyz", sys.getsizeof(gt_xyz), "pointcloud", sys.getsizeof(pointcloud),"geometry", sys.getsizeof(geometry) , sys.getsizeof(data) )
            # a.size * a.itemsize
            print("gt_xyz", gt_xyz.size, "pointcloud", pointcloud.size,"geometry", geometry.size , sys.getsizeof(data) )
            # print(time.time()-time1)
            # time1 = time.time()
            # print(batch_id)
        


