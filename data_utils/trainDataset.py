import numpy as np
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
from functools import reduce
import h5py


class trainDataset(Dataset):
    def __init__(self, cfg):
        self.root = cfg.DATA.DATA_ROOT
        self.num = cfg.DATA.NUM_POINTS
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
                    if line[5]>0:
                        self.indexlist.append([scene_name, video_name, line[0], line[1], line[3:6]])        # scene name(nightstand), video name(nightstand_3), start frame, end frame, ground truth position at last frame
                        self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    if line[8]>0:
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
        

        pointcloud=np.zeros((self.maxclip,self.num,6))
        geometry=np.zeros((self.maxclip,18))  
        gt_xyz=np.zeros((self.maxclip,3)) 

        first=np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist=[]
        rangenum=int(finalsource[3])-int(finalsource[2])
        

        for idx in range(rangenum):
            pointxyz = np.array(pointcloud_grp[f"pointxyz{idx+1+int(finalsource[2])}"])
            pointcolor = np.array(pointcloud_grp[f"pointcolor{idx+1+int(finalsource[2])}"])
            randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))        # [8192, ]
            pointcloud[idx,:,:3] = pointxyz[randomlist]                           # [8192, 3]
            pointcloud[idx,:,3:] = pointcolor[randomlist]                         # [8192, 3]
            
            frame = idx+int(finalsource[2])
            


            odometrylist.append(np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"]))
            odometry=reduce(np.dot, odometrylist)

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

            geometry[idx]=np.concatenate((transformationsource,imudata),0)
        dataset_file.close()
        return gt_xyz,pointcloud,geometry,rangenum,finalsource




