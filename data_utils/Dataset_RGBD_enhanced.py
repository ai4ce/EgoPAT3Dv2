import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import reduce
import h5py
import sys
from PIL import Image
import io
import mediapipe as mp


class EgoPAT3DDataset_Enhanced(Dataset):
    def __init__(self, cfg, mode):  
        '''
            root_annotation: path to the hdf5 file storing annotation and transformation
            root_RGB: path to the hdf5 file storing RGB files
            num: point numbers used in pointcloud
            mode: "annotrain" / "annotest" / "annovalidate" / "annotest_unseen"
        '''
        self.root_RGB = cfg.DATA.DATA_ROOT
        self.root_annotation = cfg.DATA.ENHANCED_ANNO  # ./data/annotation_transformation.hdf5.hdf5
        self.mode = mode
        
        self.indexlist = []
        self.cliplength = []
        
        if self.mode == "annotrain":
            self.maxclip = 25
        elif self.mode == "annotest" or self.mode == 'annotest_unseen' or self.mode == 'annotest_unseen_v1':
            self.maxclip = 0
            self.root_RGB = cfg.TESTING.DATASET
        else:
            self.maxclip = 0

        annotation_file = h5py.File(self.root_annotation, "r")
        gt_grp = annotation_file[self.mode]
        for scene_name in gt_grp:
            scene_grp = gt_grp[scene_name]
            for video_name in scene_grp:
                video = scene_grp[video_name]
                video = np.asarray(video)
                for line in video:
                    if line[5]>0:
                        self.indexlist.append([scene_name, video_name, line[0], line[1], line[3:6]])        # scene name(nightstand), video name(nightstand_3), start frame, end frame, ground truth position at last fram
                        self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    if line[8]>0:
                        self.indexlist.append([scene_name, video_name, line[1], line[2], line[6:]])
                        self.cliplength.append(int( int(line[2])-int(line[1]) ))
                    if self.mode!="annotrain":
                        if line[5]>0:
                            self.maxclip = max(self.maxclip, int( int(line[1])-int(line[0]) ))
                        if line[8]>0:
                            self.maxclip = max(self.maxclip, int( int(line[2])-int(line[1]) ))
        annotation_file.close()
        self.indexoff = np.where((np.array(self.cliplength)<=25)==1)[0]
        self.length = len(self.indexoff)
 
    def __len__(self):
        if self.mode=="annotrain":
            return len(self.indexoff)
        else:
            return len(self.indexlist)
    
    def __getitem__(self, index):
        if self.mode=="annotrain":
            finalsource = self.indexlist[self.indexoff[index]]
        else:
            finalsource = self.indexlist[index]
        dataset_file_RGB = h5py.File(self.root_RGB, "r")
        transformation_file = h5py.File(self.root_annotation, "r")
        
        video_path = f"color/{finalsource[0]}/{finalsource[1]}"         
        video = dataset_file_RGB[f"{video_path}"]                               # /bathroomCabinet/bathroomCabinet_1/
        # modality's group in hdf5
        # color_grp = video["color"]
        # depth_grp = video["depth"]
        odometry_grp = transformation_file[f"transformation/{finalsource[0]}/{finalsource[1]}"]
        # imu_grp = np.array(video[f"imu"])
        hand = np.zeros((self.maxclip,2))
        handLM = np.zeros((self.maxclip,42))
        mp_hands = mp.solutions.hands
        # default [max, *] to each modality
        color = np.zeros((self.maxclip, 224,224, 3))
        # depth = np.zeros((self.maxclip, 2160, 3840, 3))
        # motion = np.zeros((self.maxclip,18))  
        gt_xyz = np.zeros((self.maxclip,3)) 


        first = np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist = []
        rangenum = int(finalsource[3])-int(finalsource[2])        # rangenum is end-start, so there's n-1 frames since there should be n(end-start+1) frames in total in a clip
        
        for idx in range(rangenum):
            # color[idx] = color_grp[f"{idx+1+int(finalsource[2])}"]
            # depth[idx] = depth_grp[f"{idx+1+int(finalsource[2])}"]
            color_temp = Image.open(io.BytesIO(np.array(video[f"{idx+1+int(finalsource[2])}"]))).resize((224,224))
            # depth_temp = Image.open(io.BytesIO(np.array(depth_grp[f"{idx+1+int(finalsource[2])}"])))
            color[idx] = np.array(color_temp)
            # depth = np.array(depth_temp)
            # frame = idx+int(finalsource[2])
            odometrylist.append(np.array(odometry_grp[f"{idx+int(finalsource[2])}"]))
            odometry = reduce(np.dot, odometrylist)
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                results = hands.process(np.array(color_temp))
                if results.multi_hand_landmarks is not None:
                    for landmarks in results.multi_hand_landmarks:
                        id=0
                        for lm in landmarks.landmark:
                            handLM[idx,id*2] = lm.x
                            handLM[idx,id*2+1] = lm.x
                            if id == 5:
                                hand[idx,0] = lm.x
                                hand[idx,1] = lm.y
                            id+=1
            gt_xyz[idx,:] = np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
            # transformationsource = np.array(odometry_grp[f"{idx+int(finalsource[2])}"])[:3].reshape(-1) #[4, 4]->[3, 4]->[12, ]
                                                                                      
            # imudata=imu_grp[0][1:].reshape(-1)                                 
            # imu_sign = 0                                                        
            # for imu_data in imu_grp:                                           
            #     if int(imu_data[0])==frame:                                          
            #         imudata = imu_data[1:].reshape(-1)                          
            #         imu_sign = 1                                                
            #     else:                                                           
            #         if imu_sign == 1:                                           
            #             break                                              

            # transformation: shape (12,)
            # imu: shape (6,)
            # motion[idx]=np.concatenate((transformationsource,imudata),0)
        dataset_file_RGB.close()
        transformation_file.close()
        
        # return gt_xyz, motion, transformationsource, imudata, color, depth, rangenum, finalsource
        return gt_xyz, color, rangenum, finalsource, hand, handLM
 



        
if __name__ == '__main__':
    import torch
    # DATA_PATH = './Benchmark/'
    root_RGB="/scratch/yw5458/new_dataset/RGB_dataset.hdf5"
    root_annotation="/scratch/yw5458/new_dataset/annotation_transformation.hdf5"
    # /scratch/yw5458/new_dataset/annotation_transformation.hdf5.hdf5
    # /scratch/yw5458/new_dataset/RGB_dataset.hdf5
    mode = "annotrain"
    DATASET = EgoPAT3DDataset(root_annotation=root_annotation, root_RGB=root_RGB,num=8192, mode=mode)
    DataLoader = torch.utils.data.DataLoader(DATASET, batch_size=1, shuffle=True, num_workers=8)
    for epoch in range(5):
        for data in tqdm(DataLoader): 
            # gt_xyz, motion, transformationsource, imudata, color, depth, rangenum, finalsource = data
            gt_xyz, color, rangenum, finalsource = data
            print("gt_xyz", gt_xyz.size, "color", color.size,  sys.getsizeof(data) )

        


