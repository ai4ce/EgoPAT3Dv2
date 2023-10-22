import numpy as np
from torch.utils.data import Dataset
from functools import reduce
import h5py
from PIL import Image
import io
import mediapipe as mp


class EgoPAT3DDataset(Dataset):
    def __init__(self, mode):
        self.root = '/scratch/yc6317/EgoPAT3D/dataset.hdf5'
        self.indexlist = []
        self.cliplength = []
        self.mode = mode
        if self.mode == "annotrain":
            self.maxclip = 25
        else:
            self.maxclip = 0

        dataset_file = h5py.File(self.root, "r")
        gt_grp = dataset_file[self.mode]
        for scene_name in gt_grp:
            scene_grp = gt_grp[scene_name]
            for video_name in scene_grp:
                video = scene_grp[video_name]
                video = np.asarray(video)
                for line in video:
                    if line[5]>0.1:
                        self.indexlist.append([scene_name, video_name, line[0], line[1], line[3:6]])
                        self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    if line[-1]>0.1:
                        self.indexlist.append([scene_name, video_name, line[1], line[2], line[6:]])
                        self.cliplength.append(int( int(line[2])-int(line[1]) ))
                    if self.mode!="annotrain":
                        self.maxclip = max(self.maxclip, int( int(line[1])-int(line[0]) ))
                        self.maxclip = max(self.maxclip, int( int(line[2])-int(line[1]) ))
        dataset_file.close()
        self.indexoff = np.where((np.array(self.cliplength)<=25)==1)[0]
        self.length = len(self.indexoff)
 
    def __len__(self):
        if self.mode=="annotrain":
            return len(self.indexoff)
        else:
            return len(self.indexlist)
    
    def __getitem__(self, index):
        if self.mode=="annotrain":
            finalsource = self.indexlist[self.indexoff[2]]
        else:
            finalsource = self.indexlist[2]
        dataset_file = h5py.File(self.root, "r")
        video_path = f"sequences/{finalsource[0]}/{finalsource[1]}"         
        video = dataset_file[f"{video_path}"]                               # /bathroomCabinet/bathroomCabinet_1/
        
        # modality's group in hdf5
        color_grp = video["color"]
        # depth_grp = video["depth"]
        odometry_grp = video["transformation/odometry"]
        # imu_grp = np.array(video[f"imu"])
        mp_hands = mp.solutions.hands


        # default [max, *] to each modality
        # color = np.zeros((self.maxclip, 224,224, 3))
        # # depth = np.zeros((self.maxclip, 2160, 3840, 3))
        # # motion = np.zeros((self.maxclip,18))  
        # gt_xyz = np.zeros((self.maxclip,3)) 
        # # gt_xy = np.zeros((self.maxclip,2)) 
        hand = np.zeros((2))
        handLM = np.zeros((42))



        first = np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist = []
        rangenum = int(finalsource[3])-int(finalsource[2])        # rangenum is end-start, so there's n-1 frames since there should be n(end-start+1) frames in total in a clip
        

        color_temp = Image.open(io.BytesIO(np.array(color_grp[f"{index+1+int(finalsource[2])}"]))).resize((224,224))
        # depth_temp = Image.open(io.BytesIO(np.array(depth_grp[f"{index+1+int(finalsource[2])}"])))
        color = np.array(color_temp)
        # depth = np.array(depth_temp)
        frame = index+int(finalsource[2])
        odometrylist.append(np.array(odometry_grp[f"odometry{index+int(finalsource[2])}"]))
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
                        handLM[id*2] = lm.x
                        handLM[id*2+1] = lm.x
                        if id == 5:
                            hand[0] = lm.x
                            hand[1] = lm.y
                        id+=1


        gt_xyz = np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
        # transformationsource = np.array(odometry_grp[f"odometry{index+int(finalsource[2])}"])[:3].reshape(-1) #[4, 4]->[3, 4]->[12, ]
        # if gt_xyz[index,2]==0:
        #     gt_xyz[index,2]+=0.000001
        #     print("err")
            # gt_xy[index,0] = (gt_xyz[index,0]*1.80820276e+03/gt_xyz[index,2]+1.94228662e+03)/3840
            # gt_xy[index,1] = (gt_xyz[index,1]*1.80794556e+03/gt_xyz[index,2]+1.12382178e+03)/2160
                                                                                      
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
            # motion[index]=np.concatenate((transformationsource,imudata),0)
        dataset_file.close()
        
        
        return gt_xyz, color, rangenum, finalsource,hand,handLM
 

