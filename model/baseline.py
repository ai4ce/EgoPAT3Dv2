#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as t
import torchvision
from utils.pointconv_util import PointConvDensitySetAbstraction
from utils.rnn_util import PositionalEncoding


'''
PointCloud Model
'''
class pointconvbackbone(nn.Module):
    def __init__(self):
        super(pointconvbackbone, self).__init__()
        feature_dim = 3
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        return x

class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()
        num_LSTM=2
        midfc_channel=1024
        self.hind=midfc_channel
            
        self.backbone=pointconvbackbone()
        self.mlp_semantic=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel),
            nn.BatchNorm1d(midfc_channel),
            nn.ReLU(),
            nn.Linear(midfc_channel, midfc_channel),

        )
        if cfg.MODEL.ARCH.POINTCLOUD_MOTION:
            self.mlp_geometry=nn.Sequential(
                nn.Linear(18, midfc_channel*2),
                nn.BatchNorm1d(midfc_channel*2),
                nn.ReLU(),
                nn.Linear(midfc_channel*2, midfc_channel)
        )
            self.fine=nn.Sequential(
                nn.Linear(midfc_channel*2, midfc_channel),
                    nn.BatchNorm1d(midfc_channel),
                    nn.ReLU(),               
                    nn.Linear(midfc_channel, midfc_channel),)
        
        self.temporalnet=nn.LSTM(midfc_channel,midfc_channel,num_LSTM)

        self.initiala1=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )
        self.initiala2=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )



        self.contin1=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*3),
            nn.BatchNorm1d(midfc_channel*3),
            nn.ReLU(),
            nn.Linear(midfc_channel*3, midfc_channel)
        )
        self.contin2=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*3),
            nn.BatchNorm1d(midfc_channel*3),
            nn.ReLU(),
            nn.Linear(midfc_channel*3, midfc_channel)
        )


        

        self.x=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        self.y=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        self.z=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        
        # position encoding
        if cfg.MODEL.ARCH.POS_ENCODING is not None:
            self.pe = PositionalEncoding(d_model=midfc_channel)


        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight(), gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0)  


    def forward(self, pointxyz,pointfeat,motion,LEGHTN, cfg):

        predictlist=[]

        for sequences in range(int(LEGHTN[0])):
            eachsequences_feature=self.backbone(pointxyz[:,sequences,:,:].float(),pointfeat[:,sequences,:,:].float())
            sematic_feature = self.mlp_semantic(eachsequences_feature)

            if cfg.MODEL.ARCH.POINTCLOUD_MOTION:
                geometry_feature = self.mlp_geometry(motion[:,sequences,:].float())
                feature=self.fine(t.cat((sematic_feature,geometry_feature),-1)).unsqueeze(0)
            else:
                feature=sematic_feature.unsqueeze(0)


            if cfg.MODEL.ARCH.POS_ENCODING is not None:
                # add positional encoding
                feature = self.pe(feature, sequences) # sequences is the index of the clip

            if sequences==0:
                cinit=t.cat((self.initiala1(feature.squeeze(0)).unsqueeze(0),self.initiala2(feature.squeeze(0)).unsqueeze(0),\
                    ),0)
                hout=t.cat((self.contin1(feature.squeeze(0)).unsqueeze(0),self.contin2(feature.squeeze(0)).unsqueeze(0),\
                 ),0)


                output,(hout,cout)=self.temporalnet(feature,(hout,cinit))
                
            else:
                
                output,(hout,cout)=self.temporalnet(feature,(hout,cout))

            predictlist.append(t.cat((self.x(output[-1]).unsqueeze(1),self.y(output[-1]).unsqueeze(1),self.z(output[-1]).unsqueeze(1)),1))
        
        return predictlist
    


'''
RGB Model
'''
def dcon(x):
    resultlist=t.linspace(-1,1,1024*5).cuda()
    x=x/x.max()

    x[t.where(x<=0.5)]=0

    return (x*resultlist).sum()/x.sum()


class Baseline_RGB(nn.Module):
    def __init__(self, cfg):
        super(Baseline_RGB, self).__init__()
        num_LSTM=2
        midfc_channel=1024
        self.hind=midfc_channel

        # RGB backbone selection
        if 'ResNet' in cfg.MODEL.ARCH.RGB_BACKBONE:
            if cfg.MODEL.ARCH.RGB_BACKBONE == 'ResNet50':
                self.rgb_backbone = torchvision.models.resnet50(pretrained=True)
            elif cfg.MODEL.ARCH.RGB_BACKBONE == 'ResNet18':
                self.rgb_backbone = torchvision.models.resnet18(pretrained=True)
            elif cfg.MODEL.ARCH.RGB_BACKBONE == 'ResNet34':
                self.rgb_backbone = torchvision.models.resnet34(pretrained=True)
            elif cfg.MODEL.ARCH.RGB_BACKBONE == 'ResNet101':
                self.rgb_backbone = torchvision.models.resnet101(pretrained=True)
            elif cfg.MODEL.ARCH.RGB_BACKBONE == 'ResNet152':
                self.rgb_backbone = torchvision.models.resnet152(pretrained=True)
            else:
                raise NotImplementedError
            
            self.rgb_backbone.fc = nn.Linear(self.rgb_backbone.fc.in_features, midfc_channel)

            with t.no_grad():
                pretrained_conv1 = self.rgb_backbone.conv1.weight.clone()
                self.rgb_backbone.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
                nn.init.kaiming_normal_(self.rgb_backbone.conv1.weight,mode='fan_out',nonlinearity='relu')
                self.rgb_backbone.conv1.weight[:,:3]=pretrained_conv1

        elif 'ConvNext' in cfg.MODEL.ARCH.RGB_BACKBONE:
            if cfg.MODEL.ARCH.RGB_BACKBONE == 'ConvNext_Tiny':
                self.rgb_backbone = torchvision.models.convnext_tiny(pretrained=True)
            elif cfg.MODEL.ARCH.RGB_BACKBONE == 'ConvNext_Small':
                self.rgb_backbone = torchvision.models.convnext_small(pretrained=True)
            elif cfg.MODEL.ARCH.RGB_BACKBONE == 'ConvNext_Base':
                self.rgb_backbone = torchvision.models.convnext_base(pretrained=True)
            else:
                raise NotImplementedError
            num_ftrs = self.rgb_backbone.classifier[2].in_features
            self.rgb_backbone.classifier[2] = nn.Linear(int(num_ftrs), midfc_channel) # pylance is throwing a warning here, but it's fine
        
        else:
            raise NotImplementedError
        
        self.temporalnet=nn.LSTM(midfc_channel,midfc_channel,num_LSTM)


        if cfg.MODEL.ARCH.WITH_HAND:
            # if we are processing the hand priors, then we need a hand MLP and a fusion MLP
            self.hand=nn.Sequential(
                nn.Linear(42, midfc_channel*2),
                nn.BatchNorm1d(midfc_channel*2),
                nn.ReLU(),
                nn.Linear(midfc_channel*2, midfc_channel)
            )

            self.fine=nn.Sequential(
                nn.Linear(midfc_channel*2, midfc_channel),
                    nn.BatchNorm1d(midfc_channel),
                    nn.ReLU(),               
                    nn.Linear(midfc_channel, midfc_channel),)



        self.initiala1=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )
        self.initiala2=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )

        self.contin1=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*3),
            nn.BatchNorm1d(midfc_channel*3),
            nn.ReLU(),
            nn.Linear(midfc_channel*3, midfc_channel)
        )
        self.contin2=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*3),
            nn.BatchNorm1d(midfc_channel*3),
            nn.ReLU(),
            nn.Linear(midfc_channel*3, midfc_channel)
        )
        

        self.x=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        self.y=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        self.z=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        
        if cfg.TRAINING.LOSS == 'RGB_Ori':
            self.handx=nn.Sequential(
                nn.Linear(midfc_channel, midfc_channel*5),
                nn.BatchNorm1d(midfc_channel*5),
                nn.ReLU(),
                nn.Linear(midfc_channel*5, 1024*5)
                )
            self.handy=nn.Sequential(
                nn.Linear(midfc_channel, midfc_channel*5),
                nn.BatchNorm1d(midfc_channel*5),
                nn.ReLU(),
                nn.Linear(midfc_channel*5, 1024*5)
                )
            self.time=nn.Sequential(
                nn.Linear(midfc_channel, midfc_channel*5),
                nn.BatchNorm1d(midfc_channel*5),
                nn.ReLU(),
                nn.Linear(midfc_channel*5, 1024*5)
                )
        
        # position encoding
        if cfg.MODEL.ARCH.POS_ENCODING is not None:
            self.pe = PositionalEncoding(d_model=midfc_channel)

        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight(), gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0)  


    def forward(self,img, hand, LEGHTN, cfg):

        predictlist=[]
        
        for sequences in range(int(LEGHTN[0])):

            rgb_feature=self.rgb_backbone(img[:,sequences,:,:].float())
            if cfg.MODEL.ARCH.WITH_HAND:
                # if we are using hand priors, then we need to concatenate the hand priors with the RGB features
                hand_feature = self.hand(hand[:,sequences,:].float())
                feature=self.fine(t.cat((rgb_feature,hand_feature),-1)).unsqueeze(0)
            else:
                feature=rgb_feature.unsqueeze(0)

            
            if cfg.MODEL.ARCH.POS_ENCODING is not None:
                # add positional encoding
                feature = self.pe(feature, sequences) # sequences is the index of the clip

            if sequences==0:
                # the first LSTM input needs some initialization trick
                cinit=t.cat((self.initiala1(feature.squeeze(0)).unsqueeze(0),self.initiala2(feature.squeeze(0)).unsqueeze(0),\
                    ),0)
                hout=t.cat((self.contin1(feature.squeeze(0)).unsqueeze(0),self.contin2(feature.squeeze(0)).unsqueeze(0),\
                 ),0)
                output,(hout,cout)=self.temporalnet(feature,(hout,cinit))
            else:
                output,(hout,cout)=self.temporalnet(feature,(hout,cout))

            if cfg.TRAINING.LOSS == 'Ori':
                res = t.cat((self.x(output[-1]).unsqueeze(1),self.y(output[-1]).unsqueeze(1),self.z(output[-1]).unsqueeze(1)),1)
            elif cfg.TRAINING.LOSS == 'RGB_Ori':
                res = t.cat((self.x(output[-1]).unsqueeze(1),self.y(output[-1]).unsqueeze(1),self.z(output[-1]).unsqueeze(1),self.handx(rgb_feature).unsqueeze(1),self.handy(rgb_feature).unsqueeze(1),self.time(output[-1]).unsqueeze(1)),1)
            else:
                raise NotImplementedError
            predictlist.append(res)

        return predictlist