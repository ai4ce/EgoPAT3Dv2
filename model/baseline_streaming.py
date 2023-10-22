#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as t
import torchvision
from utils.pointconv_util import PointConvDensitySetAbstraction
from utils.rnn_util import PositionalEncoding



'''
RGB Model
'''
def dcon(x):
    resultlist=t.linspace(-1,1,1024*5).cuda()
    x=x/x.max()

    x[t.where(x<=0.5)]=0

    return (x*resultlist).sum()/x.sum()


class Baseline_RGB_Streaming(nn.Module):
    def __init__(self, cfg):
        super(Baseline_RGB_Streaming, self).__init__()
        num_LSTM=2
        midfc_channel=1024
        self.hind=midfc_channel
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


    def forward(self,img, hand, start, hout, cout, cfg,):
        rgb_feature=self.rgb_backbone(img.float())
        hand_feature = self.hand(hand.float())
        feature=self.fine(t.cat((rgb_feature,hand_feature),-1)).unsqueeze(0)
        
        # if cfg.MODEL.ARCH.POS_ENCODING is not None:
        #     # add positional encoding
        #     feature = self.pe(feature, sequences) # sequences is the index of the clip

        if start: # if it's the first frame
            cinit=t.cat((self.initiala1(feature.squeeze(0)).unsqueeze(0),self.initiala2(feature.squeeze(0)).unsqueeze(0),\
                ),0)
            hout=t.cat((self.contin1(feature.squeeze(0)).unsqueeze(0),self.contin2(feature.squeeze(0)).unsqueeze(0),\
                ),0)


            output,(hout,cout)=self.temporalnet(feature,(hout,cinit))
            
        else:
            
            output,(hout,cout)=self.temporalnet(feature,(hout,cout))

        res = t.cat((self.x(output[-1]).unsqueeze(1),self.y(output[-1]).unsqueeze(1),self.z(output[-1]).unsqueeze(1),self.handx(rgb_feature).unsqueeze(1),self.handy(rgb_feature).unsqueeze(1),self.time(output[-1]).unsqueeze(1)),1)


        return res, hout, cout