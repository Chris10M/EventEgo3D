import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blazebase import BlazeBlock, DecoderConv, Head


class JointRegressor(nn.Module):
    def __init__(self, in_channels, n_joints):
        super(JointRegressor, self).__init__()

        self.j3d_regressor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, n_joints * 3),
        )
        self.n_joints = n_joints
        
    def forward(self, x):
        x = self.j3d_regressor(x)
        x = x.view(-1, self.n_joints, 3)
        
        return x


class BlazePose(nn.Module):
    def __init__(self, config):
        super(BlazePose, self).__init__()

        self.n_joints = config.NUM_JOINTS
        self.inp_chn = config.MODEL.INPUT_CHANNEL

        self._define_layers()

    def _define_layers(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_chn, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.backbone1 = nn.ModuleList([
            BlazeBlock(16, 32, 3),
            BlazeBlock(32, 64, 4),
            BlazeBlock(64, 128, 5),
            BlazeBlock(128, 192, 6),
        ])


        self.heatmap_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])

        self.segmentation_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])

        self.heatmap_head = Head(2 * 32, self.n_joints, activation='relu')
        self.segmentation_head = Head(2 * 32, 1)
                
        self.joint_regressor_hms = JointRegressor(16, self.n_joints)

    def forward(self, x):        
        H, W = x.shape[-2:]      
        B = x.shape[0]           

        x = self.conv1(x)

        feature_maps = []
        for i, layer in enumerate(self.backbone1):
            x = layer(x)
            feature_maps.append(x)

        f5 = feature_maps[-1]
        feature_maps = feature_maps[::-1]
        
        x = f5
        for i, seg in enumerate(self.segmentation_decoder):
            x_seg = seg(x)
            x = torch.cat([x_seg, feature_maps[i]], dim=1)
        
        seg_f = torch.sigmoid(self.segmentation_head(x)) 
                
        seg_f_detached = seg_f.detach().clone()        
        seg_out = F.interpolate(seg_f, size=(H, W), mode='bilinear', align_corners=True)

        x = f5
        hms_outs = []
        for i, hms in enumerate(self.heatmap_decoder):
            x = torch.cat([hms(x), feature_maps[i]], dim=1)
            
            if i >= 1:
                inter_hms_out = F.interpolate(x[:, :self.n_joints], size=(H // 4, W // 4), mode='bilinear', align_corners=True)
                hms_outs.append(inter_hms_out)
            
        hms_out = self.heatmap_head(x)
        
        for i in range(len(hms_outs)):
            hms_out = hms_out + hms_outs[i] 
        hms_out = hms_out / (len(hms_outs) + 1)
        
        j3d = self.joint_regressor_hms(hms_out) 

        return {
            'j3d': j3d,
            'hms': hms_out,
            'seg': seg_out,
            'seg_feature': seg_f_detached
        }