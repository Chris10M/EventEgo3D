import torch
from torch import nn
from torch.nn import functional as F
from .blazepose import BlazePose


class ConfidenceNetwork(nn.Module):
    def __init__(self):
        super(ConfidenceNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=False), 
        )

    def forward(self, key):
        confidence = self.network(key)
       
        return torch.sigmoid(key.detach() * confidence)


class EROS(nn.Module):
    def  __init__(self, inp_chn, kernel_size, initial_decay_base, width, height) -> None:
        super(EROS, self).__init__()
        
        self.confidence_network = ConfidenceNetwork()
        
    def forward(self, buffer, inp, key):     
        _, _, height, width = inp.shape
            
        confidence = self.confidence_network(key)
        confidence = F.interpolate(confidence, size=(height, width), mode='bilinear', align_corners=False)
        
        out = buffer * confidence + inp
        
        old_min, old_max, new_min, new_max = out.min(), out.max(), 0, 1
        out = (out - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


        return out, confidence, buffer


class EgoHPE(nn.Module):
    def __init__(self, config):
        super(EgoHPE, self).__init__()

        self.n_joints = config.NUM_JOINTS
        
        self.blaze_pose = BlazePose(config)

        self.enable_eros = config.EROS
        inp_chn = config.MODEL.INPUT_CHANNEL
        width, height = config.MODEL.IMAGE_SIZE
        self.hm_width, self.hm_height = config.MODEL.HEATMAP_SIZE

        kernal_size = config.DATASET.EROS.KERNEL_SIZE
        decay_base = config.DATASET.EROS.DECAY_BASE

        self.EROS = EROS(inp_chn=inp_chn, kernel_size=kernal_size, height=height, width=width, initial_decay_base=decay_base)
        
    def forward(self, x, prev_buffer=None, prev_key=None, batch_first=False):
        if batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        
        T, B, C, H, W = x.shape

        buffer = prev_buffer
        if buffer is None:
            buffer = torch.zeros_like(x[0, :, :, :, :])
        
        key = prev_key
        if key is None:
            key = torch.ones(B, 1, self.hm_height, self.hm_width).to(x.device)
                    
        eross = []
        x_hmss = []
        j3ds = []
        seg_outs = []

        confidences = []
        buffers = []
        for i in range(T):
            if self.enable_eros:
                out, confidence, buffer = self.EROS(buffer, x[i], key)            
            else:
                out = x[i]

            buffers.append(buffer)
            confidences.append(confidence)
    
            outs = self.blaze_pose(out)
          
            buffer = out
                        
            x_hms = outs['hms']
            j3d = outs['j3d']
                                    
            seg_out = outs['seg']
            seg_feature = outs['seg_feature']
            
            key = seg_feature
            
            eross.append(buffer)
            x_hmss.append(x_hms)
            j3ds.append(j3d)
            seg_outs.append(seg_out)

        if prev_buffer is not None:
            prev_buffer.copy_(buffer)
            
        if prev_key is not None:
            prev_key.copy_(key)

        eross = torch.cat(eross, dim=0)
        x_hmss = torch.cat(x_hmss, dim=0)
        j3ds = torch.cat(j3ds, dim=0)
        seg_outs = torch.cat(seg_outs, dim=0)
         
        confidences = torch.cat(confidences, dim=0)
        buffers = torch.cat(buffers, dim=0)       
        
        outputs = {}
        outputs['j3d'] = j3ds  
        
        outputs['hms'] = x_hmss 
        outputs['eros'] = eross
        outputs['seg'] = seg_outs
 
        outputs['confidence'] = confidences
        outputs['buffer'] = buffers

 
        return outputs

