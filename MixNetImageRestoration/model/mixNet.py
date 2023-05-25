import torch.nn as nn
import torch
from model.block import SEModule, RB, UpDownstream
from utils.img_utils import padding_tensor

class MixSpatial(nn.Module):
    def __init__(self, in_channel, p=0.1):
        super(MixSpatial, self).__init__()
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 7, dilation=5, padding="same", groups=in_channel),
            nn.Dropout2d(p),
            nn.BatchNorm2d(in_channel)
        )  
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 5, dilation=3, padding="same", groups=in_channel),
            nn.Dropout2d(p),
            nn.BatchNorm2d(in_channel)
        )  
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, dilation=1, padding="same", groups=in_channel),
            nn.Dropout2d(p),
            nn.BatchNorm2d(in_channel)
        )  
        self.gelu = nn.GELU()
    def forward(self, x):
        y = self.conv7x7(x) + self.conv5x5(x) + self.conv3x3(x)
        return x + self.gelu(y)

class MixChannel(nn.Module):
    def __init__(self, in_channel, expand_factor=2, reduction = 16):
        super(MixChannel, self).__init__()
        expand_out_channel = in_channel*expand_factor
        self.amplify_channel = nn.Sequential(nn.Conv2d(in_channel, expand_out_channel, 1),
                                    nn.BatchNorm2d(expand_out_channel),
                                    nn.GELU())
        self.se = SEModule(expand_out_channel, reduction=reduction)
        self.mixing_channel = nn.Sequential(nn.Conv2d(expand_out_channel, in_channel, 1),
                                        nn.BatchNorm2d(in_channel),
                                        nn.GELU())
    def forward(self, x):
        y = self.amplify_channel(x)
        y = self.se(y)
        y = self.mixing_channel(y)
        return y + x
    
class DownStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownStage, self).__init__()
        self.patches_merge = nn.Sequential(nn.Conv2d(in_channel, out_channel, 2, 2))
    def forward(self, x):
        return self.patches_merge(x)
    

class MixModule(nn.Module):
    def __init__(self, in_channel, expand_factor=2):
        super().__init__()
        self.mixspa = MixSpatial(in_channel)
        self.mixchan = MixChannel(in_channel, expand_factor)
    
    def forward(self, x):
        x = self.mixspa(x)
        x = self.mixchan(x)
        return x


class MixNet(nn.Module):
    def __init__(self, in_channel = 96, conf_layer_encoder = [2,2,2,2]):
        super(MixNet, self).__init__()
        self.patches_parition = nn.Sequential(nn.Conv2d(3,in_channel, 4, 4), nn.BatchNorm2d(in_channel), nn.GELU()) #H/4xW/4xC
        self.stage1 = nn.Sequential(*[MixModule(in_channel) for i in range(conf_layer_encoder[0])]) #H/4xW/4xC
        self.stage2 = nn.Sequential(*([DownStage(in_channel, in_channel*2)] + [MixModule(in_channel*2) for i in range(conf_layer_encoder[1])])) #H/8xW/8x2C
        self.stage3 = nn.Sequential(*([DownStage(in_channel*2, in_channel*4)] + [MixModule(in_channel*4) for i in range(conf_layer_encoder[2])])) #H/16xW/16x4C
        self.stage4 = nn.Sequential(*([DownStage(in_channel*4, in_channel*8)] + [MixModule(in_channel*8) for i in range(conf_layer_encoder[3])])) #H/32xW/32x8C
#         self.bottle_neck = nn.Sequential(RB(in_channel*8, in_channel*8))
        self.rb1 = nn.Sequential(RB(in_channel*8, in_channel*8), RB(in_channel*8, in_channel*8),\
                                 UpDownstream(2, in_channel*8,in_channel*4))
        self.rb2 = nn.Sequential(RB(2*in_channel*4, in_channel*4), RB(in_channel*4, in_channel*4),\
                                 UpDownstream(2, in_channel*4,in_channel*2))
        self.rb3 = nn.Sequential(RB(2*in_channel*2, in_channel*2), RB(in_channel*2, in_channel*2),\
                                 UpDownstream(2, in_channel*2,in_channel))
        self.rb4 = UpDownstream(4, 2*in_channel, in_channel)
        self.head = nn.Conv2d(in_channel, 3, 1)
        
    def forward(self, x):
        #--------------------------------#
        n, c, h, w = x.shape
        x = padding_tensor(x)
        x = self.patches_parition(x) #H/4xC
        e1 = self.stage1(x) #H/4xC
        e2 = self.stage2(e1) #H/8x2C
        e3 = self.stage3(e2) #H/16x4C
        e4 = self.stage4(e3) #H/32x8C
        #--------------------------------#
        d4 = torch.cat([self.rb1(e4), e3], dim=1)# H/16x4C
        d3 = torch.cat([self.rb2(d4), e2], dim=1) # H/8x2C
        d2 = torch.cat([self.rb3(d3), e1], dim=1) # H/4xC
        d1 = self.rb4(d2) # HxC
        y = self.head(d1) #Hx1
        y = y[:, :, :h, :w]
        return y

if __name__ == "__main__":
    x = torch.rand(2,3,32,32)
    m = MixNet()
    out = m(x)
    print(out.shape)

    