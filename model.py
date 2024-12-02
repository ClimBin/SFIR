
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *



class SFIR(nn.Module):
    def __init__(self, num_res=6):
        super(SFIR, self).__init__()

        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2 , 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)    # 3,H/2
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # 3,H/4
        
        z2 = self.SCM2(x_2)  #  64, H/2
        z4 = self.SCM1(x_4)  #  128, H/4

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x) # 32, H
        res1 = self.Encoder[0](x_) # 32, H
        # 128
        z = self.feat_extract[1](res1) # 64, H/2
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)  # 64, H/2
        # 64
        z = self.feat_extract[2](res2) # 128, H/4
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z) # 128, H/4

        z = self.Decoder[0](z) # 128, H/4
        z_ = self.ConvsOut[0](z) # 3, H/4
        # 128
        z = self.feat_extract[3](z) # 64, H/2
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def get_model():
    return SFIR()






