
"""主要就是利用类似U-Net的结构做级联结构的模型实现"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from torchvision import models
from Res import resnet18
from torch.nn import init
from torch.nn.parameter import Parameter
# from unet_model import UNet
from DecoderUnit import DecoderRESF, MapEnhance, DecoderSegNext
from BBRF import BBBNet2
from Attention import HA
# 这里ASPP的使用并不是传统的ASPP的使用方式，只有一个分支，并没有多个分支。
# 设计一个新的金字塔模块使得最后的特征能够表达高级语义特征。
class BBBNet(nn.Module):
    def __init__(self, cfg=None):
        super(BBBNet, self).__init__()
        self.cfg = cfg
        self.resnet = resnet18()
        # self.mscan = MSCAN(
        #     embed_dims=[64, 128, 320, 512],
        #     mlp_ratios=[8, 8, 4, 4],
        #     drop_rate=0.0,
        #     drop_path_rate=0.3,
        #     depths=[3, 5, 27, 3],
        #     norm_cfg=dict(type='BN', requires_grad=True))
        # self.mscan.load_state_dict(torch.load('/data/Hms/pth/mscan_l_20230227-cef260d4.pth')['state_dict'],strict=False)
        self.resnet.load_state_dict(torch.load('/data/Hms/pth/resnet18-5c106cde.pth'), strict=False)
        #/userHome/zy/Hms/BBRF-TIP-master/resnet50-19c8e357.pth
        self.decoderRes = DecoderRESF(64, 128, 256, 512)
        # self.decoderSeg = DecoderSegNext()
        # self.Fenhance1 = MapEnhance()
        # self.Fenhance2 = MapEnhance()
        self.linear01 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        # self.cas1 = UNet(n_channels=3, n_classes=1)
        self.cas1 = BBBNet2()
        self.HA = HA()
    def forward(self, x,shape=None,mask=None):
        shape = x.size()[2:] if shape is None else shape
        x0, x1, x2, x3, x4 = self.resnet(x)
        outp, outmap = self.decoderRes(x2, x3, x4)   # self.decode1(x0, x1, x2, x3, x4)   # 考虑将第一层的监督范围增加
        # 需要注意一点其中的outmap是单通道的，outp是64通道的
        # feature1 = self.Fenhance1(outp, outmap)   # feature1做为下一个层级联结构的输入，所以要将UNet的结构的输入进行更改，现在Feature1的尺寸为H/2 W/2
        # feature1 = F.interpolate(feature1, size=x.shape[2:], mode='bilinear', align_corners=True)
        outmap = F.interpolate(outmap, size=x.shape[2:], mode='bilinear', align_corners=True)
        feature1 = self.HA(torch.sigmoid(outmap), x)
        # feature1 = F.interpolate(feature1, size=x.shape[2:], mode='bilinear', align_corners=True)
        # feature1_new = self.linear01(feature1)
        # input2 = torch.cat((feature1_new, feature1_new), 1)
        # input2 = torch.cat((feature1_new, input2), 1)   # 该部分输入到下一层的特征对于结果是有一定的影响的，个人认为直接64维度的输入好像要好一点，但是对于后续的改动很大
        p2, p11, p12, p13, p14 = self.cas1(feature1)
        # outfu, outmapu = self.cas1(feature1)
        # feature2 = self.Fenhance2(outfu, outmapu)
        #
        # feature2_new = F.interpolate(feature2, size=x.shape[2:], mode='bilinear', align_corners=True)
        # feature2_new = self.linear01(feature2_new)   # 此处转换为3通道完全只是一个尝试
        # l1, l2, l3, l4 = self.mscan(feature2_new)
        # outmaps = self.decoderSeg(l1, l2, l3, l4)
        #
        p1 = F.interpolate(outmap, size=shape, mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=shape, mode='bilinear', align_corners=True)
        # p3 = F.interpolate(outmaps, size=shape, mode='bilinear', align_corners=True)
        p1 = self.linear1(p1)
        p2 = self.linear2(p2)
        # p3 = self.linear3(p3)
        return p1, p2

