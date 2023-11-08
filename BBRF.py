import torch
import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
# from torchvision import models
#from Res import resnet50
from Res import resnet18
# from SwinV2 import SwinTransformerV2
from torch.nn import init
from torch.nn.parameter import Parameter
from DecoderUnit import DecoderRESF
from Attention import HA
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn_mom=0.1, inplace=False):
        # , bn_mom=0.1
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=bn_mom)  # , momentum=bn_mom
        self.relu = nn.ReLU(inplace=inplace)  # inplace=True

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class SDecoder(nn.Module):
    def __init__(self, inplanes, outplanes, shape):
        super(SDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 3, padding=1),
            # nn.BatchNorm2d(outplanes),
            # nn.ReLU(inplace=False),
            nn.LayerNorm(shape, elementwise_affine=True),  # nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.conv = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.LayerNorm(shape, elementwise_affine=True)  #,nn.BatchNorm2d(outplanes)
        self.relu = nn.GELU() #nn.ReLU(inplace=False)
        self.addresult = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, padding=1),
            # nn.BatchNorm2d(outplanes),
            # nn.ReLU(inplace=False),
            nn.LayerNorm(shape, elementwise_affine=True),  # nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

    def forward(self, xx):
        x = self.decoder(xx)
        x2 = self.relu(self.bn(self.conv(x)))
        # y = self.addedge(x, edge, edgeindex) #返回两个值是方便进行残差运算
        result = self.addresult(x2)
        return result


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder5 = SDecoder(512, 256, [22, 22])
        self.decoder4 = SDecoder(256, 128, [44, 44])
        self.decoder3 = SDecoder(128, 64, [88, 88])
        self.decoder2 = SDecoder(64, 64, [176, 176])
        self.way = DecoderRESF(64, 128, 256, 512)
        self.HA1 = HA()
        self.HA2 = HA()
        self.HA3 = HA()

    def forward(self, x0, s1, s2, s3, s4):
        z, zmap = self.way(s1, s2, s3, s4)
        # z = self.HA(torch.sigmoid(zmap), z)
        d = []
        # s5_ = self.aspp(s4)
        # s5_ = F.interpolate(s5_, size=s4.size()[2:], mode='bilinear', align_corners=True)  # 这里是生成一个高级的语义特征，之前的predeliation是不是可以考虑使用新的模块呢？
        z1 = F.interpolate(zmap, size=s4.size()[2:], mode='bilinear', align_corners=True)
        # s42 = self.deliation5(s4)  # 512
        # s43 = s42
        s43 = self.HA1(torch.sigmoid(z1), s4)
        s4_0 = self.decoder5(s43)  # 此时通道维度为1,第一个值没加边缘，第二个值加了边缘
        s4_ = F.interpolate(s4_0, size=s3.size()[2:], mode='bilinear', align_corners=True)  # 采用interpolate函数最主要的作用就是固定了尺寸,此处没有进行尺度的变化就是因为这里
        z2 = F.interpolate(zmap, size=s3.size()[2:], mode='bilinear', align_corners=True)
        # s32 = self.deliation4(s3)
        s33 = s4_ + s3
        s33 = self.HA2(torch.sigmoid(z2), s33)
        # s33_1 = self.toone4(s33)

        s3_0 = self.decoder4(s33)
        s3_ = F.interpolate(s3_0, size=s2.size()[2:], mode='bilinear', align_corners=True)
        z3 = F.interpolate(zmap, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s23 = s3_+s2
        s23 = self.HA3(torch.sigmoid(z3), s23)

        s2_0 = self.decoder3(s23)
        s2_ = F.interpolate(s2_0, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s13 = s2_ + s1

        s1_0 = self.decoder2(s13)
        d.append(s1_0)
        d.append(s2_0)
        d.append(s3_0)
        d.append(s4_0)
        return d, zmap


class BBBNet2(nn.Module):
    def __init__(self, cfg=None):
        super(BBBNet2, self).__init__()
        self.cfg = cfg
        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('/data/Hms/pth/resnet18-5c106cde.pth'), strict=False)   #'/userHome/zy/Hms/BBRF-TIP-master/resnet50-19c8e357.pth'
        self.decode = Decoder()
        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, shape=None, mask=None):
        shape = x.size()[2:] if shape is None else shape
        x0, x1, x2, x3, x4 = self.resnet(x)
        # print(shape)
        # print(x0.shape,x1.shape,x2.shape,x3.shape,x4.shape)
        (pred1, pred2, pred3, pred4), zmap = self.decode(x0, x1, x2, x3, x4)
        pred1 = F.interpolate(self.linear2(pred1), size=shape, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.linear3(pred2), size=shape, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(self.linear4(pred3), size=shape, mode='bilinear', align_corners=True)
        pred4 = F.interpolate(self.linear5(pred4), size=shape, mode='bilinear', align_corners=True)
        zmap = F.interpolate(zmap, size=shape, mode='bilinear', align_corners=True)
        return pred1, pred2, pred3, pred4, zmap
