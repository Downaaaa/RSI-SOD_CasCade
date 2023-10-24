
"""主要就是利用类似U-Net的结构做级联结构的模型实现"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from torchvision import models
from Res import resnet50
from torch.nn import init
from torch.nn.parameter import Parameter
from FeatureFusion2 import SGFF, SGFF1, SGFF2
from tools import ChannelAttention, SpatialAttention, ShuffleAttention, BasicConv2d, CrissCrossAttention, iAFF
# from unet_model import UNet
from unet_model import UNet
class BodyDetection(nn.Module):
    """
    该类的主要就是想和Edge类一样学习到body，做为参数之一。
    """
    def __init__(self):
        super(BodyDetection, self).__init__()
        self.bodydecoder5 = nn.Sequential(
            BasicConv2d(512, 512, 3, dilation=1, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1))
        self.bodydecoder4 = nn.Sequential(
            BasicConv2d(512+512, 512, 3, dilation=1, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.bodydecoder3 = nn.Sequential(
            BasicConv2d(512+256, 256, 3, dilation=1, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.bodydecoder2 = nn.Sequential(
            BasicConv2d(256 + 128, 128, 3, dilation=1, padding=1),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.bodydecoder1 = nn.Sequential(
            BasicConv2d(128 + 64, 64, 3, dilation=1, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1)
        )
        self.last = nn.Conv2d(64, 1, 3, padding=1)
    def forward(self, x0,x1, x2, x3, x4):
        x4_1 = self.bodydecoder5(x4)
        x4_2 = F.interpolate(x4_1+x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3_1 = self.bodydecoder4(torch.cat((x3, x4_2), 1))
        x3_2 = F.interpolate(x3_1+x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2_1 = self.bodydecoder3(torch.cat((x2, x3_2), 1))
        x2_2 = F.interpolate(x2_1+x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1_1 = self.bodydecoder2(torch.cat((x1, x2_2), 1))
        x1_2 = F.interpolate(x1_1+x1, size=x0.size()[2:], mode='bilinear', align_corners=True)
        x0_1 = self.bodydecoder1(torch.cat((x1_2, x0), 1))
        x_last = self.last(x0_1)
        x_last = torch.sigmoid(x_last)
        return x_last, x0_1, x1_1, x2_1, x3_1, x4_1
# 这里ASPP的使用并不是传统的ASPP的使用方式，只有一个分支，并没有多个分支。
# 设计一个新的金字塔模块使得最后的特征能够表达高级语义特征。
class PN(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(PN, self).__init__()
        # 一般的金字塔模型的设计都是四个膨胀卷积分支，一个1x1分支卷积，相邻分支之间会有元素加的过程。
        """
        设计思路来源于论文：EDN: Salient Object Detection via Extremely-Downsampled Network
        要点如下：
        1、最后一层encoder的输出尺寸较小14-20之间，所以要不要先进行下采样的操作将其尺寸变小，下采样的主要缺点就是会造成信息的损失，最后很有可能会使得细节信息丢失
        2、经过特征融合后的特征通道维度是64，需不需要进行split，然后最后再cat起来，
        3、膨胀卷积的膨胀系数的选择，范围如何确定，padding的加入是否会得到缓解呢。
        4、高级语义特征是为了更好的定位要检测的目标，那么此处得到的高级语义特征的利用还是按照之前的相同操作吗？
        """
        # 版本一：先只使用卷积，先不使用BasicConv2d
        self.conv1 = nn.Sequential(BasicConv2d(in_planes, out_planes, 3, padding=1))
        # 设置四个平行的膨胀卷积
        self.conv2 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 1, padding=0, dilation=1,  inplace=True),
            )
        self.conv3 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=2, dilation=2,  inplace=True),
            )
        self.conv4 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=4, dilation=4,  inplace=True),
            )
        self.conv5 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=8, dilation=8, inplace=True),
            )
        self.conv5 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=12, dilation=12, inplace=True),
        )
        self.conv6 = nn.Sequential(BasicConv2d(5*out_planes, out_planes, 3, padding=1))

    def forward(self, x):
        input = self.conv1(x)
        #(x1, x2, x3, x4) = torch.chunk(input, 4, dim=1)
        brench1 = self.conv2(input)
        brench2 = self.conv3(input+brench1)
        brench3 = self.conv4(input+brench2)
        brench4 = self.conv5(input+brench3)
        brench5 = self.conv5(input+brench4)
        output = torch.cat((brench1, brench2, brench3, brench4, brench5), 1)
        output = self.conv6(output)
        output = output + input
        return output
class Edge(nn.Module):
    #边缘模块主要的操作就是进行边缘的预测然后将输出结果，放到decoder中进行信息增益。
    def __init__(self):
        super(Edge, self).__init__()
        self.decoder5 = nn.Sequential(
            BasicConv2d(512, 512, 3, dilation=1, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1))
        self.decoder4 = nn.Sequential(
            BasicConv2d(512 + 512, 512, 3, dilation=1, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.decoder3 = nn.Sequential(
            BasicConv2d(512 + 256, 256, 3, dilation=1, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            BasicConv2d(256 + 128, 128, 3, dilation=1, padding=1),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.decoder1 = nn.Sequential(
            BasicConv2d(128 + 64, 64, 3, dilation=1, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1)
        )
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.out = nn.Conv2d(64, 1, 3, padding=1)
    def forward(self,z0,z1,z2,z3,z4):
        z4_ = self.decoder5(z4)
        z41 = z4+z4_
        z42 = F.interpolate(z41, size=z3.size()[2:], mode='bilinear', align_corners=True)
        z3_ = self.decoder4(torch.cat((z42, z3),1))
        z31 = z3_+z3
        z32 = F.interpolate(z31, size=z2.size()[2:], mode='bilinear', align_corners=True)
        z2_ = self.decoder3(torch.cat((z32,z2),1))
        z21 = z2_+z2
        z22 = F.interpolate(z21, size=z1.size()[2:], mode='bilinear', align_corners=True)
        z1_ = self.decoder2(torch.cat((z22, z1), 1))
        z11 = z1_+z1
        z12 = F.interpolate(z11, size=z0.size()[2:], mode='bilinear', align_corners=True)
        z0_ = self.decoder1(torch.cat((z12,z0),1))
        z01 = self.out(z0_)
        z02 = torch.sigmoid(z01)
        return z02, z0_, z1_, z2_, z3_, z4_
class SDecoder(nn.Module):
    def __init__(self,inplanes, outplanes):
        super(SDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=False),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=False),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=False),
        )
    def forward(self, xx):
        x = self.decoder(xx)
        result = self.decoder1(x)
        # result = result + x
        result = self.decoder2(result)
        return result
class Decoder3(nn.Module):
    """该decoder和decoder1的区别在于decoder1的目的是想要整合backbone的最终结果然后在此基础上进行下一步的补充所以decoder1的重点是想要融合语义信息、尺度信息等等的一个多信息的特征。
    而该decoder是在级联结构之后将四个级联结构的结果进行融合从而得到想要的结果。将每个级联结构都当做是一个encoder，然后每个级联块的结果都是encoder一个层的输出，然后在做下一步的操作。
    同时注意最后一层的处理顺序，个人认为如果不采用稠密连接的方式的话"""
    def __init__(self):
        super(Decoder3, self).__init__()
        self.decoder4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        # self.hence1 = ShuffleAttention(channel=64, reduction=16, G=8)
        # self.hence2 = ShuffleAttention(channel=128, reduction=16, G=8)
        # self.hence3 = ShuffleAttention(channel=128, reduction=16, G=8)
        # self.hence4 = ShuffleAttention(channel=128, reduction=16, G=8)
        self.hence1 = ChannelAttention(64)
        self.hence2 = ChannelAttention(128)
        self.hence3 = ChannelAttention(128)
        self.hence4 = ChannelAttention(128)
    def forward(self, p1, p2, p3, p4):
        """
        :param p1: 第一个级联的输出
        :param p2: 第二个级联的输出
        :param p3: 第三个级联的输出
        :param p4: 第四个级联的输出
        :return: 四个级联块的融合的结果，一个decoder的设计
        个人问题设想：上面的操作主要问题在于四个级联块的输出融合顺序究竟应该如何排列，个人认为这一点是值得讨论的。
        因为将整个级联块看作encoder部分的话，那么其四个级联块的输出明显是不受监督的
        """
        """第一版本：尝试将p4做为最后一个融合的信息。那么p1就是最先做融合的信息。"""
        """考虑到p1包含的信息比较丰富，该decoder中每一层进行一次注意力增强，注意力的采用重点应该以全局注意力和局部注意力进行结合，感觉简单的使用CA和SA明显是不够的"""
        pred1 = self.hence1(p1) *p1
        pred1 = self.decoder4(pred1)

        pred2 = torch.cat((pred1, p2), 1)
        pred2 = self.hence2(pred2) * pred2
        pred2 = self.decoder3(pred2)

        pred3 = torch.cat((pred2, p3), 1)
        pred3 = self.hence3(pred3) *pred3
        pred3 = self.decoder2(pred3)

        pred4 = torch.cat((pred3, p4), 1)
        pred4 = self.hence4(pred4) * pred4
        pred4 = self.decoder1(pred4)
        return pred1, pred2, pred3, pred4
class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.decoder5 = SDecoder(64, 64)
        self.decoder4 = SDecoder(128, 64)
        self.decoder3 = SDecoder(128, 64)
        self.decoder2 = SDecoder(128, 64)
        self.decoder1 = SDecoder(128, 64)
        self.pn5 = PN(2048, 64)
        self.pn4 = PN(1024, 64)
        self.pn3 = PN(512, 64)
        self.pn2 = PN(256, 64)
        self.pn1 = PN(64, 64)
        self.attn1 = ChannelAttention(64)
        self.attn2 = ChannelAttention(256)
        self.attn3 = ChannelAttention(512)
        self.attn4 = ChannelAttention(1024)
        self.attn5 = ChannelAttention(2048)
    def forward(self,s0,s1, s2, s3, s4):
        s4 = s4*self.attn5(s4)
        s4 = self.pn5(s4)
        s4_0 = self.decoder5(s4)    
        s4_ = F.interpolate(s4_0, size=s3.size()[2:], mode='bilinear', align_corners=True)
        s3 = s3*self.attn4(s3)
        s3 = self.pn4(s3)
        s33 = torch.cat((s4_, s3), 1)

        s3_0 = self.decoder4(s33)
        s3_ = F.interpolate(s3_0, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s2 = s2*self.attn3(s2)
        s2 = self.pn3(s2)
        s23 = torch.cat((s3_, s2), 1)

        s2_0 = self.decoder3(s23)
        s2_ = F.interpolate(s2_0, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s1 = s1*self.attn2(s1)
        s1 = self.pn2(s1)
        s13 = torch.cat((s2_, s1), 1)

        s1_0 = self.decoder2(s13)
        s1_ = F.interpolate(s1_0, size=s0.size()[2:], mode='bilinear', align_corners=True)
        s0 = s0*self.attn1(s0)
        s0 = self.pn1(s0)
        s03 = torch.cat((s1_, s0), 1)

        s0_0 = self.decoder1(s03)
        return s0_0
class BBBNet(nn.Module):
    def __init__(self, cfg=None):
        super(BBBNet, self).__init__()
        self.cfg = cfg
        self.resnet = resnet50()
        self.resnet.load_state_dict(torch.load('/userHome/zy/Hms/BBRF-TIP-master/resnet50-19c8e357.pth'), strict=False)
        self.decode1 = Decoder4()
        self.decoder = Decoder3()
        self.linear0 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.cas1 = UNet(n_channels=64, n_classes=1)
        self.cas2 = UNet(n_channels=64, n_classes=1)
        self.cas3 = UNet(n_channels=64, n_classes=1)
    def forward(self, x,shape=None,mask=None):
        shape = x.size()[2:] if shape is None else shape
        x0, x1, x2, x3, x4 = self.resnet(x)
        p0 = self.decode1(x0, x1, x2, x3, x4)
        hence1 = p0
        mp11, p11 = self.cas1(hence1)
        mp12, p12 = self.cas2(mp11)
        mp13, p13 = self.cas3(mp12)
        pred1,pred2,pred3,pred4 = self.decoder(p0, mp11, mp12, mp13)
        p01 = self.linear0(pred1)
        p11 = self.linear1(pred2)
        p12 = self.linear2(pred3)
        p13 = self.linear3(pred4)
        p1 = F.interpolate(p01, size=shape, mode='bilinear', align_corners=True)
        p2 = F.interpolate(p11, size=shape, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p12, size=shape, mode='bilinear', align_corners=True)
        p4 = F.interpolate(p13, size=shape, mode='bilinear', align_corners=True)
        return p1, p2, p3, p4

