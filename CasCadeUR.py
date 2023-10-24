
"""基本尝试了CA、SA等放的位置，本文件主要是尝试语义引导的重点，在BBDGE3中将中间的语义做了融合结果并没有有所突破，
所以此处换一种思路：不将特定层的特征进行融合而是做一个循环的操作，不断的循环做处理。"""
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
def FuriesChange(x,y):
    # 该函数的主要工作就是进行傅里叶域的交换，实现跨域的融合,x和y分别是swin以及resnet的结果

    # 2023-7-18改动：先将两个输入进行傅里叶卷积
    # fft_convs1 = FourierUnit(channel, channel)
    # fft_convs2 = FourierUnit(channel, channel)  # 进行傅里叶卷积
    # fft_convs1, fft_convs2 = fft_convs1.cuda(), fft_convs2.cuda()
    # xf = fft_convs1(x)
    # yf = fft_convs2(y)
    # 第一步进行傅里叶变换，得到频谱图
    fx = torch.fft.fftn(x, dim=(-4,-3,-2, -1))
    fy = torch.fft.fftn(y, dim=(-4,-3,-2, -1))

    # 第二步，将频谱进行中心化
    fx_shift = torch.fft.fftshift(fx, dim=(-4,-3,-2, -1))
    fy_shift = torch.fft.fftshift(fy, dim=(-4,-3,-2, -1))
    #print(fx_shift.shape)
    # 第三步，将中心化后的结果得到对应的高低频部分，取最中间的部分的为低频
    fx_low = torch.zeros_like(fx_shift)
    fx_high = fx_shift.clone()
    width1 = fx_shift.shape[2]
    height1 = fx_shift.shape[3]
    width2 = fy_shift.size(2)
    height2 = fy_shift.size(3)
    # print(width2, height2)
    fxlone = fx_shift.clone()
    fxlone[:,:, width1 // 2 - (width1 // 4): width1 // 2 + (width1 // 4), height1 // 2 - (height1 // 4):height1 // 2 + (height1 // 4)] = 0  # f1的高频部分
    # f2lone[:, width2//2-(width2//4): width2//2 + (width2//4), height2//2 - (height2//4):height2//2+(height2//4)] = 0
    fx_low[:,:, width1 // 2 - (width1 // 4): width1 // 2 + (width1 // 4), height1 // 2 - (height1 // 4):height1 // 2 + (height1 // 4)] = fx_shift[:,:, width1 // 2 - (width1 // 4): width1 // 2 + (
                                                                               width1 // 4),height1 // 2 - (height1 // 4):height1 // 2 + (height1 // 4)]
    fx_high[:,:, :, :] = fxlone  # f2_shift

    fylone = fy_shift.clone()
    fy_low = torch.zeros_like(fy_shift)
    fy_high = fy_shift.clone()
    fylone[:,:, width2 // 2 - (width2 // 4): width2 // 2 + (width2 // 4),height2 // 2 - (height2 // 4):height2 // 2 + (height2 // 4)] = 0  # f2的高频部分,将中间低频部分置为0，越往后高频部分占比越小
    fy_low[:,:, width2 // 2 - (width2 // 4): width2 // 2 + (width2 // 4),height2 // 2 - (height2 // 4):height2 // 2 + (height2 // 4)] = fy_shift[:,:,
                                                                   width2 // 2 - (width2 // 4): width2 // 2 + (width2 // 4),height2 // 2 - (height2 // 4):height2 // 2 + (height2 // 4)]
    fy_high[:, :, :,:] = fylone

    # 第四步， 进行两者频率域的交换
    fx_new = fx_low + fy_low
    fy_new = fy_high + fx_high      # 此时就完成了频率域的交换
    # 2023-7-17 将融合策略进行更改
    #our_high = (fy_high + fx_high)/2
    # fx_new = fx_low + fy_high
    # fy_new = fy_low + fx_high

    # 第五步，直接反傅里叶变换。从频率域转换到空域
    x_new = torch.fft.ifftn(torch.fft.ifftshift(fx_new, dim=(-4,-3,-2, -1)), dim=(-4,-3,-2, -1)).real
    y_new = torch.fft.ifftn(torch.fft.ifftshift(fy_new, dim=(-4,-3,-2, -1)), dim=(-4,-3,-2, -1)).real
    #返回增加了傅里叶卷积的内容
    return x_new, y_new #, xf, yf
class DCAMA(nn.Module):
    def __init__(self, inchannel1, outchannel):
        super(DCAMA, self).__init__()
        self.conv = nn.Sequential(BasicConv2d(inchannel1, outchannel, 3, padding=1),)
        # self.attn = CrissCrossAttention(outchannel)
        # self.attn2 = CrissCrossAttention(outchannel)
        self.PN = Blur(outchannel)
    def forward(self,x):
        result = self.conv(x)
        result = self.PN(result)
        # result = self.attn(result)
        # result = self.attn(result)

        return result
class DCAMM(nn.Module):
    """该类的重点就是想要将resnet和transformer两个分支的结果进行融合
    原论文是从信道以及空间两个上面的不足来分析的,两个不同的输入进行互补，或者说是进行不同的特殊处理，得到一定的补充然后合在一起。
    个人认为该模块是文中最重要的一个模块，其设计要至少保证两个输入一个输出。这样才能方便后续的模块的训练。同时进行特征融合也是一个关键的问题
    查看了一些文献，比较出名的就是Attentional Featrue Fusion WACV 2021的一篇论文,其核心就是想设定注意力权重，有一定的参考价值，但是现今还没有完整的设计思路。
    该部分进行特征融合主要的区别在于，有两个不同分支的数据要融合，简单的融合策略不能很好的得到好的信息，所以现在需要一个好的特征融合策略。"""
    """2023-7-12设计思路：利用傅里叶变换将transformer和resnet的结果进行高低频的交换，然后再乘之前的空域的特征使得保留一定的特征，结果cat后进行卷积。"""
    def __init__(self, inchannel1, inchannel2, outchannel):
        # 第一个参数是swin的输入通道，第二个参数是resnet的输入通道，最后的参数是最终的输出通道。
        super(DCAMM, self).__init__()
        #self.fourierconvd = FourierConv2d(outchannel,outchannel,3)
        self.conv1 = nn.Conv2d(inchannel1, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inchannel2, outchannel, kernel_size=1, stride=1, padding=0)

        # self.convcat1 = nn.Sequential(
        #     BasicConv2d(outchannel, outchannel, 3, padding=1, dilation=1)
        # )
        self.convcat1 = nn.Sequential(
            BasicConv2d(4*outchannel, outchannel, 3, padding=1, dilation=1)
        )
        self.convcat2 = nn.Sequential(
            BasicConv2d(outchannel, outchannel, 3, padding=1, dilation=1)
        )
        # self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        # self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        #self.se = SEModel(3*outchannel)
    def forward(self, x, y):
        # x为swin的输入，y为resnet的输入
        # 第一步，将x，y的尺度统一，利于操作
        #print(x.shape,y.shape)
        x11 = self.conv1(x)
        y1 = self.conv2(y)   #上面是想统一通道数，注意还要统一尺度
        x1 = F.interpolate(x11, size=y1.size()[2:], mode='bilinear', align_corners=True)  #这样x2和y1的尺度就是相同的了。那么此时的所有操作基本就是统一的了
        # 第二步，进行频率域的交换
        x_new, y_new = FuriesChange(x1, y1)  # 此时X_new是得到的y的高频x的低频，y_new则是相反的情况 ，2023-7-15做了更改高频和低频分开了
        #x_new, y_new = self.fourierconvd(x1,y1)  # 2023-7-17像其中加入了傅里叶卷积
        # 第二步，将交换后的结果做为权重，交叉相乘与对应相乘结合，个人暂时认为这都有所弊端，这样直接乘的做法貌似是想要将
        x1_new = torch.mul(x1, x_new)
        x2_new = torch.mul(y1, x_new)
        y1_new = torch.mul(y1, y_new)
        y2_new = torch.mul(x1, y_new)
        # xf_new = torch.mul(xf,x1)
        # yf_new = torch.mul(yf,y1)    # 将两个傅里叶卷积的结果单独提出来，进行信息增强。
        # xf2 = torch.mul(xf,y1)
        # yf2 = torch.mul(yf,x1)       # 这里先尝试交叉乘的操作
        # 第三步，将上面四者的结果进行cat，比较暴力。最后再将前面之前的原来的结果加进来进一步进行卷积，类似一个残差的过程
        wight = torch.cat((x1_new, x2_new, y1_new, y2_new), 1)
        #wight = torch.cat((x1_new, x2_new, y1_new, y2_new,xf_new,yf_new,xf2,yf2), 1)
        #wight = torch.cat((x2_new + y1, y2_new+x1), 1)
        # resulty = x2_new + y1_new + y1
        # resultx = x1_new + y2_new + x1
        # resulty = torch.cat((x2_new, y1_new,y1),1)
        # resultx = torch.cat((x1_new,y2_new, x1),1)
        #result1 = torch.cat((resultx,resulty),1)
        #result1 = resultx + resulty
        result1 = self.convcat1(wight)
        reslut2 = x1+y1+result1#torch.cat((x1, y1, result1), 1)   # 做一个类似残差连接的过程，至此所有的操作结束
        #reslut3 = self.se(result1)
        #result = self.convcat2(result1+reslut3)
        result = self.convcat2(reslut2)
        return result

class DCAM(nn.Module):
    def __init__(self,inc1,inc2,outc):  # 512 2048 512
        super(DCAM, self).__init__()
        self.conv1 = nn.Conv2d(inc1, inc2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inc2, inc1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(inc1, outc, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(inc2, outc, kernel_size=1, stride=1, padding=0)
        self.convAB1 = nn.Conv2d(inc1+inc2, outc, kernel_size=3, stride=1, padding=1)
        self.convAB2 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)
        self.bnAB1 = nn.BatchNorm2d(outc)
        self.bnAB2 = nn.BatchNorm2d(outc)
        self.convAB3 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)
        self.convAB4 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)
        self.bnAB3 = nn.BatchNorm2d(outc)
        self.bnAB4 = nn.BatchNorm2d(outc)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(inc1, inc1, 1)
        self.sftmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        # x为swin , y为resnet
        yy = F.interpolate(self.conv1(x), size=y.size()[2:], mode='bilinear', align_corners=True)
        xx = self.convg(self.gap(self.conv2(y)))
        xx = torch.mul(self.sftmax(xx) * xx.shape[1], x)    #CA
        xx = F.interpolate(xx, size=y.size()[2:], mode='bilinear', align_corners=True)
        yy = y*F.sigmoid(yy)    #SA
        ans = torch.cat((xx,yy),1)
        ans = F.relu(self.bnAB1(self.convAB1(ans)), inplace=True)
        ans = ans+self.conv4(y)+F.interpolate(self.conv3(x), size=y.size()[2:], mode='bilinear', align_corners=True)
        ans = F.relu(self.bnAB2(self.convAB2(ans)), inplace=True)
        #print(ans.shape)
        return ans
class InhibitBackground(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InhibitBackground, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
             nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
             nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
             nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
             nn.Conv2d(out_channel, 1, 3, padding=1), )
        self.channel = out_channel

    def forward(self, x, y):

        """y可以理解为上一个的输出
        x可以理解为对应decoder的输入
        最后的结果还要加上y,即是上一层的输出。"""
        a = 1 - torch.sigmoid(y)
        x1 = self.relu(self.bn(self.convert(x)))
        x = a.expand(-1, self.channel, -1, -1).mul(x1)   # 此时已经得到了相应的背景和当前输入相乘的权重，感觉下面将y加进去没有多大的意义。这样乘出来的结果更加的偏向于突出边缘
        #x = x1 + self.convs(x1)
        #x = x + x1
        return x

class Blur(nn.Module):
    def __init__(self, channel):
        super(Blur, self).__init__()
        # self.first = nn.Sequential(
        #     BasicConv2d(inchannel, channel, 3, padding=1),
        #     )
        # self.decoderpath0 = nn.Sequential(
        #     BasicConv2d(channel, channel, 1, padding=0, dilation=1),
        #     )
        #2023-9-18改为1，2，3，4 原来为3，5，7，9
        self.decoderpath1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=2, dilation=2, inplace=True),
            )
        self.decoderpath2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=4, dilation=4, inplace=True),
            )
        self.decoderpath3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=6, dilation=6, inplace=True),
            )
        self.decoderpath4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=8, dilation=8, inplace=True),
        )     # 4种膨胀系数
        self.last = nn.Sequential(
            BasicConv2d(channel * 4, channel, 3, padding=1, inplace=True),
            )
    def forward(self,xx):
        x1 = self.decoderpath1(xx)
        x2 = self.decoderpath2(xx+x1)
        x3 = self.decoderpath3(xx+x2)
        x4 = self.decoderpath4(xx+x3)
        xn = self.last(torch.cat((x1, x2, x3, x4), 1))
        xl = xx + xn
        return xl     #不同扩张卷积的结构结合的结果

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.attn5 = iAFF(512)
        # self.attn4 = iAFF(512)
        # self.attn3 = iAFF(256)
        # self.attn2 = iAFF(128)
        # self.attn1 = iAFF(64)
        self.decoder5 = SDecoder(512, 512)
        self.decoder4 = SDecoder(512+512, 256)
        self.decoder3 = SDecoder(256+256, 128)
        self.decoder2 = SDecoder(128+128, 64)
        self.decoder1 = SDecoder(64+64, 64)
        # self.last = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        # )
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.PN5 = Blur(512+512)
        # self.PN4 = Blur(512+512)
        # self.PN3 = Blur(512+256)
        # self.PN2 = Blur(256+128)
        # self.PN1 = Blur(128+64)
        # self.up1 = nn.Conv2d(256, 512, 1, padding=0)
        # self.up2 = nn.Conv2d(128, 256, 1, padding=0)
        # self.up3 = nn.Conv2d(64, 128, 1, padding=0)
        # self.outconv5 = nn.Conv2d(256, 1, 3, padding=1)
        # self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        # self.outconv3 = nn.Conv2d(256, 1, 3, padding=1)
        # self.outconv2 = nn.Conv2d(256, 1, 3, padding=1)
        # self.outconv1 = nn.Conv2d(256, 1, 3, padding=1)
        # self.pn = PN(512, 512)
    def forward(self,s0,s1, s2, s3, s4):
        # s4 = s4 * self.attn5(s4)
        # s4 = s4 * self.atts5(s4)
        # s3 = s3 * self.attn4(s3)
        # s3 = s3 * self.atts4(s3)
        # s2 = s2 * self.attn3(s2)
        # s2 = s2 * self.atts3(s2)
        # s1 = s1 * self.attn2(s1)
        # s1 = s1 * self.atts2(s1)
        # s0 = s0 * self.attn1(s0)
        # s0 = s0 * self.atts1(s0)
        d = []
        """
        注意2023年7月5日的改动，将上面的conv1-conv4都改动了，改动的根源主要是因为这里的采用了cat,已经改回了之前的情况
        7月6日的改动： 不适用body1之外的body特征.
        """
        # s5_ = self.pn(s4)
        # s5_ = F.interpolate(s5_, size=s4.size()[2:], mode='bilinear', align_corners=True)    # 这里是生成一个高级的语义特征，之前的predeliation是不是可以考虑使用新的模块呢？
        # s43 = torch.cat((s5_, s4), 1)

        s4_0 = self.decoder5(s4)    # 此时通道维度为1,第一个值没加边缘，第二个值加了边缘
        s4_ = F.interpolate(s4_0, size=s3.size()[2:], mode='bilinear', align_corners=True)  # 采用interpolate函数最主要的作用就是固定了尺寸,此处没有进行尺度的变化就是因为这里
        s33 = torch.cat((s4_, s3), 1)

        s3_0 = self.decoder4(s33)
        # outs3 = self.outconv4(s3_0)
        s3_ = F.interpolate(s3_0, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s23 = torch.cat((s2, s3_), 1)

        s2_0 = self.decoder3(s23)
        s2_ = F.interpolate(s2_0, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s13 = torch.cat((s2_,s1),1)

        s1_0 = self.decoder2(s13)
        s1_ = F.interpolate(s1_0, size=s0.size()[2:], mode='bilinear', align_corners=True)
        s03 = torch.cat((s0,s1_),1)

        s0_0 = self.decoder1(s03)
        d.append(s0_0)
        d.append(s1_0)
        d.append(s2_0)
        d.append(s3_0)
        # d.append(s4_0)
        return d
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.encoder4 = SDecoder(64,  128)
        self.encoder3 = SDecoder(128+128, 256)
        self.encoder2 = SDecoder(256+256, 512)
        self.encoder1 = SDecoder(512+512, 512)
        self.decoder4 = SDecoder(1024, 512)
        self.decoder3 = SDecoder(512+512, 256)
        self.decoder2 = SDecoder(256+256, 128)
        self.decoder1 = SDecoder(128+128, 64)
        self.last = nn.Sequential(
                nn.Conv2d(1024, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=False),
        )

    def forward(self, x, p1, p2, p3, p4):
            d = []
            """
            注意2023年7月5日的改动，将上面的conv1-conv4都改动了，改动的根源主要是因为这里的采用了cat,已经改回了之前的情况
            7月6日的改动： 不适用body1之外的body特征.
            """
            x4 = self.encoder4(x)
            x3 = F.max_pool2d(x4, kernel_size=3, stride=2, padding=1)
            p1 = F.interpolate(p1, size=x3.size()[2:], mode='bilinear', align_corners=True)
            x3 = self.encoder3(torch.cat((x3, p1),1))
            x2 = F.max_pool2d(x3, kernel_size=3, stride=2, padding=1)
            p2 = F.interpolate(p2, size=x2.size()[2:], mode='bilinear', align_corners=True)
            x2 = self.encoder2(torch.cat((x2, p2),1))
            x1 = F.max_pool2d(x2, kernel_size=3, stride=2, padding=1)
            p3 = F.interpolate(p3, size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = self.encoder1(torch.cat((x1, p3),1))
            d1 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1)
            p4 = F.interpolate(p4, size=d1.size()[2:], mode='bilinear', align_corners=True)
            d1 = self.last(torch.cat((d1, p4),1))
            d1_ = F.interpolate(d1, size=x1.size()[2:], mode='bilinear', align_corners=True)
            d2 = self.decoder4(torch.cat((d1_, x1), 1))
            d2_ = F.interpolate(d2, size=x2.size()[2:], mode='bilinear', align_corners=True)
            d3 = self.decoder3(torch.cat((d2_, x2), 1))
            d3_ = F.interpolate(d3, size=x3.size()[2:], mode='bilinear', align_corners=True)
            d4 = self.decoder2(torch.cat((d3_, x3), 1))
            d4_ = F.interpolate(d4, size=x4.size()[2:], mode='bilinear', align_corners=True)
            d5 = self.decoder1(torch.cat((d4_, x4), 1))
            d.append(d5)
            d.append(d4)
            d.append(d3)
            d.append(d2)
            d.append(d1)
            return d

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
        因为将整个级联块看作encoder部分的话，那么其四个级联块的输出明显是不受监督的这一点是需要注意的一点
        """
        """第一版本：尝试将p4做为最后一个融合的信息。那么p1就是最先做融合的信息。"""
        """考虑到p1包含的信息比较丰富，该decoder中每一层进行一次注意力进行增强，注意力的采用重点应该以全局注意力和局部注意力进行结合，当然这只是个人的见解，应为感觉简单的使用CA和SA明显是不够的"""
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
        # d = []
        """
        注意2023年7月5日的改动，将上面的conv1-conv4都改动了，改动的根源主要是因为这里的采用了cat,已经改回了之前的情况
        7月6日的改动： 不适用body1之外的body特征.
        """
        s4 = s4*self.attn5(s4)
        s4 = self.pn5(s4)
        s4_0 = self.decoder5(s4)    # 此时通道维度为1,第一个值没加边缘，第二个值加了边缘
        s4_ = F.interpolate(s4_0, size=s3.size()[2:], mode='bilinear', align_corners=True)  # 采用interpolate函数最主要的作用就是固定了尺寸,此处没有进行尺度的变化就是因为这里
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
        # d.append(s0_0)
        # d.append(s1_0)
        # d.append(s2_0)
        # d.append(s3_0)
        # # d.append(s4_0)
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
        pred1,pred2,pred3,pred4 = self.decoder(mp13, mp12, mp11, p0)
        p01 = self.linear0(pred1)
        p11 = self.linear1(pred2)
        p12 = self.linear2(pred3)
        p13 = self.linear3(pred4)
        p1 = F.interpolate(p01, size=shape, mode='bilinear', align_corners=True)
        p2 = F.interpolate(p11, size=shape, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p12, size=shape, mode='bilinear', align_corners=True)
        p4 = F.interpolate(p13, size=shape, mode='bilinear', align_corners=True)
        return p1, p2, p3, p4

