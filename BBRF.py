import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from torchvision import models
from Res import resnet50
from SwinV2 import SwinTransformerV2
from torch.nn import init
from torch.nn.parameter import Parameter

class BodyDetection(nn.Module):
    """
    该类的主要就是想和Edge类一样学习到body，做为参数之一。
    """

    def __init__(self):
        super(BodyDetection, self).__init__()
        self.bodydecoder5 = nn.Sequential(
            BasicConv2d(512, 512, 3, dilation=2, padding=2),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1))
        self.bodydecoder4 = nn.Sequential(
            BasicConv2d(512 + +512 + 512, 512, 3, dilation=2, padding=2),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.bodydecoder3 = nn.Sequential(
            BasicConv2d(512 + 512 + 256, 256, 3, dilation=2, padding=2),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.bodydecoder2 = nn.Sequential(
            BasicConv2d(256 + 256 + 128, 128, 3, dilation=2, padding=2),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.bodydecoder1 = nn.Sequential(
            BasicConv2d(128 + 128 + 64, 64, 3, dilation=2, padding=2),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1)
        )
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x0, x1, x2, x3, x4):
        x4_1 = self.bodydecoder5(x4)
        x4_2 = F.interpolate(torch.cat((x4_1, x4), 1), size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3_1 = self.bodydecoder4(torch.cat((x3, x4_2), 1))
        x3_2 = F.interpolate(torch.cat((x3_1, x3), 1), size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2_1 = self.bodydecoder3(torch.cat((x2, x3_2), 1))
        x2_2 = F.interpolate(torch.cat((x2_1, x2), 1), size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1_1 = self.bodydecoder2(torch.cat((x1, x2_2), 1))  # x1_1是最后的结果，通道数为64
        x1_2 = F.interpolate(torch.cat((x1_1, x1), 1), size=x0.size()[2:], mode='bilinear', align_corners=True)
        x0_1 = self.bodydecoder1(torch.cat((x1_2, x0), 1))
        x_last = self.last(x0_1)
        x_last = torch.sigmoid(x_last)
        return x_last, x0_1, x1_1, x2_1, x3_1, x4_1


# 这里ASPP的使用并不是传统的ASPP的使用方式，只有一个分支，并没有多个分支。
class ASPP_2468(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_2468, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, randx):
        if randx == 0:
            return self.branch1(x)
        if randx == 1:
            return self.branch2(x)
        if randx == 2:
            return self.branch3(x)
        if randx == 3:
            return self.branch4(x)
        return self.branch5(x)


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
        self.conv1 = nn.Sequential(BasicConv2d(in_planes, out_planes, 1, padding=0))
        # 设置四个平行的膨胀卷积
        self.conv2 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 1, padding=0, dilation=1, bn_mom=0.9, inplace=True),
        )
        self.conv3 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=2, dilation=2, bn_mom=0.9, inplace=True),
        )
        self.conv4 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=4, dilation=4, bn_mom=0.9, inplace=True),
        )
        self.conv5 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=8, dilation=8, bn_mom=0.9, inplace=True),
        )
        self.conv5 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=12, dilation=12, bn_mom=0.9, inplace=True),
        )
        self.conv6 = nn.Sequential(BasicConv2d(5 * out_planes, out_planes, 3, padding=1))

    def forward(self, x):
        input = self.conv1(x)
        # (x1, x2, x3, x4) = torch.chunk(input, 4, dim=1)
        brench1 = self.conv2(input)
        brench2 = self.conv3(input + brench1)
        brench3 = self.conv4(input + brench2)
        brench4 = self.conv5(input + brench3)
        brench5 = self.conv5(input + brench4)
        output = torch.cat((brench1, brench2, brench3, brench4, brench5), 1)
        output = self.conv6(output)
        output = output + x
        return output


class Edge(nn.Module):
    # 边缘模块主要的操作就是进行边缘的预测然后将输出结果，放到decoder中进行信息增益。
    def __init__(self):
        super(Edge, self).__init__()
        self.decoder5 = nn.Sequential(
            BasicConv2d(512, 512, 3, dilation=2, padding=2),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1))
        self.decoder4 = nn.Sequential(
            BasicConv2d(512 + 512 + 512, 512, 3, dilation=2, padding=2),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.decoder3 = nn.Sequential(
            BasicConv2d(512 + 512 + 256, 256, 3, dilation=2, padding=2),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            BasicConv2d(256 + 256 + 128, 128, 3, dilation=2, padding=2),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.decoder1 = nn.Sequential(
            BasicConv2d(128 + 128 + 64, 64, 3, dilation=2, padding=2),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, 1, 1, padding=0)

    def forward(self, z0, z1, z2, z3, z4):
        z4_ = self.decoder5(z4)
        z41 = torch.cat((z4, z4_), 1)
        z42 = F.interpolate(z41, size=z3.size()[2:], mode='bilinear', align_corners=True)
        z3_ = self.decoder4(torch.cat((z42, z3), 1))
        z31 = torch.cat((z3_, z3), 1)
        z32 = F.interpolate(z31, size=z2.size()[2:], mode='bilinear', align_corners=True)
        z2_ = self.decoder3(torch.cat((z32, z2), 1))
        z21 = torch.cat((z2_, z2), 1)
        z22 = F.interpolate(z21, size=z1.size()[2:], mode='bilinear', align_corners=True)
        z1_ = self.decoder2(torch.cat((z22, z1), 1))
        z11 = torch.cat((z1_, z1), 1)
        z12 = F.interpolate(z11, size=z0.size()[2:], mode='bilinear', align_corners=True)
        z0_ = self.decoder1(torch.cat((z12, z0), 1))
        z01 = self.out(z0_)
        z02 = torch.sigmoid(z01)
        return z02, z0_, z1_, z2_, z3_, z4_




def FuriesChange(x, y):
    # 该函数的主要工作就是进行傅里叶域的交换，实现跨域的融合,x和y分别是swin以及resnet的结果

    # 2023-7-18改动：先将两个输入进行傅里叶卷积
    # fft_convs1 = FourierUnit(channel, channel)
    # fft_convs2 = FourierUnit(channel, channel)  # 进行傅里叶卷积
    # fft_convs1, fft_convs2 = fft_convs1.cuda(), fft_convs2.cuda()
    # xf = fft_convs1(x)
    # yf = fft_convs2(y)
    # 第一步进行傅里叶变换，得到频谱图
    fx = torch.fft.fftn(x, dim=(-4, -3, -2, -1))
    fy = torch.fft.fftn(y, dim=(-4, -3, -2, -1))

    # 第二步，将频谱进行中心化
    fx_shift = torch.fft.fftshift(fx, dim=(-4, -3, -2, -1))
    fy_shift = torch.fft.fftshift(fy, dim=(-4, -3, -2, -1))
    # print(fx_shift.shape)
    # 第三步，将中心化后的结果得到对应的高低频部分，取最中间的部分的为低频
    fx_low = torch.zeros_like(fx_shift)
    fx_high = fx_shift.clone()
    width1 = fx_shift.shape[2]
    height1 = fx_shift.shape[3]
    width2 = fy_shift.size(2)
    height2 = fy_shift.size(3)
    # print(width2, height2)
    fxlone = fx_shift.clone()
    fxlone[:, :, width1 // 2 - (width1 // 4): width1 // 2 + (width1 // 4),
    height1 // 2 - (height1 // 4):height1 // 2 + (height1 // 4)] = 0  # f1的高频部分
    # f2lone[:, width2//2-(width2//4): width2//2 + (width2//4), height2//2 - (height2//4):height2//2+(height2//4)] = 0
    fx_low[:, :, width1 // 2 - (width1 // 4): width1 // 2 + (width1 // 4),
    height1 // 2 - (height1 // 4):height1 // 2 + (height1 // 4)] = fx_shift[:, :,
                                                                   width1 // 2 - (width1 // 4): width1 // 2 + (
                                                                           width1 // 4),
                                                                   height1 // 2 - (height1 // 4):height1 // 2 + (
                                                                               height1 // 4)]
    fx_high[:, :, :, :] = fxlone  # f2_shift

    fylone = fy_shift.clone()
    fy_low = torch.zeros_like(fy_shift)
    fy_high = fy_shift.clone()
    fylone[:, :, width2 // 2 - (width2 // 4): width2 // 2 + (width2 // 4),
    height2 // 2 - (height2 // 4):height2 // 2 + (height2 // 4)] = 0  # f2的高频部分,将中间低频部分置为0，越往后高频部分占比越小
    fy_low[:, :, width2 // 2 - (width2 // 4): width2 // 2 + (width2 // 4),
    height2 // 2 - (height2 // 4):height2 // 2 + (height2 // 4)] = fy_shift[:, :,
                                                                   width2 // 2 - (width2 // 4): width2 // 2 + (
                                                                               width2 // 4),
                                                                   height2 // 2 - (height2 // 4):height2 // 2 + (
                                                                               height2 // 4)]
    fy_high[:, :, :, :] = fylone

    # 第四步， 进行两者频率域的交换
    fx_new = fx_low + fy_low
    fy_new = fy_high + fx_high  # 此时就完成了频率域的交换
    # 2023-7-17 将融合策略进行更改
    # our_high = (fy_high + fx_high)/2
    # fx_new = fx_low + fy_high
    # fy_new = fy_low + fx_high

    # 第五步，直接反傅里叶变换。从频率域转换到空域
    x_new = torch.fft.ifftn(torch.fft.ifftshift(fx_new, dim=(-4, -3, -2, -1)), dim=(-4, -3, -2, -1)).real
    y_new = torch.fft.ifftn(torch.fft.ifftshift(fy_new, dim=(-4, -3, -2, -1)), dim=(-4, -3, -2, -1)).real
    # 返回增加了傅里叶卷积的内容
    return x_new, y_new  # , xf, yf



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
        # self.fourierconvd = FourierConv2d(outchannel,outchannel,3)
        self.conv1 = nn.Conv2d(inchannel1, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inchannel2, outchannel, kernel_size=1, stride=1, padding=0)

        # self.convcat1 = nn.Sequential(
        #     BasicConv2d(outchannel, outchannel, 3, padding=1, dilation=1)
        # )
        self.convcat1 = nn.Sequential(
            BasicConv2d(4 * outchannel, outchannel, 3, padding=1, dilation=1)
        )
        self.convcat2 = nn.Sequential(
            BasicConv2d(outchannel, outchannel, 3, padding=1, dilation=1)
        )


    def forward(self, x, y):
        # x为swin的输入，y为resnet的输入
        # 第一步，将x，y的尺度统一，利于操作
        # print(x.shape,y.shape)
        x11 = self.conv1(x)
        y1 = self.conv2(y)  # 上面是想统一通道数，注意还要统一尺度
        x1 = F.interpolate(x11, size=y1.size()[2:], mode='bilinear', align_corners=True)  # 这样x2和y1的尺度就是相同的了。那么此时的所有操作基本就是统一的了
        # 第二步，进行频率域的交换
        x_new, y_new = FuriesChange(x1, y1)  # 此时X_new是得到的y的高频x的低频，y_new则是相反的情况 ，2023-7-15做了更改高频和低频分开了
        # x_new, y_new = self.fourierconvd(x1,y1)  # 2023-7-17像其中加入了傅里叶卷积
        # 第二步，将交换后的结果做为权重，交叉相乘与对应相乘结合，个人暂时认为这都有所弊端，这样直接乘的做法貌似是想要将
        x1_new = torch.mul(x1, x_new)
        x2_new = torch.mul(y1, x_new)
        y1_new = torch.mul(y1, y_new)
        y2_new = torch.mul(x1, y_new)
        # 第三步，将上面四者的结果进行cat，比较暴力。最后再将前面之前的原来的结果加进来进一步进行卷积，类似一个残差的过程
        wight = torch.cat((x1_new, x2_new, y1_new, y2_new), 1)
        result1 = self.convcat1(wight)
        reslut2 = x1 + y1 + result1  # torch.cat((x1, y1, result1), 1)   # 做一个类似残差连接的过程，至此所有的操作结束
        result = self.convcat2(reslut2)
        return result



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


class Blur(nn.Module):
    def __init__(self, channel):
        super(Blur, self).__init__()
        # self.first = nn.Sequential(
        #     BasicConv2d(inchannel, channel, 3, padding=1),
        #     )
        # self.decoderpath0 = nn.Sequential(
        #     BasicConv2d(channel, channel, 1, padding=0, dilation=1),
        #     )
        # 2023-9-18改为1，2，3，4 原来为3，5，7，9
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
            BasicConv2d(channel, channel, 5, padding=8, dilation=4, inplace=True),
        )  # 4种膨胀系数
        self.last = nn.Sequential(
            BasicConv2d(channel * 4, channel, 3, padding=1, inplace=True),
        )

    def forward(self, xx):
        # x1_1 = self.conv1(x)
        # xx = self.first(x)
        # x0 = self.decoderpath0(xx)
        x1 = self.decoderpath1(xx)
        # x12 = self.decoderpath12(x+x1)
        x2 = self.decoderpath2(xx + x1)
        x3 = self.decoderpath3(xx + x2)
        x4 = self.decoderpath4(xx + x3)
        xn = self.last(torch.cat((x1, x2, x3, x4), 1))
        xl = xx + xn
        return xl  # 不同扩张卷积的结构结合的结果




class SDecoder(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=False),
        )
        self.conv = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=False)
        self.addresult = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=False),
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
        self.SUM5 = DCAMM(1024, 2048, 512)
        self.SUM4 = DCAMM(1024, 1024, 512)
        self.SUM3 = DCAMM(512, 512, 256)
        self.SUM2 = DCAMM(256, 256, 128)
        self.SUM1 = DCAMM(128, 64, 64)
        self.aspp = PN(512, 512)  # 采用金字塔模型的构造，使得decoder最先的输入是经过一种金字塔的设计的，
        self.deliation5 = Blur(512)
        self.deliation4 = Blur(512)
        self.deliation3 = Blur(256)  # PN(256, 256)#
        self.deliation2 = Blur(128)  # PN(128, 128)#
        self.deliation1 = Blur(64)  # PN(64, 64)#Blur(64)
        self.toone4 = nn.Sequential(
            BasicConv2d(512 + 512, 512, 3, dilation=1, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.toone3 = nn.Sequential(
            BasicConv2d(512 + 256, 256, 3, dilation=1, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.toone2 = nn.Sequential(
            BasicConv2d(256 + 128, 128, 3, dilation=1, padding=1),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.toone1 = nn.Sequential(
            BasicConv2d(128 + 64, 64, 3, dilation=1, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1)
        )
        self.decoder5 = SDecoder(512, 512)
        self.decoder4 = SDecoder(512, 512)
        self.decoder3 = SDecoder(256, 256)
        self.decoder2 = SDecoder(128, 128)
        self.decoder1 = SDecoder(64, 64)

    def forward(self, ss0, ss1, ss2, ss3, ss4, x0, x1, x2, x3, x4):
        s4 = self.SUM5(ss4, x4)  # c-512
        s3 = self.SUM4(ss3, x3)  # c-512
        s2 = self.SUM3(ss2, x2)  # c-256
        s1 = self.SUM2(ss1, x1)
        s0 = self.SUM1(ss0, x0)
        d = []
        """
        注意2023年7月5日的改动，将上面的conv1-conv4都改动了，改动的根源主要是因为这里的采用了cat,已经改回了之前的情况
        7月6日的改动： 不适用body1之外的body特征.
        """
        s5_ = self.aspp(s4)
        s5_ = F.interpolate(s5_, size=s4.size()[2:], mode='bilinear', align_corners=True)  # 这里是生成一个高级的语义特征，之前的predeliation是不是可以考虑使用新的模块呢？

        s42 = self.deliation5(s4)  # 512

        s4_0 = self.decoder5(s42 + s5_)  # 此时通道维度为1,第一个值没加边缘，第二个值加了边缘
        s4_ = F.interpolate(s4_0, size=s3.size()[2:], mode='bilinear', align_corners=True)  # 采用interpolate函数最主要的作用就是固定了尺寸,此处没有进行尺度的变化就是因为这里
        s32 = self.deliation4(s3)
        s33 = torch.cat((s4_, s32), 1)
        s33_1 = self.toone4(s33)

        s3_0 = self.decoder4(s33_1)
        s3_ = F.interpolate(s3_0, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s22 = self.deliation3(s2)
        s23 = torch.cat((s3_, s22), 1)
        s23_1 = self.toone3(s23)

        s2_0 = self.decoder3(s23_1)
        s2_ = F.interpolate(s2_0, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s12 = self.deliation2(s1)
        s13 = torch.cat((s2_, s12), 1)
        s13_1 = self.toone2(s13)

        s1_0 = self.decoder2(s13_1)
        s1_ = F.interpolate(s1_0, size=s0.size()[2:], mode='bilinear', align_corners=True)
        s02 = self.deliation1(s0)
        s03 = torch.cat((s1_, s02), 1)
        s03_1 = self.toone1(s03)

        s0_0 = self.decoder1(s03_1)
        d.append(s0_0)
        d.append(s1_0)
        d.append(s2_0)
        d.append(s3_0)
        d.append(s4_0)
        return d


class BBBNet2(nn.Module):
    def __init__(self, cfg=None):
        super(BBBNet2, self).__init__()
        self.cfg = cfg
        self.bkbone = SwinTransformerV2(img_size=256, window_size=8, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], embed_dim=128)
        self.resnet = resnet50()
        self.bkbone.load_state_dict(
            torch.load('/userHome/zy/Hms/BBRF-TIP-master/checkpoint/swinv2_base_patch4_window8_256.pth')['model'])
        self.resnet.load_state_dict(torch.load('/userHome/zy/Hms/BBRF-TIP-master/resnet50-19c8e357.pth'), strict=False)
        self.decode = Decoder()
        self.linear5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, shape=None, mask=None):
        shape = x.size()[2:] if shape is None else shape
        x0, x1, x2, x3, x4 = self.resnet(x)
        s0, s1, s2, s3, s4 = self.bkbone(x)
        (p0, pred1, pred2, pred3, pred4) = self.decode(s0, s1, s2, s3, s4, x0, x1, x2, x3, x4)
        pred = F.interpolate(self.linear1(p0), size=shape, mode='bilinear')
        # edge = F.interpolate(oute1, size=shape, mode='bilinear')
        pred1 = F.interpolate(self.linear2(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linear3(pred2), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.linear4(pred3), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.linear5(pred4), size=shape, mode='bilinear')
        # body11 = F.interpolate(outbody, size=shape, mode='bilinear')
        return pred, pred1, pred2, pred3, pred4
               # edge, body11  # ,body21,body31,body41,body51   edge11,edge21,edge31,edge41,edge51
