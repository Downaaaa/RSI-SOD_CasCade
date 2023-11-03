import torch
import torch.nn as nn
from tools import BasicConv2d
import torch.nn.functional as F


class PN(nn.Module):
    def __init__(self, out_planes):
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
        # 设置四个平行的膨胀卷积
        self.conv2 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 1, padding=0, dilation=1, inplace=True),
        )
        self.conv3 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=2, dilation=2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=4, dilation=4, inplace=True),
        )
        self.conv5 = nn.Sequential(
            BasicConv2d(out_planes, out_planes, 3, padding=6, dilation=6, inplace=True),
        )
        self.conv6 = nn.Sequential(BasicConv2d(4 * out_planes, out_planes, 3, padding=1))

    def forward(self, input):
        brench1 = self.conv2(input)
        brench2 = self.conv3(input + brench1)
        brench3 = self.conv4(input + brench2)
        brench4 = self.conv5(input + brench3)

        output = torch.cat((brench1, brench2, brench3, brench4), 1)
        output = self.conv6(output)
        output = output + input
        return output


class DecoderRES(nn.Module):
    """
    日期：2023-10-30
    作用：重新设计Decoder，使得监督的范围缩小同时减弱底层特征的考虑，只是backbone为ResNet的Decoder的设计
    作者：Amos
    """

    def __init__(self, channel1, channel2, channel3):
        super(DecoderRES, self).__init__()
        self.BM1 = BasicConv2d(channel1, 64, 3, padding=1)
        self.BM2 = BasicConv2d(channel2, 64, 3, padding=1)
        self.BM3 = BasicConv2d(channel3, 64, 3, padding=1)
        # 将其降到64通道是为了减少计算量
        self.FM3 = BasicConv2d(64, 64, 3, padding=1)
        self.FM2 = BasicConv2d(64, 64, 3, padding=1)
        self.FM1 = BasicConv2d(64, 64, 3, padding=1)
        self.PN1 = PN(64)
        self.PN2 = PN(64)
        self.PN3 = PN(64)

        # self.AFM1 = BasicConv2d(64, 64, 3, padding=1)
        self.AFM2 = BasicConv2d(64, 64, 3, padding=1)

        self.result1 = BasicConv2d(128, 64, 3, padding=1)
        self.result = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3):
        """
        :param x1: backbone的第三层的特征， resnet50 为512通道
        :param x2: backbone第四层的特征， resnet50 为1024通道
        :param x3: backbone第五层的特征，resnet50 为2048通道
        :return: 返回两个结果，分别是单通道未被激活的特征，以及上一步的结果. 简单点说就是返回64通道和返回1通道的结果
        """
        """主要思考思路：
        1、类似FPN简单获取多的信息但是有可能结果丢失比较严重，同时本身上采样的选择影响比较的大。
        2、多层级的操作直到最终只有一个输出。
        3、考虑其中加入空间注意力或者通道注意力来尝试是否会对结果提升较大。
        4、尽可能多的获取显著性目标的范围，为后续层次的特征多考虑。"""

        x3 = self.BM3(x3)
        x2 = self.BM2(x2)
        x1 = self.BM1(x1)
        # 使用膨胀卷积来扩大感受野
        x3 = self.PN3(x3)
        x2 = self.PN2(x2)
        x1 = self.PN1(x1)

        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x3 = self.FM3(x3)
        x2_new = x3 + x2

        x2_new = F.interpolate(x2_new, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x2_new = self.FM2(x2_new)
        x1_new = x2_new + x1

        x1_new = self.upsample(x1_new)
        x1_new = self.FM1(x1_new)  # 将尺寸还原到了H/2 W/2

        x2_n2 = self.upsample(x2_new)
        x2_n2 = self.AFM2(x2_n2)
        result = torch.cat((x2_n2, x1_new), 1)
        result = self.result1(result)  # 得到的是融合后的特征，同时没有进行最后显著性图的卷积操作
        resultmap = self.result(result)  # 得到的是没有激活的显著性图，单个通道
        # 现在x2_new已经是x1的尺寸，x1_new的尺寸已经是2x(x1)的尺寸，
        return result, resultmap


class DecoderRESF(nn.Module):
    """
    日期：2023-11-1
    作用：DecoderRES的更改版本，让其实现粗分割的目的，也是只包含后三层backbone输出的特征。
    作者：Amos
    """

    def __init__(self, channel1, channel2, channel3, channel4):
        super(DecoderRESF, self).__init__()
        # 将其降到64通道是为了减少计算量
        self.BM4 = BasicConv2d(channel4, channel3, 1, padding=0)
        self.BM3 = BasicConv2d(channel3, channel2, 1, padding=0)
        self.BM2 = BasicConv2d(channel2, channel1, 1, padding=0)
        # self.FM1 = BasicConv2d(channel1, channel1, 1, padding=0)
        self.FM3 = BasicConv2d(channel3, channel3, 1, padding=0)
        self.FM2 = BasicConv2d(channel2, channel2, 1, padding=0)
        self.FM1 = BasicConv2d(channel1, channel1, 1, padding=0)
        self.result1 = BasicConv2d(channel1, 64, 3, padding=1)
        self.result = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):
        """
        :param x1: backbone的第三层的特征， resnet50 为512通道
        :param x2: backbone第四层的特征， resnet50 为1024通道
        :param x3: backbone第五层的特征，resnet50 为2048通道
        :return: 返回两个结果，分别是单通道未被激活的特征，以及上一步的结果. 简单点说就是返回64通道和返回1通道的结果
        """
        """主要思考思路：
        1、类似FPN简单获取多的信息但是有可能结果丢失比较严重，同时本身上采样的选择影响比较的大。
        2、多层级的操作直到最终只有一个输出。
        3、考虑其中加入空间注意力或者通道注意力来尝试是否会对结果提升较大。
        4、尽可能多的获取显著性目标的范围，为后续层次的特征多考虑。"""
        x4 = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x4 = self.BM4(x4)
        x3 = self.FM3(x3)
        x3_new = x4 + x3

        x3_new = F.interpolate(x3_new, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x3 = self.BM3(x3_new)

        x2 = self.FM2(x2)
        x2_new = x3 + x2

        x2_new = F.interpolate(x2_new, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x2_new = self.BM2(x2_new)

        x1 = self.FM1(x1)
        x1_new = x2_new + x1

        x1_new = self.upsample(x1_new)
        result = self.result1(x1_new)  # 将尺寸还原到了H/2 W/2
        resultmap = self.result(result)  # 得到的是没有激活的显著性图，单个通道
        # 现在x2_new已经是x1的尺寸，x1_new的尺寸已经是2x(x1)的尺寸，
        return result, resultmap


class MapEnhance(nn.Module):
    """
    日期：2023-10-30
    作用：将上一层级联块的输出进行增强，生成相应的权重矩阵，然后点乘一个64通道的特征，生成一个显著性目标结果增强的特征用于下一阶段的输入
    作者：Amos
    """
    def __init__(self):
        super(MapEnhance, self).__init__()
        self.conv = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, oute, outmap):
        # 这里简单的得出高低频然后利用高低频信息来得到权重矩阵乘上上一层的特征
        outmap = torch.sigmoid(outmap)   # 先将显著性图激活得到真正的显著性图结果
        map_l, map_h = change(outmap)    # 将高低频，分离。注意，目的并不是为了得到高低频，是想将高低频做为权重来相乘
        outel = oute * map_l
        outeh = oute * map_h
        out = torch.cat((outel, outeh), 1)  # 对权重做卷积
        result = self.conv(out)  # 依托卷积的方式来得到结果，目的就是想学习一定的参数
        last = result + oute
        return last


def change(x):
    # x为swin的结果，y为resnet的结果
    # 第一步进行傅里叶变换，得到频谱图
    fx = torch.fft.fftn(x, dim=(-4, -3, -2, -1))

    # 第二步，将频谱进行中心化
    fx_shift = torch.fft.fftshift(fx, dim=(-4, -3, -2, -1))
    # print(fx_shift.shape)
    # 第三步，将中心化后的结果得到对应的高低频部分，取最中间的部分的为低频
    width1 = fx_shift.shape[2]
    height1 = fx_shift.shape[3]
    # print(width2, height2)
    mask = torch.zeros_like(fx_shift)
    radius = min(width1, height1) // 4
    center_x, center_y = width1 // 2, height1 // 2
    x, y = torch.meshgrid(torch.arange(0, width1), torch.arange(0, height1))
    circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    mask[:, :, circle] = 1.0
    fx_low = fx_shift * mask    # 中间是低频
    fx_high = fx_shift - fx_low  # 四周是高频
    fx_lnew = torch.fft.ifftn(torch.fft.ifftshift(fx_low, dim=(-4, -3, -2, -1)), dim=(-4, -3, -2, -1)).real
    fx_hnew = torch.fft.ifftn(torch.fft.ifftshift(fx_high, dim=(-4, -3, -2, -1)), dim=(-4, -3, -2, -1)).real
    # 返回增加了傅里叶卷积的内容
    return fx_lnew, fx_hnew


class DecoderSegNext(nn.Module):
    """
        日期：2023-10-30
        作用：SegNext层的Decoder的设计，个人认为该Decoder的设计应该主要的思考是将结果生成好，而不再是第一层级联结构想要更广泛的得到信息和帮助。
        所以主要的思考应该是将得到的特征进行充分的利用，特别是多尺度的特征，以及不同语义的特征。
        作者：Amos
        """
    def __init__(self):
        super(DecoderSegNext, self).__init__()
        self.conv4 = BasicConv2d(512, 320, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(320, 128, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(128, 64, kernel_size=3, padding=1)
        self.conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.cscale3 = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.cscale2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.cscale1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convlast = nn.Sequential(BasicConv2d(128+64+64, 64, kernel_size=3, padding=1),
                                      nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
    def forward(self, x1, x2, x3, x4):
        """
        :param x1: 第一层Encoder的结果， 64通道
        :param x2: 第二层Encoder的结果， 128通道
        :param x3: 第三层Encoder的结果， 320通道
        :param x4: 第四层Encoder的结果， 520通道
        :return: 返回最终的结果显著性图。
        """
        x4 = self.conv4(x4)
        x4 = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x4 = self.cscale3(x4)

        x3_new = x4 + x3
        x3_new = self.conv3(x3_new)
        x3_new = F.interpolate(x3_new, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x3_new = self.cscale2(x3_new)

        x2_new = x3_new + x2
        x2_new = self.conv2(x2_new)
        x2_new = F.interpolate(x2_new, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x2_new = self.cscale1(x2_new)

        x1_new = x2_new + x1
        x1_new = self.conv1(x1_new)

        x3_new = F.interpolate(x3_new, size=x1.shape[2:], mode='bilinear', align_corners=True)

        result = torch.cat((x1_new, x2_new, x3_new), 1)
        result = self.convlast(result)
        return result
