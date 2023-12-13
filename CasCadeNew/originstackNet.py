import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_conv import CoordConvTh


"""注意该文件是定义一个网络，一个四个层次的网络，该网络是有4个阶段，
需要注意的是该网络是用于人脸关键点检测的，如果要用于显著性目标检测还需要进行输入和输出的更改
同时只需要网络的设置就可以，并不需要其余的信息"""


class Activation(nn.Module):
    """将归一化层和激活层统一写到一个类中，输入关键字进行选择"""
    def __init__(self, kind: str = 'relu', channel=None):
        super().__init__()
        self.kind = kind

        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'


class ConvBlock(nn.Module):
    """卷积块，批量归一和激活层可以灵活的去除，初始设置的是批量归一去除，激活保留，只需控制两个布尔值就可以观察结果"""
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size - 1) // 2, groups=groups, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """残差连接块，该残差连接块其实就是resnet的残差块，不过该残差块是设计的3个卷积、归一、激活的块.是Bottleblock的数量,而BasicBlock是2个卷积、归一、激活的块。这一点需要注意
    初始只需要两个信息，输入通道维度以及输出维度。中间维度可以不用考虑，默认是输出维度的一半。"""
    def __init__(self, inp_dim, out_dim, mid_dim=None):
        super(ResBlock, self).__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(mid_dim)
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)   # 为什么是先批量归一？
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    """沙漏网络的设计，注意一点这里的沙漏网络和传统的沙漏网络的区别在何处？上采样方式采用最近邻"""
    def __init__(self, n, f, increase=0, up_mode='bilinear',
                 add_coord=False, first_one=False, x_dim=64, y_dim=64):   # nearest'
        super(Hourglass, self).__init__()
        # 其中f表示输入维度， N表示定义的层次数量， increse就是其中隐藏维度增加的数量
        nf = f + increase

        Block = ResBlock

        if add_coord:
            self.coordconv = CoordConvTh(x_dim=x_dim, y_dim=y_dim,
                                         with_r=True, with_boundary=True,
                                         relu=False, bn=False,
                                         in_channels=f, out_channels=f,
                                         first_one=first_one,
                                         kernel_size=1,
                                         stride=1, padding=0)
            #  可以暂时将其看作是将上一个沙漏的输出如x进行结合的操作。
        else:
            self.coordconv = None
        self.up1 = Block(f, f)

        # Lower branch
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.low1 = Block(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            increase = f
            self.low2 = Hourglass(n=n - 1, f=nf, increase=increase, up_mode=up_mode, add_coord=False)   # 此时不需要增加coord卷积是只包含
            # 采用递归的写法，直到变为1就采用残差块。也就是说其中会嵌套3次up1+pool1 + low1 + net + low3,其中net都是一次沙漏块的循环，也就是说其实是逐步的维度变小，完全可以UNet的写法。
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)   # 将heatmap 和原始输入通过coordconv合在一起，不过这里的heatmap没说维度？在最开始进行
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)      #输出的维度和输入的维度明显是一样的
        return up1 + up2


"""也就是说此时只有上述的模块是会用到的，后面的很多东西需要去除人脸关键点检测的特有的数据"""


class StackedHGNetV1(nn.Module):
    def __init__(self, config, classes_num,nstack=4, nlevels=4, in_channel=256, increase=0,
                 add_coord=True):
        super(StackedHGNetV1, self).__init__()
        # increase应该如何设置？不可能一直采用256维度进行训练
        self.cfg = config
        self.nstack = nstack
        self.add_coord = add_coord

        self.num_heats = classes_num

        if self.add_coord:
            convBlock = CoordConvTh(x_dim=self.cfg.width, y_dim=self.cfg.height,
                                    with_r=True, with_boundary=False,
                                    relu=True, bn=True,
                                    in_channels=3, out_channels=64,
                                    kernel_size=7,
                                    stride=2, padding=3)     # 初始加入coord卷积
        else:
            convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)

        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        Block = ResBlock

        self.pre = nn.Sequential(
            convBlock,
            Block(64, 128),
            pool,
            Block(128, 128),
            Block(128, in_channel)
        )    # 数据的预先卷积的过程，目的就是直接的进行后续的沙漏网络的输入和上一层的输出更好的融合

        self.hgs = nn.ModuleList(
            [Hourglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, first_one=(_ == 0),
                       x_dim=int(self.cfg.width / self.nstack), y_dim=int(self.cfg.height / self.nstack))
             for _ in range(nstack)])    #每个沙漏网络的设计，不过特征的宽高的设计是否需要特殊的处理？

        self.features = nn.ModuleList([
            nn.Sequential(
                Block(in_channel, in_channel),
                ConvBlock(in_channel, in_channel, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.out_heatmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_heats, 1, relu=False, bn=False)
             for _ in range(nstack)])

        self.merge_features = nn.ModuleList(
            [ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack - 1)])
        self.merge_heatmaps = nn.ModuleList(
            [ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack - 1)])

        self.nstack = nstack

        self.heatmap_act = Activation("in+relu", self.num_heats)

        self.inference = False

    def set_inference(self, inference):
        self.inference = inference

    def forward(self, x1):
        x = self.pre(x1)   # 先预先获得预处理阶段的输出

        fusionmaps = []
        heatmaps = None
        for i in range(self.nstack):
            # nstack是指堆叠的个数，根据堆叠的个数来
            hg = self.hgs[i](x, heatmap=heatmaps)
            feature = self.features[i](hg)

            heatmaps0 = self.out_heatmaps[i](feature)   # 将得到的特征生成类别的热图，也就是语义分割最后一层的前面一层
            heatmaps = self.heatmap_act(heatmaps0)   # 每个阶段都得到相应的激活的热图
            # print(heatmaps.shape)
            fusion_heatmaps = F.interpolate(heatmaps0, size=x1.shape[2:], mode='bilinear', align_corners=True)
            # print(fusion_heatmaps.shape)

            if i < self.nstack - 1:
                x = x + self.merge_features[i](feature) + \
                    self.merge_heatmaps[i](heatmaps)     # 将三者的维度换到同一个维度，也就是之前的pre的结果
            fusionmaps.append(fusion_heatmaps)

        return fusionmaps    #此处的输出是直接output、热图、关键点图。