import torch
import torch.nn as nn
import torch.nn.functional as F
from .SwinV2 import SwinTransformerV2
from .coord_conv import CoordConvTh
from .ACmixBlock import ACmix

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
            self.relu = nn.ReLU(inplace=False)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvBlock2(nn.Module):
    """卷积块，批量归一和激活层可以灵活的去除，初始设置的是批量归一去除，激活保留，只需控制两个布尔值就可以观察结果"""

    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, ln=False, gelu=True, groups=1, width=128):
        super(ConvBlock2, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size - 1) // 2, groups=groups, bias=True)
        self.gelu = None
        self.ln = None
        if gelu:
            self.gelu = nn.GELU()
        if ln:
            self.ln = nn.LayerNorm([width, width], elementwise_affine=True)

    def forward(self, x):
        x = self.conv(x)
        if self.ln is not None:
            x = self.ln(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    """残差连接块，该残差连接块其实就是resnet的残差块，不过该残差块是设计的3个卷积、归一、激活的块.是Bottleblock的数量,而BasicBlock是2个卷积、归一、激活的块。这一点需要注意
    初始只需要两个信息，输入通道维度以及输出维度。中间维度可以不用考虑，默认是输出维度的一半。"""

    def __init__(self, inp_dim, out_dim, mid_dim=None, stride=1, down=False):
        super(ResBlock, self).__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.conv1 = conv1x1(inp_dim, mid_dim)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv2 = conv3x3(mid_dim, mid_dim, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.conv3 = conv1x1(mid_dim, out_dim)
        self.skip_layer = conv1x1(inp_dim, out_dim, stride=stride)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        if down:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 为什么是先批量归一？
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = out + residual
        return out


"""ResBlock是传统的残差块的BasicLayer，这里尝试新的块"""


class ACmixBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, k_att=7, head=4, k_conv=3, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ACmixBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = ACmix(width, width, k_att, head, k_conv, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes),
            norm_layer(planes), )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Hourglass(nn.Module):
    """沙漏网络的设计，注意一点这里的沙漏网络和传统的沙漏网络的区别在何处？上采样方式采用最近邻"""

    def __init__(self, n, f, increase=0, up_mode='bilinear',
                 add_coord=False, first_one=False, x_dim=64, y_dim=64):  # nearest'
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
        self.pool1 = Block(f, f, stride=2, down=True)

        self.low1 = Block(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            increase = nf
            self.low2 = Hourglass(n=n - 1, f=nf, increase=increase, up_mode=up_mode,
                                  add_coord=False)  # 此时不需要增加coord卷积是只包含
            # 采用递归的写法，直到变为1就采用残差块。也就是说其中会嵌套3次up1+pool1 + low1 + net + low3,其中net都是一次沙漏块的循环，也就是说其实是逐步的维度变小，完全可以UNet的写法。
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)  # 将heatmap 和原始输入通过coordconv合在一起，不过这里的heatmap没说维度？在最开始进行
        up1 = self.up1(x)
        # print(up1.shape)
        pool1 = self.pool1(up1)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)  # 输出的维度和输入的维度明显是一样的
        return up1 + up2


"""也就是说此时只有上述的模块是会用到的，后面的很多东西需要去除人脸关键点检测的特有的数据"""


class ACmixHoutglass(nn.Module):
    def __init__(self, n, f, increase=0, up_mode='bilinear',
                 add_coord=False, first_one=False, x_dim=64, y_dim=64):  # nearest'
        super(ACmixHoutglass, self).__init__()
        # 其中f表示输入维度， N表示定义的层次数量， increse就是其中隐藏维度增加的数量
        nf = f + increase

        Block = ACmixBlock
        Resblock = ResBlock
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
            self.low2 = ACmixHoutglass(n=n - 1, f=nf, increase=increase, up_mode=up_mode,
                                       add_coord=False)  # 此时不需要增加coord卷积是只包含
            # 采用递归的写法，直到变为1就采用残差块。也就是说其中会嵌套3次up1+pool1 + low1 + net + low3,其中net都是一次沙漏块的循环，也就是说其实是逐步的维度变小，完全可以UNet的写法。
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)  # 将heatmap 和原始输入通过coordconv合在一起，不过这里的heatmap没说维度？在最开始进行
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)  # 输出的维度和输入的维度明显是一样的
        return up1 + up2


class StackedHGNetV1(nn.Module):
    def __init__(self, config, classes_num, nstack=4, nlevels=4, in_channel=256, increase=256,
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
                                    stride=2, padding=3)  # 初始加入coord卷积
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
        )  # 数据的预先卷积的过程，目的就是直接的进行后续的沙漏网络的输入和上一层的输出更好的融合
        """数据的预处理阶段的尺寸应该是(B,256,64,64) 如果将预处理阶段的代码换成Swin Transformer的块那么明显就可以快速的获取底层信息
"""
        # self.pre2 = SwinTransformerV2(img_size=256, window_size=8,  depths=[2], num_heads=[4], embed_dim=128)
        # self.pre3 = Block(128, in_channel)
        self.hgs = nn.ModuleList(
            [Hourglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, first_one=(_ == 0),
                       x_dim=int(self.cfg.width / (self.nstack)), y_dim=int(self.cfg.height / (self.nstack)))
             for _ in range(nstack)])  # 每个沙漏网络的设计，不过特征的宽高的设计是否需要特殊的处理？

        self.features = nn.ModuleList([
            nn.Sequential(
                Block(in_channel, in_channel),
                ConvBlock(in_channel, in_channel, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.out_heatmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_heats, 1, relu=True, bn=True)
             for _ in range(nstack)])

        self.merge_features = nn.ModuleList(
            [ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack - 1)])
        self.merge_heatmaps = nn.ModuleList(
            [ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack - 1)])

        self.nstack = nstack

        self.heatmap_act = Activation("bn+relu", self.num_heats)

        self.inference = False

        self.decoder = NetDecoder(in_channel)
        self.last1 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last2 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last3 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last4 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)

    def set_inference(self, inference):
        self.inference = inference

    def forward(self, x1):
        x = self.pre(x1)  # 先预先获得预处理阶段的输出
        features = []
        heatmaps = None
        for i in range(self.nstack):
            # nstack是指堆叠的个数，根据堆叠的个数来
            hg = self.hgs[i](x, heatmap=heatmaps)
            feature = self.features[i](hg)
            features.append(feature)
            # print(feature.shape)
            heatmaps = self.out_heatmaps[i](feature)  # 将得到的特征生成类别的热图，也就是语义分割最后一层的前面一层
            # heatmaps = self.heatmap_act(heatmaps0)  # 每个阶段都得到相应的激活的热图
            # fusion_heatmaps = F.interpolate(heatmaps0, size=x1.shape[2:], mode='bilinear', align_corners=True)

            if i < self.nstack - 1:
                x = x + self.merge_features[i](feature) + \
                    self.merge_heatmaps[i](heatmaps)  # 将三者的维度换到同一个维度，也就是之前的pre的结果
            # fusionmaps.append(fusion_heatmaps)
        out1, out2, out3, out4 = self.decoder(features[0], features[1], features[2], features[3])
        out1 = F.interpolate(out1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        out2 = F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        out3 = F.interpolate(out3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=x1.shape[2:], mode='bilinear', align_corners=True)
        return out1, out2, out3, out4  # 此处的输出是直接output、热图、关键点图。
        # return fusionmaps  # 此处的输出是直接output、热图、关键点图。


class StackedHGNetV2(nn.Module):
    def __init__(self, config, classes_num, nstack=4, nlevels=4, in_channel=256, increase=0,
                 add_coord=True):
        super(StackedHGNetV2, self).__init__()
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
                                    stride=2, padding=3)  # 初始加入coord卷积
        else:
            convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        Block = ACmixBlock

        self.pre = nn.Sequential(
            convBlock,
            self.maxpool,
            Block(64, 128),
            Block(128, 128),
            Block(128, in_channel)
        )  # 数据的预先卷积的过程，目的就是直接的进行后续的沙漏网络的输入和上一层的输出更好的融合
        """数据的预处理阶段的尺寸应该是(B,256,64,64) 如果将预处理阶段的代码换成Swin Transformer的块那么明显就可以快速的获取底层信息
"""

        self.hgs = nn.ModuleList(
            [ACmixHoutglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, first_one=(_ == 0),
                            x_dim=int(self.cfg.width / self.nstack), y_dim=int(self.cfg.height / self.nstack))
             for _ in range(nstack)])  # 每个沙漏网络的设计，不过特征的宽高的设计是否需要特殊的处理？

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
        self.decoder = NetDecoder(in_channel)
        self.last1 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last2 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last3 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last4 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
    def set_inference(self, inference):
        self.inference = inference

    def forward(self, x1):
        x = self.pre(x1)  # 先预先获得预处理阶段的输出
        # print(x.shape)
        # fusionmaps = []
        features = []
        heatmaps = None
        for i in range(self.nstack):
            # nstack是指堆叠的个数，根据堆叠的个数来
            hg = self.hgs[i](x, heatmap=heatmaps)
            feature = self.features[i](hg)     #  将该输出作为每个阶段的特征输出即(B, 256, H/2, W/2)的同一纬度，
            features.append(feature)
            # print(feature.shape)
            heatmaps0 = self.out_heatmaps[i](feature)  # 将得到的特征生成类别的热图，也就是语义分割最后一层的前面一层
            heatmaps = self.heatmap_act(heatmaps0)  # 每个阶段都得到相应的激活的热图
            # fusion_heatmaps = F.interpolate(heatmaps0, size=x1.shape[2:], mode='bilinear', align_corners=True)

            if i < self.nstack - 1:
                x = x + self.merge_features[i](feature) + \
                    self.merge_heatmaps[i](heatmaps)  # 将三者的维度换到同一个维度，也就是之前的pre的结果
            # fusionmaps.append(fusion_heatmaps)

        out1, out2, out3, out4 = self.decoder(features[0],features[1],features[2],features[3])
        out1 = F.interpolate(self.last1(out1), size=x1.shape[2:], mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.last1(out2), size=x1.shape[2:], mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.last1(out3), size=x1.shape[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.last1(out4), size=x1.shape[2:], mode='bilinear', align_corners=True)
        return out1, out2, out3, out4# 此处的输出是直接output、热图、关键点图。


class NetDecoder(nn.Module):
    """考虑StackedHGNetV1作为Encoder后续加入一个小的Decoder。但是问题如下：
    1、是否像传统的Decoder那样做语义引导？----个人认为不能像传统的Decoder，因为每个阶段的输出都是有一定的优点也有一点缺失，所以不能像传统的手段。
    2、针对每个stage的输出都是一样的维度重点是什么？----个人认为重点是进行信息的互补，因为每个阶段信息都是有缺失的。
    3、作为Decoder的重点就是恢复维度，同时信息补充，那么将所有的输出直接一起作用于一个模块是不是一定有效呢？"""
    def __init__(self, in_channel):
        super(NetDecoder, self).__init__()
        self.decoder1 = ConvBlock2(in_channel, in_channel, 3, ln=True, gelu=True, width=64)
        self.decoder2 = ConvBlock2(in_channel, in_channel, 3, ln=True, gelu=True, width=64)
        self.decoder3 = ConvBlock2(in_channel, in_channel, 3, ln=True, gelu=True, width=64)
        self.decoder4 = ConvBlock2(in_channel, in_channel, 3, ln=True, gelu=True, width=64)
        self.last1 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last2 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last3 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.last4 = ConvBlock(in_channel, 1, 3, bn=False, relu=False)
        self.tochannel1 = nn.Sequential(
            ConvBlock(in_channel*3, in_channel, 3, bn=True, relu=True),
            ConvBlock(in_channel, in_channel, 3, bn=True, relu=True),
            ConvBlock(in_channel, in_channel, 3, bn=True, relu=True),
        )
        self.tochannel2 = nn.Sequential(
            ConvBlock(in_channel * 3, in_channel, 3, bn=True, relu=True),
            ConvBlock(in_channel, in_channel, 3, bn=True, relu=True),
            ConvBlock(in_channel, in_channel, 3, bn=True, relu=True),
        )
        self.tochannel3 = nn.Sequential(
            ConvBlock(in_channel * 3, in_channel, 3, bn=True, relu=True),
            ConvBlock(in_channel, in_channel, 3, bn=True, relu=True),
            ConvBlock(in_channel, in_channel, 3, bn=True, relu=True),
        )
    def forward(self, x1, x2, x3, x4):
        dout1 = self.decoder1(x1)
        out1 = self.last1(dout1)

        inp2 = torch.cat((dout1, x1, x2), 1)
        inp2 = self.tochannel1(inp2)
        dout2 = self.decoder2(inp2)
        out2 = self.last2(dout2)

        inp3 = torch.cat((dout2, inp2, x3), 1)
        inp3 = self.tochannel2(inp3)
        dout3 = self.decoder3(inp3)
        out3 = self.last3(dout3)

        inp4 = torch.cat((dout3, inp3, x4), 1)
        inp4 = self.tochannel3(inp4)
        dout4 = self.decoder4(inp4)
        out4 = self.last4(dout4)

        return out1, out2, out3, out4