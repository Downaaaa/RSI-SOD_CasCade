import torch
import torch.nn as nn
import torch.nn.functional as F
from .SwinV2 import SwinTransformerV2
from .coord_conv import CoordConvTh
from .pvtv2 import Block, OverlapPatchEmbed
import math
from timm.models.layers import  trunc_normal_
from functools import partial
"""
文件名：
主要改动：
1、将预训练阶段即pre阶段换成了PVT的块
2、将沙漏网络的部分块换成了PVT
"""


class PVTBlock(nn.Module):
    """
    该类主要用于pre阶段的代码
    """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 256, 256, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[1, 1, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=1, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        # cur += self.depths[1]
        # for i in range(self.depths[2]):
        #     self.block3[i].drop_path.drop_prob = dpr[cur + i]
        #
        # cur += self.depths[2]
        # for i in range(self.depths[3]):
        #     self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        # x, H, W = self.patch_embed3(x)
        # for i, blk in enumerate(self.block3):
        #     x = blk(x, H, W)
        # x = self.norm3(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)
        #
        # # stage 4
        # x, H, W = self.patch_embed4(x)
        # for i, blk in enumerate(self.block4):
        #     x = blk(x, H, W)
        # x = self.norm4(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        return outs

        # return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class OnePVTBlock(nn.Module):
    """该类作为一个独立使用的PVTBlock，不过其设置是要进行更改才能进行使用的"""
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=1000, embed_dims=64,
                 num_heads=None, mlp_ratios=None, qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=1, sr_ratios=None, stride=1):
        super().__init__()
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if mlp_ratios is None:
            mlp_ratios = [8, 8, 4, 4]
        if num_heads is None:
            num_heads = [5, 8]
        """作为单个的PVTBlock需要参数更改的地方有：img_size、in_chans、embed——dims、depths, 此处的patch_size可以具象化为卷积核的大小， stride需要重新的定义"""
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans,
                                              embed_dim=embed_dims)
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths)])
        self.norm1 = norm_layer(embed_dims)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depths)]
        cur = 0
        for i in range(self.depths):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x


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
        out = self.conv1(out)  
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


class PVTHourglass(nn.Module):
    """
    主要改动：
    1、利用PVTBlock完成了PVT的沙漏网络的设计，不过只是将Encoder的第一层换成了PVTBlock，不过这里也是改变了沙漏网络的结构
    2、和上面的沙漏网络有不一样的地方，up1之前作为Encoder和Decoder之间连接的块，现在变成了Encoder的第一个块。沙漏网络的结构有所变化
    """

    def __init__(self, n, f, increase=0, up_mode='bilinear',
                 add_coord=False, first_one=False, x_dim=64, y_dim=64):  # nearest'
        super(PVTHourglass, self).__init__()
        # 其中f表示输入维度， N表示定义的层次数量， increse就是其中隐藏维度增加的数量
        nf = f + increase

        Block = ResBlock
        PVTBlock = OnePVTBlock
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
        self.up1 = PVTBlock(in_chans=f, embed_dims=f, img_size=x_dim, depths=1, patch_size=3, stride=2)

        # Lower branch
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.low1 = Block(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            increase = f
            self.low2 = PVTHourglass(n=n - 1, f=nf, increase=increase, up_mode=up_mode, x_dim=(x_dim//2), y_dim=(x_dim//2), add_coord=False)  
            # 采用递归的写法，直到变为1就采用残差块。也就是说其中会嵌套3次up1+pool1 + low1 + net + low3,其中net都是一次沙漏块的循环，也就是说其实是逐步的维度变小，完全可以UNet的写法。
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)  # 将heatmap 和原始输入通过coordconv合在一起，不过这里的heatmap没说维度？在最开始进行
        up1 = self.up1(x)
        # pool1 = self.pool1(x)
        low1 = self.low1(up1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        result = up1+low3
        result = self.up2(result)
        return result


"""也就是说此时只有上述的模块是会用到的，后面的很多东西需要去除人脸关键点检测的特有的数据"""


class StackedHGNetV1(nn.Module):
    """
    主要改动：
    1、将pre阶段换成了PVTBlock
    2、用PVTHourglass
    """
    def __init__(self, config, classes_num,nstack=4, nlevels=4, in_channel=256, increase=0,
                 add_coord=True):
        super(StackedHGNetV1, self).__init__()
        # increase应该如何设置？不可能一直采用256维度进行训练
        self.cfg = config
        self.nstack = nstack
        self.add_coord = add_coord

        self.num_heats = classes_num

        # if self.add_coord:
        #     convBlock = CoordConvTh(x_dim=self.cfg.width, y_dim=self.cfg.height,
        #                             with_r=True, with_boundary=False,
        #                             relu=True, bn=True,
        #                             in_channels=3, out_channels=64,
        #                             kernel_size=7,
        #                             stride=2, padding=3)     # 初始加入coord卷积
        # else:
        # convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)
        #
        # pool = nn.MaxPool2d(kernel_size=2, stride=2)

        Block = ResBlock

        # self.pre = nn.Sequential(
        #     convBlock,
        #     Block(64, 128),
        #     pool,
        #     Block(128, 128),
        #     Block(128, in_channel)
        # )    # 数据的预先卷积的过程，目的就是直接的进行后续的沙漏网络的输入和上一层的输出更好的融合
        """数据的预处理阶段的尺寸应该是(B,256,64,64) 如果将预处理阶段的代码换成Swin Transformer的块那么明显就可以快速的获取底层信息
"""
        self.pre2 = PVTBlock()
        self.hgs = nn.ModuleList(
            [PVTHourglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, first_one=(_ == 0),
                       x_dim=int(self.cfg.width / (self.nstack-2)), y_dim=int(self.cfg.height / (self.nstack-2)))
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
        # x = self.pre(x1)   # 先预先获得预处理阶段的输出
        (x2,x) = self.pre2(x1)
        # x = self.pre3(x)
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