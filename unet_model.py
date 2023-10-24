""" Full assembly of the parts to form the complete network """

from unet_parts import *
from tools import ChannelAttention, SpatialAttention,ShuffleAttention

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64,shape=[128,128]))
        self.down1 = (Down(64, 128, [64, 64]))
        self.down2 = (Down(128, 256, [32, 32]))
        self.down3 = (Down(256, 512, [16,16]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, [8,8]))
        self.up1 = (Up(1024, 512 // factor, bilinear,[16,16]))
        self.up2 = (Up(512, 256 // factor, bilinear, [32,32]))
        self.up3 = (Up(256, 128 // factor, bilinear,[64,64]))
        self.up4 = (Up(128, 64, bilinear, [128,128]))
        self.outc = (OutConv(64, n_classes))
        self.attn0 = ChannelAttention(64)
        self.attn1 = ChannelAttention(128)
        self.attn2 = ChannelAttention(256)
        self.attn3 = ChannelAttention(512)
        self.attn4 = ChannelAttention(1024)
    def forward(self, x1):
        x1 = self.inc(x1)  #64
        x2 = self.down1(x1)  #128

        x3 = self.down2(x2)  #256

        x4 = self.down3(x3)  #512

        x5 = self.down4(x4)  #1024或512 中间的那一层
        x1 = x1*self.attn0(x1)
        x2 = x2*self.attn1(x2)
        x3 = x3*self.attn2(x3)
        x4 = x4*self.attn3(x4)
        x5 = x5*self.attn4(x5)
        x = self.up1(x5, x4)  #512或256 如果结果的通道数是大的话就是要有一个反卷积的过程。
        # x = self.up2(x4, x3)
        x = self.up2(x, x3)   # 256 或128
        x = self.up3(x, x2)   # 128 或 64
        x = self.up4(x, x1)    # 64  #此处的X1明显是没有处理过的，是上一个级联块的结果。上面的X2就是经过处理的，所以才会使得最终的结果几乎没有多大的变化
        logits = self.outc(x)
        return x,logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)