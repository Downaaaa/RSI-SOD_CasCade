import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from BBRF2 import BBBNet2
from GeleNet.model.GeleNet_models import GeleNet
from Attention import HA
class CasCadeGR(nn.Module):
    def __init__(self, cfg=None):
        super(CasCadeGR, self).__init__()
        self.cfg = cfg
        self.cas1 = GeleNet()
        self.cas1.load_state_dict(torch.load('/data/Hms/pth/GeleNet_EORSSD_PVT.pth'))
        self.cas2 = BBBNet2()
        self.HA = HA()
    def forward(self, x,shape=None,mask=None):
        shape = x.size()[2:] if shape is None else shape
        outns, outs = self.cas1(x)
        input2 = self.HA(outs, outns)
        input21 = torch.cat((input2, input2), 1)
        input21 = torch.cat((input21, input2), 1)
        # input2 = F.interpolate(input2, size=[256, 256], mode='bilinear', align_corners=True)
        p2, p11, p12, p13, p14 = self.cas2(input21, shape=shape)
        p1 = F.interpolate(outns, size=shape, mode='bilinear', align_corners=True)
        return p1, p2, p11, p12, p13, p14

