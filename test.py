#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
#from BBRNet import BBBNet
#from BBDGE import BBBNet
#from BBDGE2 import BBBNet
#from BBDGE3 import BBBNet
from CasCadeUR import BBBNet
import logging as logger
TAG = "DSSv1"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%(TAG), filemode="w")
from mmseg.visualization import SegLocalVisualizer
import mmcv
def visualize(img, feature,out_file,imgname):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend'),
                      dict(type='TensorboardVisBackend'),
                      dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    image = mmcv.imread(img, 'color')
    drawn_img = seg_visualizer.draw_featmap(feature, image, channel_reduction='squeeze_mean', )  # 'select_max''squeeze_mean' topk=6, arrangement=(2, 3)
    mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file + '/' + f'{imgname}.png')
def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn

def hsigmoid(x):
    x = 0.2*x+0.5
    x[x>1]=1
    x[x<0]=0
    return x

def clabel(x):
    bx = bbox(x)
    res = torch.rand(bx.shape[0],1).float().cuda()
    for i in range(bx.shape[0]):
        res[i] = ((bx[i][2]-bx[i][0])*(bx[i][3]-bx[i][1]))/(x.shape[2]*x.shape[3])
    res[res<0]=-res[res<0]
    return res

def select(p1,p2,p3,c):
   # if c[0][0]>=0.8:
   #     return p1
    return p1
class Tolast(nn.Module):
    def __init__(self):
        super(Tolast, self).__init__()
        self.linearx_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linearx_2 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.linearx_3 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        self.linearx_4 = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1)
        self.linears_1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linears_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linears_3 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.linears_4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
    def forward(self,xx1,xx2,xx3,xx4,ss1,ss2,ss3,ss4,shape):
        #print((self.linearx_1(xx1)).shape)
        x1 = F.interpolate(self.linearx_1(xx1), size=shape, mode='bilinear')
        x2 = F.interpolate(self.linearx_2(xx2), size=shape, mode='bilinear')
        x3 = F.interpolate(self.linearx_3(xx3), size=shape, mode='bilinear')
        x4 = F.interpolate(self.linearx_4(xx4), size=shape, mode='bilinear')
        s1 = F.interpolate(self.linears_1(ss1), size=shape, mode='bilinear')
        s2 = F.interpolate(self.linears_2(ss2), size=shape, mode='bilinear')
        s3 = F.interpolate(self.linears_3(ss3), size=shape, mode='bilinear')
        s4 = F.interpolate(self.linears_4(ss4), size=shape, mode='bilinear')
        return x1,x2,x3,x4,s1,s2,s3,s4
def toCHW(s1,s2,s3,s4):
    #x1,x2,x3,x4,
    # x1 = torch.squeeze(x1, 0)
    # x2 = torch.squeeze(x2, 0)
    # x3 = torch.squeeze(x3, 0)
    # x4 = torch.squeeze(x4, 0)
    s1 = torch.squeeze(s1, 0)
    s2 = torch.squeeze(s2, 0)
    s3 = torch.squeeze(s3, 0)
    s4 = torch.squeeze(s4, 0)
    # s5 = torch.squeeze(s5, 0)
    return s1,s2,s3,s4
def toOnemap(x1,x2,x3,x4,s1,s2,s3,s4,index):
    if index==1:
        sumx1 = torch.sum(x1, dim=(1, 2))
        _, indices = torch.topk(sumx1, 1)
        x1 = x1[indices]
        sumx2 = torch.sum(x2, dim=(1, 2))
        _, indices2 = torch.topk(sumx2, 1)
        x2 = x2[indices2]
        sumx3 = torch.sum(x3, dim=(1, 2))
        _, indices3 = torch.topk(sumx3, 1)
        x3 = x3[indices3]
        sumx4 = torch.sum(x4, dim=(1, 2))
        _, indices4 = torch.topk(sumx4, 1)
        x4 = x4[indices4]
        sums1 = torch.sum(s1, dim=(1, 2))
        _, indices5 = torch.topk(sums1, 1)
        s1 = s1[indices5]
        sums2 = torch.sum(s2, dim=(1, 2))
        _, indices6 = torch.topk(sums2, 1)
        s2 = s2[indices6]
        sums3 = torch.sum(s3, dim=(1, 2))
        _, indices7 = torch.topk(sums3, 1)
        s3 = s3[indices7]
        sums4 = torch.sum(s4, dim=(1, 2))
        _, indices8 = torch.topk(sums4, 1)
        s4 = s4[indices8]
    else:
        x1 = torch.mean(x1, dim=0)
        x2 = torch.mean(x2, dim=0)
        x3 = torch.mean(x3, dim=0)
        x4 = torch.mean(x4, dim=0)
        s1 = torch.mean(s1, dim=0)
        s2 = torch.mean(s2, dim=0)
        s3 = torch.mean(s3, dim=0)
        s4 = torch.mean(s4, dim=0)
    return x1,x2,x3,x4,s1,s2,s3,s4
class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)

       # print(self.net)
        self.net.train(False)
        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
        # device = "cpu"
        # self.net = self.net.to(device)

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, mask, (H, W), maskpath in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()

                start_time = time.perf_counter()
                p= self.net(image)
                pred = torch.sigmoid(p[0,0])
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                cost_time += end_time - start_time
                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.cfg.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)
    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.float().cuda()
                p0, p1, p2,p3= self.net(image, shape=shape)
		# 将四个级联的输出都进行保存
                out0 = torch.sigmoid(p0[0, 0])
                pred0 = (out0 * 255).cpu().numpy()
                out1 = torch.sigmoid(p1[0, 0])
                pred1 = (out1 * 255).cpu().numpy()
                out2 = torch.sigmoid(p2[0, 0])
                pred2 = (out2 * 255).cpu().numpy()
                out3 = torch.sigmoid(p3[0, 0])
                pred3 = (out3 * 255).cpu().numpy()
                headp1 = '/data/Hms/resultmap/out10-23-2/' + self.cfg.datapath.split('/')[-1] + 'p1'
                headp2 = '/data/Hms/resultmap/out10-23-2/' + self.cfg.datapath.split('/')[-1] + 'p2'
                headp3 = '/data/Hms/resultmap/out10-23-2/' + self.cfg.datapath.split('/')[-1] + 'p3'
                headp4 = '/data/Hms/resultmap/out10-23-2/' + self.cfg.datapath.split('/')[-1] + 'p4'
                if not os.path.exists(headp1):
                        os.makedirs(headp1)
                if not os.path.exists(headp2):
                    os.makedirs(headp2)
                if not os.path.exists(headp3):
                    os.makedirs(headp3)
                if not os.path.exists(headp4):
                    os.makedirs(headp4)

                cv2.imwrite(headp1 + '/' + name[0] + '.png', np.round(pred0))
                cv2.imwrite(headp2 + '/' + name[0] + '.png', np.round(pred1))
                cv2.imwrite(headp3 + '/' + name[0] + '.png', np.round(pred2))
                cv2.imwrite(headp4 + '/' + name[0] + '.png', np.round(pred3))
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    path = '/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test'
    model = '//data/Hms/model_Cas/CasCade-10-23-2/model-73'
    t = Test(dataset, BBBNet, path, model)
    t.save()
