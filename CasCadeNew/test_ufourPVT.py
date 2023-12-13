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
from CasCadeUFOUR.stackNetB import StackedHGNetV1
import logging as logger
import imageio
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

    # seg_visualizer.add_datasample(
    #     name='predict',
    #     image=image,
    #     data_sample=result,
    #     draw_gt=False,
    #     draw_pred=True,
    #     wait_time=0,
    #     out_file=None,
    #     show=False)

    # add feature map to wandb visualizer
    drawn_img = seg_visualizer.draw_featmap(feature, image, channel_reduction='squeeze_mean', )  # 'select_max''squeeze_mean' topk=6, arrangement=(2, 3)
        # seg_visualizer.show(drawn_img)
    mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file + '/' + f'{imgname}.png')
class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test', width=256, height=256)
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg, 1)

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
                #image = image.cuda().float()
                image = image.float().cuda()
                (p1, p2, p3, p4) = self.net(image)  #, p2, p3, p4, edge, body
                p1 = F.interpolate(p1, size=shape, mode='bilinear', align_corners=True)
                p2 = F.interpolate(p2, size=shape, mode='bilinear', align_corners=True)
                p3 = F.interpolate(p3, size=shape, mode='bilinear', align_corners=True)
                p4 = F.interpolate(p4, size=shape, mode='bilinear', align_corners=True)
                out1 = torch.sigmoid(p1[0, 0])
                pred1 = (out1 * 255).cpu().numpy()

                out2 = torch.sigmoid(p2[0, 0])
                pred2 = (out2 * 255).cpu().numpy()

                out3 = torch.sigmoid(p3[0, 0])
                pred3 = (out3 * 255).cpu().numpy()

                out4 = torch.sigmoid(p4[0, 0])
                pred4 = (out4 * 255).cpu().numpy()
                # out0 = torch.sigmoid(pred2[0, 0])
                # pred2 = (out0 * 255).cpu().numpy()
                #
                # out2 = torch.sigmoid(pred3[0, 0])
                # pred3 = (out2 * 255).cpu().numpy()

                headp1 = '/data/Hms/resultmapNew/CasUFour12-8-1/' + self.cfg.datapath.split('/')[-1] + 'p1'
                headp2 = '/data/Hms/resultmapNew/CasUFour12-8-1/' + self.cfg.datapath.split('/')[-1] + 'p2'
                headp3 = '/data/Hms/resultmapNew/CasUFour12-8-1/' + self.cfg.datapath.split('/')[-1] + 'p3'
                headp4 = '/data/Hms/resultmapNew/CasUFour12-8-1/' + self.cfg.datapath.split('/')[-1] + 'p4'
                if not os.path.exists(headp1):
                        os.makedirs(headp1)
                if not os.path.exists(headp2):
                    os.makedirs(headp2)
                if not os.path.exists(headp3):
                    os.makedirs(headp3)
                if not os.path.exists(headp4):
                    os.makedirs(headp4)
                # imageio.imsave(headp1 + '/' + name[0] + '.png', res)
                cv2.imwrite(headp1 + '/' + name[0] + '.png', np.round(pred1))
                cv2.imwrite(headp2 + '/' + name[0] + '.png', np.round(pred2))
                cv2.imwrite(headp3 + '/' + name[0] + '.png', np.round(pred3))
                cv2.imwrite(headp4 + '/' + name[0] + '.png', np.round(pred4))
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    path = '/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test'
    model = '/data/Hms/model_newcas/CasCadeUFour-12-8-1/model-222'
    t = Test(dataset, StackedHGNetV1, path, model)
    t.save()
