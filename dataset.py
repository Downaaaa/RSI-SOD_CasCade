#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask,edge,body):
        image = (image - self.mean)/self.std
        mask /= 255
        edge /= 255
        body /= 255
        return image, mask, edge, body


class NormalizeT(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask,edge,body):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], edge[p0:p1,p2:p3], body[p0:p1, p2:p3]
class RandomCropT(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3]
class RandomFlip(object):
    def __call__(self, image, mask,edge, body):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1],edge[:, ::-1], body[:, ::-1]
        else:
            return image, mask, edge,body
class RandomFlipT(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1]
        else:
            return image, mask
class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,edge,body):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask,edge,body
class ResizeT(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask, edge,body):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        edge = torch.from_numpy(edge)
        body = torch.from_numpy(body)
        return image, mask,edge,body
class ToTensorT(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(256, 256)
        self.totensor   = ToTensor()
        self.normalizeT = NormalizeT(mean=cfg.mean, std=cfg.std)
        self.randomcropT = RandomCropT()
        self.randomflipT = RandomFlipT()
        self.resizeT = ResizeT(256, 256)
        self.totensorT = ToTensorT()
        self.samples    = []
        lst = glob.glob(cfg.datapath +'/labels/'+'*.png')
        for each in lst:
            img_name = each.split("/")[-1]
            img_name = img_name.split(".")[0]
            self.samples.append(img_name)
    def __getitem__(self, idx):
        name  = self.samples[idx]
        tig='.jpg'
        if self.cfg.datapath=='../data/HKU-IS':
            tig='.png'
        image = cv2.imread(self.cfg.datapath+'/images/'+name+tig)[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.cfg.datapath+'/labels/' +name+'.png', 0).astype(np.float32)
        # edge = cv2.imread(self.cfg.datapath + '/edge2/' + name + '.png', 0).astype(np.float32)
        # body = cv2.imread(self.cfg.datapath + '/body/' + name + '.png', 0).astype(np.float32)
        """新加入先验知识：
        1、edge 2023年6月份就加入了
        2、body 2023年7月3日加入
        两个信息的加入都是进行相应的增强和背景加入"""
        shape = mask.shape

        if self.cfg.mode=='train':
            edge = cv2.imread(self.cfg.datapath + '/edge2/' + name + '.png', 0).astype(np.float32)
            body = cv2.imread(self.cfg.datapath + '/body/' + name + '.png', 0).astype(np.float32)
            image, mask,edge,body= self.normalize(image, mask,edge,body)
            image, mask,edge,body= self.randomcrop(image, mask,edge,body)
            image, mask ,edge,body= self.randomflip(image, mask,edge,body)
            return image, mask,edge,body
        else:
            image, mask= self.normalizeT(image, mask)
            image, mask= self.resizeT(image, mask)
            image, mask = self.totensorT(image, mask)
            return image, mask, shape, name

    def collate(self, batch):
        #size =[256, 288, 320, 352][np.random.randint(0, 4)]#[320, 352, 384, 416, 448][np.random.randint(0, 5)]#d
        size = 256
        image, mask,edge,body = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i] = cv2.resize(edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            body[i] = cv2.resize(body[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        body = torch.from_numpy(np.stack(body, axis=0)).unsqueeze(1)
        return image, mask, edge,body

    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='../data/DUTS')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
