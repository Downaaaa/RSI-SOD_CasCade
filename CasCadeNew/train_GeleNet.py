import sys
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from GeleNet.model.GeleNet_models import GeleNet
from apex import amp
import random
from utils import adjust_lr,clip_gradient
from torch.autograd import Variable
import pytorch_iou
import json
def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask= image.cuda().float(), mask.cuda().float()
            p1,pred = model(image)    # predd1, predd2, predd3, predd4
            pred = pred[0, 0]
            avg_mae += torch.abs(pred - mask[0]).mean()

    model.train(True)

    # print('validating MAE:', (avg_mae / nums).item())
    return (avg_mae / nums).item()

def train(Dataset, Network):

    # dataset
    cfg = Dataset.Config(datapath='/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/train', savepath='/data/Hms/model_newcas/GeLeNet',mode='train', batch=16
                         , lr=1e-3, momen=0.9,
                         decay=5e-4, epoch=120, width=256, height=256)
    data = Dataset.Data(cfg)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True,
                        num_workers=0, pin_memory=True)   #, pin_memory=True

    # val dataloader
    val_cfg1 = Dataset.Config(datapath='/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test', mode='test', width=256, height=256)
    val_data1 = Dataset.Data(val_cfg1)
    val_loader1 = DataLoader(val_data1, batch_size=1, shuffle=False, num_workers=0)

    # val mse
    # min_mse1 = 1.0
    min_mse2 = 1.0
    best_epoch = cfg.epoch

    # network
    net = Network()
    net.train(True)

    # apex
    net = net.cuda()
    # device = torch.device('cuda')
    # net = nn.DataParallel(net)
    # net.to(device)

    # net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name or 'resnet' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.Adam(net.parameters(),cfg.lr)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O0')
    global_step = 0

    max_itr = cfg.epoch * len(loader)
    CE = torch.nn.BCEWithLogitsLoss()
    IOU = pytorch_iou.IOU(size_average=True)
    BCE = torch.nn.BCELoss()
    ToLine = nn.Sigmoid()
    jsonnum = {}
    lrnum = []
    MAEnum = []
    lossnum = []
    for epoch in range(cfg.epoch):
        adjust_lr(optimizer, cfg.lr, epoch, decay_rate=0.5, decay_epoch=20)
        for step, (image, mask, edge,body) in enumerate(loader):
            image, mask, edge, body =Variable(image), Variable(mask), Variable(edge), Variable(body)
            image, mask, edge, body = image.float().cuda(), mask.float().cuda(), edge.float().cuda(), body.float().cuda()
            p, p_sigmoid = net(image)  #, predd1, predd2, predd3, predd4
            loss = BCE(p_sigmoid, mask) + IOU(p_sigmoid, mask)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            # clip_gradient(optimizer, 0.5)
            optimizer.step()
            # scheduler.step()
            global_step += 1
            if (step+1) % 87 == 0:
                print('%s | step:%d/%d | lr=%.6f loss=%.6f' % (datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],loss.item()))
                lossnum.append(loss)
                lrnum.append(optimizer.param_groups[0]['lr'])
        if not os.path.exists(cfg.savepath):
            os.makedirs(cfg.savepath)
            print("ok")
        if epoch >= 5:
            # mse1 = validate(net, val_loader1, 485)
            mse2 = validate(net, val_loader1, 600)
            # mse3 = validate(net, val_loader3, 300)
            print('EORSSD:%s' % (mse2))     #利用MSE来判断最好的几个模型
            MAEnum.append(mse2)
            if mse2 <= min_mse2:
                min_mse2 = mse2
                best_epoch = epoch + 1
            print('best epoch is ', best_epoch, min_mse2)
            if (epoch >= 25 and best_epoch == epoch + 1) or epoch == 31 or epoch == 79:

                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
                if epoch == 119:
                    jsonnum['lr'] = lrnum
                    jsonnum['loss'] = lossnum
                    jsonnum['mse'] = MAEnum
                    with open('GeLeNetdata.json', 'w') as f:
                        json.dump(jsonnum, f)
             # torch.cuda.empty_cache()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    train(dataset, GeleNet)