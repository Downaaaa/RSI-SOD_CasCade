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
from CasCadeUFOUR.stackNetPVTNetV1 import StackedHGNetV1
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
            (p1,p2,p3,pred) = model(image)    # predd1, predd2, predd3, predd4
            pred = torch.sigmoid(pred[0, 0])
            avg_mae += torch.abs(pred - mask[0]).mean()

    model.train(True)

    # print('validating MAE:', (avg_mae / nums).item())
    return (avg_mae / nums).item()
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        #elif isinstance(obj, array):
        #    return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def train(Dataset, Network):
    # dataset
    cfg = Dataset.Config(datapath='/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/train', savepath='/data/Hms/model_newcas/CasCadeUFour-12-8-1',mode='train', batch=8
                         , lr=1e-3, momen=0.9,
                         decay=5e-4, epoch=600, width=256, height=256)
    data = Dataset.Data(cfg)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True,
                        num_workers=0, pin_memory=True)   #, pin_memory=True

    # val dataloader
    val_cfg1 = Dataset.Config(datapath='/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test', mode='test', width=256, height=256)
    val_data1 = Dataset.Data(val_cfg1)
    val_loader1 = DataLoader(val_data1, batch_size=1, shuffle=False, num_workers=0)

    min_mse2 = 1.0
    best_epoch = cfg.epoch

    net = Network(cfg, 1)
    net.train(True)

    # apex
    net = net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    global_step = 0

    max_itr = cfg.epoch * len(loader)
    CEW = torch.nn.BCEWithLogitsLoss()
    IOU = pytorch_iou.IOU(size_average=True)
    BCE = torch.nn.BCELoss()
    ToLine = nn.Sigmoid()
    jsonnum = {}
    lrnum = []
    MAEnum = []
    lossnum = []
    for epoch in range(cfg.epoch):
        if epoch < 100:
            adjust_lr(optimizer, 1e-6, epoch, decay_rate=10, decay_epoch=20)
            
        else:
            if epoch < 400:
                    optimizer.param_groups[0]['lr'] = cfg.lr
            else:
                    adjust_lr(optimizer, cfg.lr, epoch-350, decay_rate=0.5, decay_epoch=50)
        for step, (image, mask, edge,body) in enumerate(loader):
            # image, mask, edge, body =Variable(image), Variable(mask), Variable(edge), Variable(body)
            image, mask, edge, body = image.float().cuda(), mask.float().cuda(), edge.float().cuda(), body.float().cuda()
            # print(image.shape, mask.shape)
            (p1,p2,p3,p4) = net(image)  #, predd1, predd2, predd3, predd4
            # print(torch.unique(p1))
            # print(torch.unique(p2))
            # print(torch.unique(p3))
            # print(torch.unique(p4))
            loss = CEW(p1, mask) + IOU(ToLine(p1), mask) + CEW(p2, mask) + IOU(ToLine(p2), mask) + CEW(p3, mask) + IOU(ToLine(p3), mask) + CEW(p4, mask) + IOU(ToLine(p4), mask)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            # clip_gradient(optimizer, 0.5)
            optimizer.step()
            # scheduler.step()
            global_step += 1
            if (step+1) % 175 == 0:
                print('%s | step:%d/%d | lr=%.6f loss=%.6f' % (datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],loss.item()))
                lossnum.append(loss)
                lrnum.append(optimizer.param_groups[0]['lr'])
        if not os.path.exists(cfg.savepath):
            os.makedirs(cfg.savepath)
            print("ok")
        if epoch >= 65:
            # mse1 = validate(net, val_loader1, 485)
            mse2 = validate(net, val_loader1, 600)
            # mse3 = validate(net, val_loader3, 300)
            print('EORSSD:%s' % (mse2))     #利用MSE来判断最好的几个模型
            MAEnum.append(mse2)
            if mse2 <= min_mse2:
                min_mse2 = mse2
                best_epoch = epoch + 1
            print('best epoch is ', best_epoch, min_mse2)
            if (epoch >= 25 and best_epoch == epoch + 1) or epoch == 41 or epoch == 599:

                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
                if epoch == 599:
                    jsonnum['lr'] = lrnum
                    jsonnum['loss'] = lossnum
                    jsonnum['mse'] = MAEnum
                    with open('/data/Hms/model_newcas/CasCadeUFour-12-8-1/data.json', 'w') as f:
                        json.dump(jsonnum, f, cls=MyEncoder,indent=4, ensure_ascii=False)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    train(dataset, StackedHGNetV1)