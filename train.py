import sys
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
#from BBDGE import BBBNet
#from BBDGE2 import BBBNet
#from BBDGE3 import BBBNet
from CasCadeUR import BBBNet
#from BBRNet import BBBNet
from apex import amp
import random
import pytorch_iou
def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    model.cfg.mode='test'
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask= image.cuda().float(), mask.cuda().float()
            p0, p1, p2, p3 = model(image)   #,pred1,pred2,pred3,edge,body
            pred = torch.sigmoid(p3[0, 0])
            avg_mae += torch.abs(pred - mask[0]).mean()

    model.train(True)
    model.cfg.mode='train'
    # print('validating MAE:', (avg_mae / nums).item())
    return (avg_mae / nums).item()

def structure_loss(pred, mask,weit=1):
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def bce_iou_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(Dataset, Network):

    # dataset
    cfg = Dataset.Config(datapath='/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/train', savepath='/data/Hms/model_Cas/CasCade-11-3-1',mode='train', batch=8, lr=0.04, momen=0.9,decay=5e-4, epoch=100)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)

    # val dataloader
    val_cfg1 = Dataset.Config(datapath='/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test', mode='test')
    val_data1 = Dataset.Data(val_cfg1)
    val_loader1 = DataLoader(val_data1, batch_size=1, shuffle=False, num_workers=8)

    # val mse
    # min_mse1 = 1.0
    min_mse2 = 1.0
    best_epoch = cfg.epoch

    # network
    net = Network(cfg)
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
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O0')
    # sw = SummaryWriter(cfg.savepath)
    # lambda1 = lambda epoch: epoch // 30

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50], gamma=0.1)
    global_step = 0

    max_itr = cfg.epoch * len(loader)
    CE = torch.nn.BCEWithLogitsLoss()
    IOU = pytorch_iou.IOU(size_average=True)
    BCE = torch.nn.BCELoss()
    ToLine = nn.Sigmoid()
    for epoch in range(cfg.epoch):
        if epoch < 32:
            optimizer.param_groups[0]['lr'] = (1-abs((epoch + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
            optimizer.param_groups[1]['lr'] = (1-abs((epoch + 1) / (32 + 1) * 2 - 1)) * cfg.lr
        else:
            if epoch < 64:
                if epoch % 2 == 0:
                    optimizer.param_groups[0]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1*0.5
                    optimizer.param_groups[1]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr*0.5
                else:
                    optimizer.param_groups[0]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1*0.5
                    optimizer.param_groups[1]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr*0.5
            else:
                if epoch<80:
                    if epoch % 2 == 0:
                        optimizer.param_groups[0]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1 * 0.1
                        optimizer.param_groups[1]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
                    else:
                        optimizer.param_groups[0]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1 * 0.1
                        optimizer.param_groups[1]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
                else:
                    if epoch % 2 == 0:
                        optimizer.param_groups[0]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1 * 0.1 * 0.5
                        optimizer.param_groups[1]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1 * 0.5
                    else:
                        optimizer.param_groups[0]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1 * 0.1 * 0.5
                        optimizer.param_groups[1]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1 * 0.5


        for step, (image, mask, edge,body) in enumerate(loader):
            image, mask ,edge,body= image.float().cuda(), mask.float().cuda(), edge.float().cuda(), body.float().cuda()
            p1, p2 = net(image, mask=mask)
            #loss1 = CE(p0,mask)+IOU(ToLine(p0),mask)
            loss1 = CE(p1,mask)  # +IOU(ToLine(p1),mask)
            loss2 = CE(p2,mask)+IOU(ToLine(p2),mask)
            #loss4 = CE(p3,mask)+ IOU(ToLine(p3),mask)
            loss =loss1 +loss2  #+ loss3 + loss4  
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            # scheduler.step()
            global_step += 1
            if (step+1) % 175 == 0:
                print('%s | step:%d/%d | lr=%.6f  loss=%.6f loss1=%.6f loss2=%.6f ' % (datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[1]['lr'],loss,loss1,loss2))

        if epoch >= 9:
            # mse1 = validate(net, val_loader1, 485)
            mse2 = validate(net, val_loader1, 600)
            # mse3 = validate(net, val_loader3, 300)
            print('EORSSD:%s' % (mse2))     #利用MSE来判断最好的几个模型
            if mse2 <= min_mse2:
                min_mse2 = mse2
                best_epoch = epoch + 1
            print('best epoch is ', best_epoch, min_mse2)
            if (epoch >= 27 and best_epoch == epoch + 1) or epoch == 79 or epoch == 149:
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
            # torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    train(dataset, BBBNet)
