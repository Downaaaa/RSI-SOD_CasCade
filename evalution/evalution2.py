import numpy as np
import torch
import py_sod_metrics as M
from tqdm import tqdm
import cv2
import os
import torch.nn.functional as F
FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()

mask_root = '/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test-labels/'
pred_root = '/data/Hms/resultmap/out11-1-1/testp2'  # '/data/Hms/resultmap/out10-20-3/testp4'
mask_name_list = sorted(os.listdir(mask_root))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_path = os.path.join(mask_root, mask_name)
    # mask_name = mask_name.split('.')[0] + '_SASOD.png'
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    # pred = torch.from_numpy(pred)
    # pred = torch.unsqueeze(pred, dim=0)
    # pred = torch.unsqueeze(pred, dim=0)
    # # print(mask.shape)
    # pred = F.interpolate(pred, size=mask.shape, mode='bilinear')
    # pred = torch.squeeze(pred,0)
    # pred = torch.squeeze(pred, 0)
    # pred = tor_arr.numpy(pred)
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

fm = FM.get_results()['fm']
wfm = WFM.get_results()['wfm']
sm = SM.get_results()['sm']
em = EM.get_results()['em']
mae = MAE.get_results()['mae']

print(
    'Smeasure:', sm.round(4), '; ',
    'wFmeasure:', wfm.round(4), '; ',
    'MAE:', mae.round(5), '; ',
    'adpEm:', em['adp'].round(4), '; ',
    'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(4), '; ',
    'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(4), '; ',
    'adpFm:', fm['adp'].round(4), '; ',
    'meanFm:', fm['curve'].mean().round(4), '; ',
    'maxFm:', fm['curve'].max().round(4),


    sep=''
)
