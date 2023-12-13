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
sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
FMv2 = M.FmeasureV2(
    metric_handlers={
        # 灰度数据指标
        "fm": M.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": M.FmeasureHandler(**sample_gray, beta=0.1),
        "pre": M.PrecisionHandler(**sample_gray),
        "rec": M.RecallHandler(**sample_gray),
        # "fpr": M.FPRHandler(**sample_gray),
        "iou": M.IOUHandler(**sample_gray),
        "dice": M.DICEHandler(**sample_gray),
        "spec": M.SpecificityHandler(**sample_gray),
        "ber": M.BERHandler(**sample_gray),
        "oa": M.OverallAccuracyHandler(**sample_gray),
        "kappa": M.KappaHandler(**sample_gray),
        # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均,计算的是整个的前后景的指标的平均
        "sample_bifm": M.FmeasureHandler(**sample_bin, beta=0.3),
        "sample_bif1": M.FmeasureHandler(**sample_bin, beta=1),
        "sample_bipre": M.PrecisionHandler(**sample_bin),
        "sample_birec": M.RecallHandler(**sample_bin),
        # "sample_bifpr": M.FPRHandler(**sample_bin),
        "sample_biiou": M.IOUHandler(**sample_bin),
        "sample_bidice": M.DICEHandler(**sample_bin),
        "sample_bispec": M.SpecificityHandler(**sample_bin),
        "sample_biber": M.BERHandler(**sample_bin),
        "sample_bioa": M.OverallAccuracyHandler(**sample_bin),
        "sample_bikappa": M.KappaHandler(**sample_bin),
        # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标，换了另一种方式计算整个前后景指标
        "overall_bifm": M.FmeasureHandler(**overall_bin, beta=0.3),
        "overall_bif1": M.FmeasureHandler(**overall_bin, beta=1),
        "overall_bipre": M.PrecisionHandler(**overall_bin),
        "overall_birec": M.RecallHandler(**overall_bin),
        # "overall_bifpr": M.FPRHandler(**overall_bin),
        "overall_biiou": M.IOUHandler(**overall_bin),
        "overall_bidice": M.DICEHandler(**overall_bin),
        "overall_bispec": M.SpecificityHandler(**overall_bin),
        "overall_biber": M.BERHandler(**overall_bin),
        "overall_bioa": M.OverallAccuracyHandler(**overall_bin),
        "overall_bikappa": M.KappaHandler(**overall_bin),
    }
)


mask_root = '/userHome/zy/Hms/ACCoNet/dataset/train_dataset/EORSSD/test-labels/'
pred_root = '/data/Hms/resultmapNew/CasUFour12-12-1/testp4'  # '/data/Hms/resultmap/out10-20-3/testp4'
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
    FMv2.step(pred=pred, gt=mask)

fm = FM.get_results()['fm']
wfm = WFM.get_results()['wfm']
sm = SM.get_results()['sm']
em = EM.get_results()['em']
mae = MAE.get_results()['mae']
fmv2 = FMv2.get_results()
"""第一种计算miou的方式是计算两个类的IOU然后取平均，第二种是计算所有的IOU，也就是获取所有的TP、TN、FP、FN然后计算的指标"""
print(
    'Smeasure:', sm.round(4), '; ',
    'wFmeasure:', wfm.round(4), '; ',
    'maxFm:', fm['curve'].max().round(4),'; ',
    'meanFm:', fm['curve'].mean().round(4), '; ',
    'adpFm:', fm['adp'].round(4), '; ',
    'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(4), '; ',
    'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(4), '; ',
    'adpEm:', em['adp'].round(4), '; ',
    'MAE:', mae.round(5), '; ',
    'maxiou:',fmv2["iou"]["dynamic"].max().round(4),'; ',
    'meaniou:',fmv2["iou"]["dynamic"].mean().round(4),'; ',
    "adpiou:", fmv2["iou"]["adaptive"].round(4),'; ',
    'miou:',fmv2["sample_biiou"]["binary"].round(4),'; ',
    'miou2:',fmv2["overall_biiou"]["binary"].round(4),'; ',
    'maxfm2:',fmv2["fm"]["dynamic"].max().round(4), '; ',
    'meanfm2:',fmv2["fm"]["dynamic"].mean().round(4), '; ',
    'adpfm2:',fmv2["fm"]["adaptive"].round(4), '; ',
    "sample_bifm:",fmv2["sample_bifm"]["binary"].round(4), '; ',
    "overall_bifm:", fmv2["overall_bifm"]["binary"].round(4),
    sep=''
)