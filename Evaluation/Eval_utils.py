import os
import time
import numpy as np
import torch
from torchvision import transforms

eps = torch.finfo(torch.float64).eps

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------material--------------------
def _eval_pr(y_pred, y, num):

    prec, recall = torch.zeros(num).to(device), torch.zeros(num).to(device)
    thlist = torch.linspace(0, 1 - 1e-10, num).to(device)

    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + eps), tp / (y.sum() + eps)
    return prec, recall


def _eval_e(y_pred, y, num):
    score = torch.zeros(num).to(device)
    thlist = torch.linspace(0, 1 - 1e-10, num).to(device)

    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + eps)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + eps)
    return score


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + eps)

    return score


def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg

    return Q


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1).to(device) * round(cols / 2)
        Y = torch.eye(1).to(device) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.from_numpy(np.arange(0, cols)).to(device).float()
        j = torch.from_numpy(np.arange(0, rows)).to(device).float()

        X = torch.round((gt.sum(dim=0) * i).sum() / total + eps)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total + eps)

    return X.long(), Y.long()


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + eps)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + eps)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + eps)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + eps)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q


# --------------------Eval--------------------
def Eval_mae(data):
    avg_mae, img_num = 0.0, 0.0
    with torch.no_grad():
        for pred, gt in data:
            pred.to(device)
            gt.to(device)

            mea = torch.abs(pred - gt).mean()

            if mea == mea:
                avg_mae += mea
                img_num += 1.0

        avg_mae /= img_num

        return avg_mae


def Eval_fmeasure(data):

    beta2 = 0.3
    avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for pred, gt in data:
            pred.to(device)
            gt.to(device)

            pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                               torch.min(pred) + eps)

            prec, recall = _eval_pr(pred, gt, 255)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for Nan
            avg_f += f_score
            avg_p += prec
            avg_r += recall
            img_num += 1.0

        Fm = avg_f / img_num
        avg_p = avg_p / img_num
        avg_r = avg_r / img_num
        return Fm, avg_p, avg_r


def Eval_Emeasure(data):
    avg_e, img_num = 0.0, 0.0
    with torch.no_grad():
        Em = torch.zeros(255).to(device)
        for pred, gt in data:
            pred.to(device)
            gt.to(device)

            pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                               torch.min(pred) + eps)
            Em += _eval_e(pred, gt, 255)
            img_num += 1.0

        Em /= img_num
        return Em


def Eval_Smeasure(data):
    alpha, avg_q, img_num = 0.5, 0.0, 0.0

    with torch.no_grad():
        for pred, gt in data:
            pred.to(device)
            gt.to(device)
            pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                               torch.min(pred) + eps)
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x

                img_num += 1.0
                avg_q += Q

            elif y == 1:
                x = pred.mean()
                Q = x

                img_num += 1.0
                avg_q += Q

            else:
                gt[gt >= 0.5] = 1
                gt[gt < 0.5] = 0
                Q = alpha * _S_object(pred, gt) + (1 - alpha) * _S_region(
                    pred, gt)

                img_num += 1.0
                avg_q += Q
                
                if Q.item() < 0:
                    img_num += 1.0
                    avg_q += 0.0

        avg_q /= img_num
        return avg_q
