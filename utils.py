import numpy as np
import os
from PIL import Image
import torch
from torchvision.transforms import functional as F
import math
from options import args

eps = torch.finfo(torch.float64).eps


# ******************************计算模型参数******************************
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


# ******************************梯度裁剪******************************
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# ******************************学习率衰减******************************
def adjust_lr(optimizer, epoch, decay_rate=0.9, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    print("lr=", optimizer.param_groups[0]["lr"])

    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, final_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        self.final_lr = final_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred=0.0):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = self.final_lr + 0.5 * (self.lr - self.final_lr) * (
                        1 + math.cos(1.0 * (T - self.warmup_iters) / (self.N - self.warmup_iters) * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'linear':
            if T < self.warmup_iters:
                lr = self.lr
            else:
                lr = self.lr * 1.0 * (2 - T / self.warmup_iters)
        else:
            raise NotImplemented

        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr * 0.1
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            # ******************************图片保存******************************


def Save_result(img, frame_image_path):
    path_split = frame_image_path.split("/")

    image_save_path = os.path.join(args.result_path, path_split[3], path_split[4])

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    image_save_path = image_save_path + '/' + path_split[-1][:-4] + '.png'

    img = F.to_pil_image(img)
    img.save(image_save_path)
