from asyncio.log import logger
import os

import torch
import time
from model.model import Model
from dataload.dataload import VideoDataset
from torch.utils.data import DataLoader
from dataload.transforms import get_train_transforms, get_transforms
from datetime import datetime
from Loss import multi_loss
import torch.optim as optim
from utils import LR_Scheduler, clip_gradient
from test import test
from tqdm import tqdm
from torch.utils import tensorboard
import logging
from options import args


# 开始训练
def train(train_data, model, optimizer, scheduler, Epoch):
    model.train()

    total_step = len(train_data)

    total_losses = 0.0
    already_time = 0.0

    for i, packs in enumerate(train_data):

        start_time = time.time()

        i = i + 1

        optimizer.zero_grad()
        lr = scheduler(optimizer, i - 1, Epoch - 1)

        images, gts, not_gts = [], [], []
        for pack in packs:
            image, gt, not_gt = pack["image"], pack["gt"], pack['not_gt']
            images.append(image.to(device))
            gts.append(gt.to(device))
            not_gts.append(not_gt.to(device))

        images = torch.cat(images, dim=0)
        gts = torch.cat(gts, dim=0)
        not_gts = torch.cat(not_gts, dim=0)

        # 解码
        foreground, background, out = model(images)

        # Loss 计算
        loss_out = multi_loss(out, gts)
        loss_foreground = multi_loss(foreground, not_gts)
        loss_background = multi_loss(background, gts)

        loss = loss_out + loss_background + loss_foreground

        total_losses += loss.data

        # 反向传播
        loss.backward()

        clip_gradient(optimizer, args.clip)
        optimizer.step()

        speed = time.time() - start_time
        already_time += speed

        # 显示
        if i % 5 == 0 or i == total_step:
            print('{},Epoch:{:02d}/{:02d},Step:{:04d}/{:04d},Loss:{:0.4f},Lr:{:0.4f},time:{:0.2f}/{:0.2f}min\n'
                  .format(datetime.now().strftime('%H:%M'), Epoch, args.total_epoch, i, total_step, total_losses / i,
                          lr * 1e6,
                          already_time / 60.0, (total_step - i) * speed / 60.0))

    return model


# **************************************************************************************************
# **************************************************************************************************

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("start model training!!!")

model = Model()

# 训练集加载
train_transforms = get_train_transforms(input_size=(args.size, args.size))
train_dataset = VideoDataset(root_dir=args.train_path, train_set_list=args.train_dataset, training=True,
                             transforms=train_transforms, clip_len=args.clip_len)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, drop_last=True, pin_memory=True)

# 测试集加载
test_transforms = get_transforms(input_size=(args.size, args.size))
test_dataset = VideoDataset(root_dir=args.test_path, train_set_list=args.test_dataset, training=True,
                            transforms=test_transforms, clip_len=args.clip_len)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, drop_last=True, pin_memory=True)

# 加载至cuda
model.to(device)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
scheduler = LR_Scheduler(mode='cos', base_lr=args.base_lr, final_lr=args.final_lr, num_epochs=args.total_epoch,
                         iters_per_epoch=len(train_dataloader))

# 加载模型
path = args.model_path + '/best_image_model.pth'
if os.path.exists(path):
    model_dict = model.state_dict()
    checkpoint = torch.load(path, map_location=torch.device(device))
    pretrained_dict = checkpoint

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('*' * 20, 'Load best_image_model Done !!! ', '*' * 20)

if __name__ == '__main__':

    logging.basicConfig(filename=args.log_dir + '/train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    writer = tensorboard.SummaryWriter(args.log_dir + '/train')

    for epoch in range(1, args.total_epoch + 1):
        training_model = train(train_dataloader, model, optimizer, scheduler, epoch)

        S_lambda, F_beta, Mae = test(test_dataloader, training_model)

        val = S_lambda + F_beta + (1 - Mae)

        writer.add_scalar(tag='val/S_lambda', scalar_value=S_lambda, global_step=epoch)
        writer.add_scalar(tag='val/F_beta', scalar_value=F_beta, global_step=epoch)
        writer.add_scalar(tag='val/Mae', scalar_value=Mae, global_step=epoch)
        writer.add_scalar(tag='val/total_val', scalar_value=val, global_step=epoch)

        print('Epoch = {}, the result are: S_lambda = {:0.4f}, F_beta = {:0.4f}, Mae = {:0.4f}, Val = {:0.4f}'
              .format(epoch, S_lambda, F_beta, Mae, val))

        logging.info('Epoch = {}, the result are: S_lambda = {:0.4f}, F_beta = {:0.4f}, Mae = {:0.4f}, Val = {:0.4f}'
                     .format(epoch, S_lambda, F_beta, Mae, val))

        torch.save(training_model.state_dict(),
                   args.model_path + '/best_{}_model_{}.pth'.format(args.train_type, epoch))

    writer.close()
