# -*- coding:utf-8 -*-
"""
author: win10
date: 2021-11-26
"""
import numpy as np
import torch
import csv
from dataset import CheXDataset, data_generate
from torch.utils import data
from net.model import DenseNet121
from tqdm import tqdm
from pytorchtools import EarlyStopping
from torch.optim import lr_scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    readFile = open("train.csv", "r", newline='')
    reader = csv.reader(readFile)

    train_images, train_labels, val_images, val_labels = data_generate(reader)
    num_train = len(train_labels)
    num_val = len(val_labels)
    train_batch_size = 128

    trainDataset = CheXDataset(train_images, train_labels)
    valDataset = CheXDataset(val_images, val_labels)
    trainData_iter = data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True)
    valData_iter = data.DataLoader(valDataset, batch_size=1, shuffle=True)

    model_loss = torch.nn.CrossEntropyLoss()
    model = DenseNet121()
    model.to(device)


    epoches = 1000
    epoch_step_train = num_train // train_batch_size
    epoch_step_val = num_val // train_batch_size

    patience = 10
    T_max = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience, verbose=True)
    lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0)

    for epoch in range(epoches):
        train_loss = 0
        val_loss = 0
        print('Start Training')
        model.train()
        # 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批
        # 数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        with tqdm(total=epoch_step_train, desc=f'Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(trainData_iter):  # batch:(X, y)
                if iteration > epoch_step_train:
                    break

                images, labels = batch

                outputs = model(images.to(device))

                loss = model_loss(outputs, labels.to(device))

                loss = loss / len(outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                pbar.set_postfix(**{'loss': train_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        print('Finished Training')

        print('Start Validation')
        model.eval()
        # 如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值
        # 和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval() 是利用到了所有网络连接，即不进行随机舍弃神经元。
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(valData_iter):  # batch:(X, y)
                if iteration > epoch_step_val:
                    break

                images, labels = batch

                outputs = model(images.to(device))

                loss = 0
                for i in range(len(outputs)):
                    loss += model_loss(outputs[i], labels[i].to(device))
                loss = loss / len(outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                val_loss += loss
                pbar.set_postfix(**{'loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        print('Finished Validation')

        print('Epoch:' + str(epoch + 1) + '/' + str(epoches))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss / epoch_step_train, val_loss / epoch_step_val))

        if epoch % 10 == 0:
            torch.save(model, 'weights/CheXNet_Val_Loss:_%.3f.pth' % (val_loss / epoch_step_val))

        lr_scheduler.step()
        early_stopping(val_loss / epoch_step_val, model)
        if early_stopping.early_stop:
            break

    torch.save(model, 'weights/CheXNet.pth')



# test123
