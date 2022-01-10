# -*- coding:utf-8 -*-
"""
author: win10
date: 2021-11-26
"""
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


def data_generate(reader):
    images = []
    labels = []

    start = True
    for row in reader:
        if start:
            start = False
            continue
        images.append(row[0])
        labels.append([int(row[i + 1]) for i in range(14)])

    images = np.array(images)
    labels = np.array(labels)
    train_images = images[:80000]
    train_labels = labels[:80000]
    test_images = images[80000:]
    test_labels = labels[80000:]

    return train_images, train_labels, test_images, test_labels


class CheXDataset(Dataset):
    def __init__(self, images, labels):
        super(CheXDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])
                                                         ])

    def __len__(self):
        return len(self.images)

    # 可以使用水平翻转做数据增强
    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        image = self.transform(image)

        label = self.labels[item]
        label = np.array(label, dtype=np.float32)
        label = torch.Tensor(label)
        return image, label
