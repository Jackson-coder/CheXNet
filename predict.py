# -*- coding:utf-8 -*-
"""
author: win10
date: 2021-11-27
"""

import torch
from PIL import Image
import cv2
import numpy as np
import torchvision
from net.model import DenseNet121, DenseNet121_torch_version

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
               'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
               'Pleural_Thickening', 'Hernia']

if __name__ == '__main__':
    root = "data_pic/00000012_000.png"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(root).convert('RGB')
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])
                                                ])
    img = transform(img)
    model_path = 'weights/CheXNet.pth'
    model = torch.load(model_path)
    img = img.unsqueeze(0)
    model = model.eval()
    results = model(img.to(device))

    tensor_one = torch.Tensor([1] * 14).to(device)
    tensor_zero = torch.Tensor([0] * 14).to(device)
    results = torch.where(results > 0.5, tensor_one, tensor_zero)
    results = results[0]


    im_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    t = 0
    for n in range(len(results)):
        if results[n] == 1:
            print(CLASS_NAMES[n])
            cv2.putText(im_color, str(CLASS_NAMES[n]), (100, 100 + t), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 8)
            t += 100

    cv2.imwrite("img1.png", im_color)
