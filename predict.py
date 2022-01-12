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
from net.model import DenseNet121

if __name__ == '__main__':
    root = "/kaggle/input/data/images_003/images/00006199_010.png"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'CheXNet.pth'
    img = Image.open(root).convert('RGB')
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])
                                                ])
    img = transform(img)
    model = DenseNet121().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    result = model(img.to(device))
    print(result)

    im_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    cv2.imwrite("img.png", im_color)
