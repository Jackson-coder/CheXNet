# -*- coding:utf-8 -*-
"""
author: win10
date: 2021-11-24
"""

import torch
import torchvision
from torchsummary import summary

class DenseBlock(torch.nn.Module):
    '''
    input_features: 输入特征图通道数
    bn_size: DenseNet-B 结构的优化参数，使用1*1卷积核
    growth_rate: 增长率k
    '''

    def __init__(self, input_features=64, growth_rate=32, bn_size=4):
        super(DenseBlock, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(input_features)
        self.conv1 = torch.nn.Conv2d(input_features, bn_size * growth_rate, kernel_size=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = torch.nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=(3, 3), padding=1)

    def __call__(self, x):
        x = self.bn1(x)
        x = torch.nn.ReLU(inplace=True)(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = torch.nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = torch.nn.Dropout(0.5)(x)
        return x


class DenseBlockBody(torch.nn.Module):
    '''
        layer_nums: DenseBlock层数
        num_init_features: 输入特征图通道数
        growth_rate: 增长率k
    '''

    def __init__(self, layer_nums, num_init_features, growth_rate):
        super(DenseBlockBody, self).__init__()
        self.features = torch.nn.ModuleDict()

        for i in range(layer_nums):
            init_features = num_init_features + growth_rate * i
            new_features = DenseBlock(init_features, growth_rate)
            self.features.add_module("dense%d" % (i + 1), new_features)
            # x = torch.cat([x, new_features], dim=1)

    def __call__(self, x):
        for name, layer in self.features.items():
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x


class Transition(torch.nn.Module):
    '''
    input_features: 输入特征图通道数
    output_features: 输出特征图通道数
    '''

    def __init__(self, input_features, output_features):
        super(Transition, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(input_features)
        self.conv1 = torch.nn.Conv2d(input_features, output_features, kernel_size=(1, 1))
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.input_features = input_features
        self.output_features = output_features

    def __call__(self, x):
        x = self.bn1(x)
        x = torch.nn.ReLU(inplace=True)(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x


class DenseNet121(torch.nn.Module):
    '''
        input_layer: 输入层图像数据
        num_init_features: 输入特征图通道数
        growth_rate: 增长率k
    '''

    def __init__(self, num_init_features=64, growth_rate=32, class_num=14):
        super(DenseNet121, self).__init__()
        self.config = [6, 12, 24, 16]

        self.conv = torch.nn.Conv2d(3, num_init_features, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = torch.nn.BatchNorm2d(num_init_features)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.features = torch.nn.Sequential()

        for i, layer_num in enumerate(self.config):
            block = DenseBlockBody(layer_num, num_init_features, growth_rate)
            self.features.add_module('block%d' % (i + 1), block)
            num_init_features = num_init_features + growth_rate * layer_num
            if i != 3:
                trans = Transition(num_init_features, num_init_features // 2)
                self.features.add_module('trans%d' % (i + 1), trans)
                # print(x.shape)
                num_init_features = num_init_features // 2

        self.bn2 = torch.nn.BatchNorm2d(num_init_features)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(7, 7), stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(num_init_features, class_num)
        self.sigmoid = torch.nn.Sigmoid()

    def __call__(self, input_layer):
        x = self.conv(input_layer)
        # print(x.shape)
        x = self.bn1(x)
        x = torch.nn.ReLU(inplace=True)(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.features(x)
        x = self.bn2(x)
        x = torch.nn.ReLU(inplace=True)(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class DenseNet121_torch_version(torch.nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(DenseNet121_torch_version, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, out_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32).cuda()
    model = DenseNet121().cuda()
    print(model)
    output = model(input).cuda()
    print(output.shape)
    summary(model, (3, 224, 224), batch_size=-1, device='cuda')
