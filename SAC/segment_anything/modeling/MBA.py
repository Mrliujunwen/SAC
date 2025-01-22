#!/usr/bin/env python

# -- coding: utf-8 --
# @Time : 2024/12/5 20:22
# @Author : junwen Liu
# @Site : 
# @File : MBA.py
# @Email   : junwenLiu0201@126.com
# @Software: PyCharm

import torch
import torch.nn as nn
# from .backbone import resnet18
from .CBAM import CBAM
# from block import MEEM

class diff_moudel(nn.Module):
    def __init__(self,in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        # self.simam = simam_module()
        self.cbam = CBAM(in_channel)
    def forward(self, x):
        x = self.cbam.forward(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        return out
class MBA(nn.Module):
    def __init__(self,in_channel):
        super(MBA, self).__init__()

        self.diff_1 = diff_moudel(in_channel)


    def forward(self,x1):
        d1 = self.diff_1(x1)
        # d2 = self.diff_2(x2)


        return d1


# class MBA(nn.Module):
#     def __init__(self):
#         super(MBA, self).__init__()
#         self.CBM1 = CBM(128)
#         self.CBM2 = CBM(256)
#         self.CBM3 = CBM(512)
#         # self.MEEM1 = MEEM(3, 128, 16, 4, norm=nn.BatchNorm2d, act=nn.ReLU)
#         # self.MEEM2 = MEEM(3, 256, 16, 4, norm=nn.BatchNorm2d, act=nn.ReLU)
#         # self.MEEM3 = MEEM(3, 512, 16, 4, norm=nn.BatchNorm2d, act=nn.ReLU)
#         self.conv = nn.Conv2d(256,256,1,1,0)
#         self.avg_pool3= nn.AvgPool2d((3, 3), stride=1, padding=1)
#         self.avg_pool2 = nn.AvgPool2d((3, 3), stride=2, padding=1)
#         self.avg_pool1 = nn.AvgPool2d((4, 4), stride=4, padding=1)
#         self.pool = nn.AdaptiveAvgPool2d(16)
#         self.conv1 = nn.Conv2d(128,256,1,1,0)
#         self.conv2 = nn.Conv2d(256,256,1,1,0)
#         self.conv3 = nn.Conv2d(512,256,1,1,0)
#         self.cbam = CBAM(256)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu = nn.ReLU()
#
#     def forward(self,x):
#         # ou = self.inp.forward(x)
#         out1 = self.MEEM1(x)
#         out2 = self.MEEM2(x)
#         out3 = self.MEEM3(x)
#         out3 = self.CBM3.forward(out3)
#         out2 = self.CBM2.forward(out2)
#         out1 = self.CBM1.forward(out1)
#         out1 =self.pool(out1)
#         out2 =self.pool(out2)
#         out3 =self.pool(out3)
#
#         out1 = self.conv1(out1)
#         out1 = self.bn1(out1)
#         out1 = self.relu(out1)
#
#         out2 = self.conv2(out2)
#         out2 = self.bn1(out2)
#         out2 = self.relu(out2)
#
#         out3 = self.conv3(out3)
#         out3 = self.bn1(out3)
#         out3 = self.relu(out3)
#         out2_1 =out2-out1
#         out2_3 =out2-out3
#         x = out2_3+out2_1
#
#         x = self.cbam(x)
#
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         return x
