import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import  torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import shutil

class ResentBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super.__init__()
        #padding 左右填充数
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False)
        #批标准化
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out),inplace=False)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), inplace=False)
        out += residual
        return F.relu(out)


