import torch
import torch.nn as nn
from PIL import Image
from torch.utils import data
import torch.nn.functional as F
import  torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import shutil
import glob
from torchvision import transforms

#定义基础基础卷积模型：卷积+B呢n+激活
class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels, **kwargs):
        super(BasicConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return  F.relu(x,inplace=True)

class InceptionBlock(nn.Module):
    def __init__(self,in_channels,pool_features):
        super(InceptionBlock,self).__init__()
        self.b_1x1 = BasicConv(in_channels,64,kernel_size=1)
        self.b_3x3_1 = BasicConv(in_channels, 64, kernel_size=1)
        self.b_3x3_2 = BasicConv(64, 96, kernel_size=3,padding=1)
        self.b_5x5_1 = BasicConv(in_channels, 48, kernel_size=1)
        self.b_5x5_2 = BasicConv(48, 48, kernel_size=5, padding=2)

        self.b_pool = BasicConv(in_channels,pool_features,kernel_size=1)
    def forward(self,x):
        b_1x1_out = self.b_1x1(x)

        b_3x3 = self.b_3x3_1(x)
        b_3x3_out = self.b_3x3_2(b_3x3)

        b_5x5 = self.b_5x5_1(x)
        b_5x5_out = self.b_5x5_2(b_5x5)

        b_pool_out = F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        b_pool_out = self.b_pool(b_pool_out)

        out_put = [b_1x1_out,b_3x3_out,b_5x5_out,b_pool_out]
        return torch.cat(out_put,dim=1)

my_inception_block = InceptionBlock(32,64)
