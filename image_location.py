import torch
import torch.nn as nn
from PIL import Image
from torch.utils import data
import torch.nn.functional as F
import  torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision
import os
import shutil
import glob
from torchvision import transforms
from lxml import etree

batch_size = 8
pil_img = Image.open(r"dataset/images/Abyssinian_1.jpg")
print(np.array(pil_img).shape)
xml = open(r"dataset/annotations/xmls/Abyssinian_1.xml").read()
print(xml)
sel = etree.HTML(xml)
width = sel.xpath("//size/width/text()")[0]
print(width)