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
np_img = np.array(pil_img)
print(np.array(pil_img).shape)
xml = open(r"dataset/annotations/xmls/Abyssinian_1.xml").read()
print(xml)
sel = etree.HTML(xml)
width = int(sel.xpath("//size/width/text()")[0])
height = int(sel.xpath("//size/height/text()")[0])
print(width)

ymax = int(sel.xpath("//bndbox/ymax/text()")[0])

ymin = int(sel.xpath("//bndbox/ymin/text()")[0])
xmax = int(sel.xpath("//bndbox/xmax/text()")[0])
xmin = int(sel.xpath("//bndbox/xmin/text()")[0])



plt.imshow(np_img)

rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color="red")
ax = plt.gca()
ax.axes.add_patch(rect)
plt.show()

img = pil_img.resize((224,224))
xmin = (xmin/width)*224
ymin = (ymin/height)*224
xmax= (xmax/width)*224
ymax = (ymax/height)*224


plt.imshow(img)

rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color="blue")
ax = plt.gca()
ax.axes.add_patch(rect)
plt.show()