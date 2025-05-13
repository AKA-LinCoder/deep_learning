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


images = glob.glob(r"dataset/images/*.jpg")
anno = glob.glob(r"dataset/annotations/xmls/*.xml")
#标签和图片不一一对应，
xml_name = [x.split("/")[-1].replace(".xml","") for x in anno]
images = [x for x in images if x.split("/")[-1].replace(".jpg","") in xml_name]
print(len(images))
print(len(xml_name))

def to_labels(path):
    xml = open(r"{}".format(path)).read()
    sel = etree.HTML(xml)
    width = int(sel.xpath("//size/width/text()")[0])
    height = int(sel.xpath("//size/height/text()")[0])
    ymax = int(sel.xpath("//bndbox/ymax/text()")[0])
    ymin = int(sel.xpath("//bndbox/ymin/text()")[0])
    xmax = int(sel.xpath("//bndbox/xmax/text()")[0])
    xmin = int(sel.xpath("//bndbox/xmin/text()")[0])
    return  [xmin/width,ymin/height,xmax/width,ymax/height]

labels = [to_labels(p) for p in anno]

index = np.random.permutation(len(images))
images = np.array(images)[index]
labels = np.array(labels)[index]

labels = labels.astype(np.float32)

i = int(len(images) * 0.8)
train_imgs = images[:i]
train_labels = labels[:i]
test_imgs = images[i:]
test_labels = labels[i:]
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class OXford_dataset(data.Dataset):
    def __init__(self,images_path,label_list):
        self.imgs = images_path
        self.labels = label_list
    def __getitem__(self, index):
        img = self.imgs[index]
        pil_img = Image.open(img)
        img_tensor = transform(pil_img)
        l1,l2,l3,l4 = self.labels[index]
        return  img_tensor,l1,l2,l3,l4
    def __len__(self):
        return len(self.imgs)

train_ds = OXford_dataset(train_imgs,train_labels)
test_ds = OXford_dataset(test_imgs,test_labels)

