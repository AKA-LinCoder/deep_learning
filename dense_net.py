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



images_path = glob.glob("dataset/birds/*/*.jpg")
# print(images_path[10].split("/")[1].split(".")[1])
all_labels_name = [ img.split("/")[2].split(".")[1] for img in images_path ]
print(all_labels_name[-5])

unique_labels = np.unique(all_labels_name)
label_to_index = dict((v,k) for k,v in enumerate(unique_labels))
index_to_label = dict((v,k) for k,v in label_to_index.items())

all_labels = [label_to_index.get(name) for name in all_labels_name]
np.random.seed(2021)
random_index = np.random.permutation(len(images_path))
#乱序 乱
images_path = np.array(images_path)[random_index]
all_labels = np.array(all_labels)[random_index]

i = int(len(images_path) *0.8)
train_path = images_path[:i]
train_labels = all_labels[:i]
test_path = images_path[i:]
test_lables = all_labels[i:]

tranform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class BirdsDataset(data.Dataset):
    def __init__(self,images_path,labels):
        self.images_path = images_path
        self.labels = labels
    def __getitem__(self, index):
        img = self.images_path[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        # 处理黑白图片
        np_img = np.asarray(pil_img,dtype=np.uint8)
        if len(np_img.shape)== 2:
            img_data = np.repeat(np_img[:,:,np.newaxis],3,axis=2)
            pil_img = Image.fromarray(img_data)
        img_tensor = tranform(pil_img)
        return img_tensor,label
    def __len__(self):
        return len(self.images_path)
train_ds = BirdsDataset(train_path,train_labels)
test_ds = BirdsDataset(test_path,test_lables)
batch_size = 32
train_dl = data.DataLoader(train_ds,batch_size=batch_size)
test_dl = data.DataLoader(test_ds,batch_size=batch_size)
img_batch,label_batch = next(iter(train_dl))
print(img_batch.shape)
plt.figure(figsize=(12,8))

for i,(img,label) in enumerate(zip(img_batch[:6],label_batch[:6])):
    img = img.permute(1,2,0).numpy()
    plt.subplot(2,3,i+1)
    plt.title(index_to_label.get(label.item()))
    plt.imshow(img)
plt.show()
#使用densenet提取特征
my_densenet = torchvision.models.densenet121(pretrained = True).features
if torch.cuda.is_available():
    my_densenet = my_densenet.cuda()
for p in my_densenet.parameters():
    p.requires_grad = False
train_features = []
train_features_labels = []
for im,a in train_dl:
    out = my_densenet(im)
    out = out.view(out.size(0),-1)
    train_features.extend(out.cpu().data)
    train_features_labels.extend(a)

test_features = []
test_features_labels = []
for im,a in test_dl:
    out = my_densenet(im)
    # 扁扁平化
    out = out.view(out.size(0),-1)
    test_features.extend(out.cpu().data)
    test_features_labels.extend(a)

print(test_features)