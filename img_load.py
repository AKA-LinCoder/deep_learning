import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from softmax import label_batch


class MyDatSet(data.Dataset):
    def __init__(self,root):
        self.imgs_path = root
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path
    def __len__(self):
        return len(self.imgs_path)

import glob
all_imgs_path = glob.glob(r"/Users/lin/Desktop/inin/dataset/dataset2/*.jpg")
print(all_imgs_path)
weather_dataset = MyDatSet(all_imgs_path)
wh_dl = torch.utils.data.DataLoader(weather_dataset,batch_size=4)
print(next(iter(wh_dl)))
species = ["cloudy","rain","shine","sunrise"]
species_to_index = dict((c,i) for i,c in enumerate(species))
index_to_specise = dict((v,k) for k,v in species_to_index.items())

all_labels = []
for img in all_imgs_path:
    for i,c in enumerate(species):
        if c in img:
            all_labels.append(i)

transform = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor()
])

#划分训练集和测试集
index = np.random.permutation(len(all_imgs_path))
all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]
s = int(len(all_imgs_path)*0.8)
train_imgs = all_imgs_path[:s]
train_lables = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_lables = all_labels[s:]

class newDataSet(data.Dataset):
    def __init__(self,img_paths,labels,transform):
        self.imgs = img_paths
        self.labels = labels
        self.trainfroms = transform
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        data = self.trainfroms(pil_img)
        return data,label
    def __len__(self):
        return len(self.imgs)

weather_data = newDataSet(all_imgs_path,all_labels,transform)
weather_dl = data.DataLoader(
    weather_data,batch_size=16,shuffle=True
)
imgs_batch,label_batch = next(iter(weather_dl))

plt.figure(figsize=(12,8))
for i,(img,label) in enumerate(zip(imgs_batch[:6],label_batch[:6])):
    img = img.permute(1,2,0).numpy()
    plt.subplot(2,3,i+1)
    plt.title(index_to_specise.get(label.item()))
    plt.imshow(img)

plt.show()

train_ds = newDataSet(train_imgs,train_lables,transform)
test_ds = newDataSet(test_imgs,test_lables,transform)
train_dl = data.DataLoader(train_ds,batch_size=16,shuffle=True)
test_dl = data.DataLoader(test_ds,batch_size=16)
