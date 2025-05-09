import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
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
class newDataSet(data.Dataset):
    def __init__(self,img_paths,labels):
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