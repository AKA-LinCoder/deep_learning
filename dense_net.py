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
import glob

from img_load import test_lables

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
