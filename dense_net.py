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
my_densenet = torchvision.models.densenet121(weights = True).features
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

class FeatureDataset(data.Dataset):
    def __init__(self,frat_list,label_list):
        self.frat_list = frat_list
        self.label_list = label_list
    def __getitem__(self, index):
        return self.frat_list[index],self.label_list[index]
    def __len__(self):
        return len(self.frat_list)

train_feat_ds = FeatureDataset(train_features,train_features_labels)
test_feat_ds = FeatureDataset(test_features,test_features_labels)
train_feat_dl = data.DataLoader(train_feat_ds,batch_size=batch_size,shuffle=True)
test_feat_dl = data.DataLoader(test_feat_ds,batch_size=batch_size)
class FCModel(torch.nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.lin = torch.nn.Linear(in_size,out_size)
    def forward(self,input):
        return self.lin(input,50176)
in_feat_size = train_features[0].shape[0]
net = FCModel(in_feat_size,200)

loss_fn = nn.CrossEntropyLoss()
optim = optim.Adam(net.parameters(),lr = 0.0001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fit(epoch,model,trainloader,testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train() #训练模式
    for x,y in trainloader:
        y = torch.tensor(y,dtype=torch.long)
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
           y_pred = torch.argmax(y_pred,dim=1)
           correct += (y_pred == y).sum().item()
           total += y.size(0)
           running_loss += loss.item()
    epoch_acc = correct /total
    epoch_loss = running_loss / len(trainloader.dataset)
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    model.eval() #预测模式，主要影响dropout层，BN层
    with torch.no_grad():
        for x,y in testloader:
            y = torch.tensor(y, dtype=torch.long)
            x,y = x.to(device),y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_acc = test_correct /test_total
    epoch_test_loss = test_running_loss / len(testloader.dataset)


    print("epoch ", epoch, "loss: ", round(epoch_loss,3),

      "accuracy: ", round(epoch_acc, 3),
      "test_loss", round(epoch_test_loss,3),
      "test_acc", round(epoch_test_acc,3))
    return epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc


epoches = 20

train_loss  = []
train_acc = []
test_loss = []
test_acc =  []
for epoch in range(epoches):
    epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc = fit(epoch,net,train_dl,test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)