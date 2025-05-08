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

from softmax import loss_fn

#从分类的文件夹中创建dataset数据集
# torchvision.datasets.ImageFolder()
base_dir = r"./dataset/4weather"
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir,"train")
    test_dir = os.path.join(base_dir,"test")
    os.mkdir(train_dir)
    os.mkdir(test_dir)
species = ["cloudy", "rain", "shine", "sunrise"]

for train_or_test in ["train","test"]:
    for spec in species:
        if not os.path.exists(os.path.join(base_dir,train_or_test,spec)):
            os.mkdir(os.path.join(base_dir,train_or_test,spec))

image_dir = r"./dataset/dataset2"
for i, img in  enumerate(os.listdir(image_dir)):
    for spec in species:
        if spec in img:
            s = os.path.join(image_dir, img)
            if i%5==0:
                d = os.path.join(base_dir,"test",spec,img)
            else:
                d = os.path.join(base_dir, "train", spec, img)
            shutil.copy(s,d)

##读取图片
from torchvision import transforms
#统一图片大小
transform = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
train_dir = os.path.join(base_dir,"train")
test_dir = os.path.join(base_dir,"test")
train_ds = torchvision.datasets.ImageFolder(train_dir,transform=transform)
test_ds = torchvision.datasets.ImageFolder(test_dir,transform=transform)
print(train_ds.classes)
#创建dataloader
batch_size = 64
train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=batch_size)
imgs,labels = next(iter(train_dl))
print(imgs.shape) #torch.Size([64, 3, 96, 96]) 64:batch_size,3:channel
# im  = imgs[0].permute(1,2,0)
# im = im.numpy()
# im = (im+1)/2
# plt.imshow(im)
# plt.show()
id_to_class = dict((v,k) for k,v in train_ds.class_to_idx.items())
plt.figure(figsize=(12,8))
for i,(img,label) in enumerate(zip(imgs[:6],labels[:6])):
        img = (img.permute(1,2,0).numpy()+1)/2
        plt.subplot(2,3,i+1)
        plt.title(id_to_class.get(label.item()))
        plt.imshow(img)
plt.show()
#模型创建
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        #BN层，批处理层
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(64*10*10,1024)
        self.bn_f1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024,256)
        self.bn_f2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,4)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn3(x)
        # print(x.size())
        x = x.view(-1,x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = self.bn_f1(x)
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.bn_f2 (x)
        x = self.drop(x)
        x = self.fc3(x)
        return x

model = Net()
preds = model(imgs)
print(torch.argmax(preds,1))
if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##编写一个fit函数，输入模型，输入数据（train_dl，test_dl），对数据输入在模型上训练，并且返回loss和acc变化
def fit(epoch,model,trainloader,testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train() #训练模式
    for x,y in trainloader:
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
    epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc = fit(epoch,model,train_dl,test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

#处理过拟合：最好的办法：增加训练数据 dropout，随机丢弃一些神经元
# 为什么dropout可以解决过拟合
# 1 取平均的作用
# 2 减少神经元之间复杂的共适应关系
# 3 dropout类似于性别在生物进化中的角色