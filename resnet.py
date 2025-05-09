import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import torchvision
import torch.utils.data

# from cnn_weather import train_dir, batch_size

base_dir = r"./dataset/4weather"

train_dir = os.path.join(base_dir,"train")
test_dir = os.path.join(base_dir,"test")

from torchvision import transforms
#统一图片大小
#使用VGG，不能太小
transform = transforms.Compose([
    transforms.Resize((192,192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])

#
test_transform = transforms.Compose([
    transforms.Resize((192,192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),


])
train_transform = transforms.Compose([
    transforms.Resize(224),
#增加随机性
    transforms.RandomCrop(192),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
])

train_ds = torchvision.datasets.ImageFolder(train_dir,transform=train_transform)
test_ds = torchvision.datasets.ImageFolder(test_dir,transform=test_transform)

batch_size = 32
train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=batch_size)
imgs,labels = next(iter(train_dl))
#第一次会下载
model = torchvision.models.vgg16(pretrained=True)

for p in model.features.parameters():
    p.requires_grad = False

print(model.classifier[-1])
model.classifier[-1].out_features = 4
print(model.classifier[-1])
optim = torch.optim.Adam(model.classifier.parameters(),lr=0.0001)


#学习速率衰减的办法

from  torch.optim import lr_scheduler
#1
# for p in optim.param_groups:
#     p["lr"] *=0.9
#2
exp_lr_scheduler = lr_scheduler.StepLR(optim,step_size=5,gamma=0.9)

lr_scheduler.MultiStepLR(optim,[20,50,80],gamma=0.9)





loss_fn = nn.CrossEntropyLoss()
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
            #执行衰减
    exp_lr_scheduler.step()
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


epoches = 10

train_loss  = []
train_acc = []
test_loss = []
test_acc =  []
for epoch in range(epoches):
    epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc = fit(epoch,model,train_dl,test_dl)

    #学习速率衰减
    if epoch%5 == 0:
        for p in optim.param_groups:
            p["lr"] *=0.9

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)