import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import  matplotlib.pyplot as plt
import torchvision
import torch.utils.data
from  torchvision import datasets,transforms


transformmation = transforms.Compose([
    transforms.ToTensor(),
])
train_ds = datasets.MNIST("dataset",train=True,transform=transformmation,download=True)
test_ds = datasets.MNIST("dataset",train=False,transform=transformmation,download=True)

train_dl = torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=256)

images,labels = next(iter(train_dl))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积
        self.conv1 = nn.Conv2d(1,6,kernel_size=5)
        #池化层
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(6,16,5)
        #全连接
        self.liner1 = nn.Linear(16*4*4,256)
        self.liner2 = nn.Linear(256,10)
    def forward(self,input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.size())
        #展平
        x = x.view(x.size(0),-1) # x = x.view(-1，16*4*4)
        x = F.relu(self.liner1(x))
        x = self.liner2(x)
        return  x

model = Model()
print(model)
model(images)

#创建损失函数
loss_fn = torch.nn.CrossEntropyLoss()


##编写一个fit函数，输入模型，输入数据（train_dl，test_dl），对数据输入在模型上训练，并且返回loss和acc变化
def fit(epoch,model,trainloader,testloader):
    correct = 0
    total = 0
    running_loss = 0
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

optim = torch.optim.Adam(model.parameters(),lr=0.0001)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#在GPU上训练
# 将模型转移搭到GPU
# 将每一个批次的训练数据转移到GPU
"""
model = Model()
model.to(device)
"""