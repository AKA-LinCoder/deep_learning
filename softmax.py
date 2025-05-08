import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
## 多分类问题
data = pd.read_csv("dataset/iris.csv")
# 将最后一列的字符串类型数值转为0，1，2这样的int值
data["Species"] = pd.factorize(data["Species"])[0]
X = data.iloc[:,1:-1].values
Y = data["Species"].values
from sklearn.model_selection import  train_test_split
# 拆分数据为训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(X,Y)
train_x = torch.from_numpy(train_x).type(torch.float32)
test_x = torch.from_numpy(test_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.int64)
test_y = torch.from_numpy(test_y).type(torch.int64)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
batch = 8
train_ds = TensorDataset(train_x,train_y)
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True)

test_ds = TensorDataset(test_x,test_y)
test_dl = DataLoader(test_ds,batch_size=batch)

#创建模型
from torch import nn
import torch.nn.functional  as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(4,32)
        self.liner_2 = nn.Linear(32,32)
        self.liner_3 = nn.Linear(32,3)
    def forward(self,input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x) ##nn.CrossEntropyLoss损失函数
        return x

model = Model()
print(model)
#定义损失函数
loss_fn = nn.CrossEntropyLoss()

input_batch,label_batch = next(iter(train_dl))
y_pred = model(input_batch)
torch.argmax(y_pred,dim=1)
print(torch.argmax(y_pred,dim=1))

##创建训练函数
def accuracy(y_pred,y_true):
    y_pred = torch.argmax(y_pred,dim=1)
    acc = (y_pred == y_true).float().mean()
    return acc

epochs = 20
optim = torch.optim.Adam(model.parameters(),lr=0.001)
train_loss  = []
train_acc = []
test_loss = []
test_acc =  []

for epoch in range(epochs):
    for x,y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        epoch_acc = accuracy(model(train_x),train_y)
        epoch_loss = loss_fn(model(train_x),train_y).data

        epoch_test_acc = accuracy(model(test_x), test_y)
        epoch_test_loss = loss_fn(model(test_x), test_y).data


        print("epoch ",epoch,"loss: ",epoch_loss.item(),

              "accuracy: ",round(epoch_acc.item(),3),
              "test_loss",epoch_test_loss.item(),
              "test_acc",epoch_test_acc.item())

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

plt.plot(range(1,epochs+1),train_loss,label = "train_loss")
plt.plot(range(1,epochs+1),test_loss,label = "test_loss")
plt.legend()
plt.show()

plt.plot(range(1,epochs+1),train_acc,label = "train_acc")
plt.plot(range(1,epochs+1),test_acc,label = "test_acc")
plt.legend()
plt.show()

##编写一个fit函数，输入模型，输入数据（train_dl，test_dl），对数据输入在模型上训练，并且返回loss和acc变化
def fit(epoch,model,trainloader,testloader):
    correct = 0
    total = 0
    running_loss = 0
    for x,y in trainloader:
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

model1 = Model()
optim = torch.optim.Adam(model1.parameters(),lr=0.0001)
epochs = 30
train_loss  = []
train_acc = []
test_loss = []
test_acc =  []
for epoch in range(epochs):
    epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc = fit(epoch,model1,train_dl,test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

