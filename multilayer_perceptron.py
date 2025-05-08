import  torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.预处理数据，将字符类型的数据转换为计算机能识别的类型
original_data = pd.read_csv("dataset/HR.csv")

print(original_data.groupby(["salary","part"]).size())
# 将字符类型，转换未01或者布尔值
print(pd.get_dummies(original_data["salary"])) #将分类变量转换为虚拟变量（dummy variables）或指示变量（indicator variables）。这种转换通常用于机器学习和统计分析中，因为许多模型只能处理数值型数据，而不能直接处理分类数据。
original_data = original_data.join(pd.get_dummies(original_data["salary"]))
del  original_data["salary"]
original_data = original_data.join(pd.get_dummies(original_data["part"]))
del  original_data["part"]
print(original_data.head())

Y_data = original_data["left"].values.reshape(-1,1)
Y = torch.from_numpy(Y_data).type(torch.float32)
X_data = original_data[[c for c in original_data.columns if c != "left"]].values
X = torch.from_numpy(X_data).type(torch.FloatTensor)

# 2.创建模型
from torch import nn
"""
自定义模型：
 nn.Module：继承这个类
 init：初始化所有的层
 forward：定义模型的运算过程/前向传播的过程
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(20,64)
        self.liner_2 = nn.Linear(64,64)
        self.liner_3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.liner_1(input)
        x = self.relu(x)
        x = self.liner_2(x)
        x = self.relu(x)
        x = self.liner_3(x)
        x = self.sigmoid(x)
        return x

model = Model()
#改写模型
import torch.nn.functional  as F
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(20,64)
        self.liner_2 = nn.Linear(64,64)
        self.liner_3 = nn.Linear(64, 1)

    def forward(self, input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        x = F.sigmoid(self.liner_3(x))
        return x


lr  = 0.001
def get_model():
    model3 = Model2()
    opt = torch.optim.Adam(model3.parameters(),lr=lr)
    return model3,opt

my_model,optim = get_model()

#定义损失函数
loss_fn = nn.BCELoss()
batch = 64
no_of_batches = len(original_data)//batch
epochs = 100

for epoch in range(epochs):
    for i in range(no_of_batches):
        start = i*batch
        end = start + batch
        x = X[start:end]
        y = Y[start,end]
        y_pred = my_model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print("epoch ",epoch,"loss: ",loss_fn(my_model(X),Y).data.item())

#使用dataset类进行重构
from  torch.utils.data import  TensorDataset
HRdataset = TensorDataset(X,Y)

model1,optim1 = get_model()
for epoch in range(epochs):
    for i in range(no_of_batches):
        x,y = HRdataset[i*batch:i*batch+batch]
        y_pred = my_model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print("epoch ",epoch,"loss: ",loss_fn(my_model(X),Y).data.item())

#使用dataloader类
from  torch.utils.data import DataLoader
HR_ds = TensorDataset(X,Y)
HT_dl = DataLoader(HR_ds,batch_size=batch,shuffle=True)

model2,optim2 = get_model()
for epoch in range(epochs):
    for x,y in HT_dl:
        y_pred = my_model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print("epoch ",epoch,"loss: ",loss_fn(my_model(X),Y).data.item())

# 过拟合与欠拟合
#过拟合：对于训练数据过度拟合，对未知数据预测很差
#欠拟合：都很差
import sklearn
from sklearn.model_selection import train_test_split
#将数据划分为测试数据和训练数据
train_x,test_x,train_y,test_y = train_test_split(X_data,Y_data)
train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.float32)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.float32)

train_ds = TensorDataset(train_x,train_y)
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True)

test_ds = TensorDataset(test_x,test_y)
test_dl = DataLoader(test_ds,batch_size=batch)
# 计算正确率

# (y_pred == la)
def accuracy(y_pred,y_true):
    y_pred = (y_pred > 0.5).type(torch.int32)
    acc = (y_pred == y_true).float().mean()
    return  acc

model2,optim2 = get_model()
for epoch in range(epochs):
    for x,y in train_dl:
        y_pred = my_model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        epoch_acc = accuracy(model2(train_x),train_y)
        epoch_loss = loss_fn(my_model(train_x),train_y).data

        epoch_test_acc = accuracy(model2(test_x), test_y)
        epoch_test_loss = loss_fn(my_model(test_x), test_y).data


        print("epoch ",epoch,"loss: ",epoch_loss.item(),

              "accuracy: ",round(epoch_acc.item(),3),
              "test_loss",epoch_test_loss.item(),
              "test_acc",epoch_test_acc.item())