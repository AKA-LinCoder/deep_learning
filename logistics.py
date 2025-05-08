import pandas as pd
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/credit-a.csv",header=None)
#预处理
X = data.iloc[:,:-1] #取所有行，取前15列，即最后一列除外
Y = data.iloc[:,-1].replace(-1,0) #取所有行，最后一列
X = torch.from_numpy(X.values).type(torch.float32)
Y = torch.from_numpy(Y.values.reshape(-1,1)).type(torch.float32)
from  torch import nn
model = nn.Sequential(
    nn.Linear(15,1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss() #2元交叉损失
opt = torch.optim.Adam(model.parameters(),lr=0.0001)
batches = 16
no_of_batch = 653 //16
epoches = 1000
for epoch in range(epoches):
    for i in range(no_of_batch):
        start = i *batches
        end = start+batches
        x = X[start: end]
        y = Y[start: end]
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
print("RESULT:")
print(((model(X).data.numpy() > 0.5).astype('int') == Y.numpy()).mean())

