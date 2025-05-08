import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载（不推荐长期使用）
# print(torch.randn(3,5))
data = pd.read_csv("dataset/Income1.csv")
plt.scatter(data["Education"],data["Income"])
plt.show()
print(data)
X = torch.from_numpy(data["Education"].values.reshape(-1,1).astype(np.float32))
Y = torch.from_numpy(data["Income"].values.reshape(-1,1).astype(np.float32))

#分解写法
w = torch.randn(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
learning_rate = 0.0001
for epoch in range(5000):
    for x,y in zip(X,Y):
        y_pred = torch.matmul(x,w) + b
        loss = (y-y_pred).pow(2).mean()
        if not w.grad is None:
            w.grad.data.zero_()
        if not w.grad is None:
            w.grad.data.zero_()
        loss.backward()
        with torch.no_grad():
            w.data -= w.grad.data*learning_rate
            b.data -= b.grad.data * learning_rate
plt.scatter(data["Education"],data["Income"])
plt.plot(X.numpy(),(w*X+b).data.numpy(),c='r')
plt.show()






model = nn.Linear(1,1) #等价于创建了个线性函数
loss_fn = nn.MSELoss() #损失函数
opt = torch.optim.SGD(model.parameters(),lr=0.0001) #优化方法，梯度下降
for epoch in range(5000):
    for x,y in zip(X,Y):
        y_pred = model(x) #使用模型预测
        loss = loss_fn(y,y_pred) # 根据预测结果计算损失
        opt.zero_grad() #变量梯度清0
        loss.backward() #反向传播求解梯度
        opt.step() #模型优化参数
print(model.weight)
plt.scatter(data["Education"],data["Income"])
plt.plot(X.numpy(),model(X).data.numpy(),c='r')
plt.show()