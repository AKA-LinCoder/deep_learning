# torchvision内置了常用的数据集和最常见的模型
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from  torchvision import datasets,transforms
transformmation = transforms.Compose([
    transforms.ToTensor(), #转为一个tensor，转换到0-1之间,会将channel放在第一个维度上
    # transforms.Normalize()
])
train_ds = datasets.MNIST('dataset',train=True,transform=transformmation,download=True)
test_ds = datasets.MNIST('dataset',train=False,transform=transformmation,download=True)
train_dl = torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)
test_dl =torch.utils.data.DataLoader(test_ds,batch_size=256)

imgs,labels = next(iter(test_dl))
print(imgs.shape)
#在pytorch里图片的表示形式：【batch,channel,height,width】
img = imgs[0]
img = img.numpy()
img = np.squeeze(img)
plt.imshow(img)

def imshow(img):
    nimg = img.numpy()
    nimg = np.squeeze(nimg)
    plt.imshow(nimg)

plt.figure(figsize=(10,1))
for i,img in enumerate(imgs[:10]):
    plt.subplot(1,10,i+1)
    imshow(img)

#创建模型
from torch import nn
import torch.nn.functional  as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner_1 = nn.Linear(28*28,120)
        self.liner_2 = nn.Linear(120,84)
        self.liner_3 = nn.Linear(84,10)
    def forward(self,input):
        #展平
        x = input.view(-1,28*28)
        x = F.relu(self.liner_1(x))
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x) ##nn.CrossEntropyLoss损失函数
        return x
model = Model()
#创建损失函数
loss_fn = torch.nn.CrossEntropyLoss()


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


data = torchvision.datasets.FashionMNIST("dataset",train=True,download=True)