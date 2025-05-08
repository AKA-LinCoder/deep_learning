import torch


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