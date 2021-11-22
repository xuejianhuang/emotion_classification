"""
训练模型
"""
import torch
import config
from word_sequence import WordSequence
from lstm_model import LSTM_Model
from dataset import get_dataloader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = config.device

loss_list = []


def train(epoch,model,optimizer,train_dataloader):
    model.train()
    bar = tqdm(train_dataloader, total=len(train_dataloader))  #配置进度条
    for idx, (input, target) in enumerate(bar):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        #loss_list.append(loss.cpu().data)
        #print(loss.cpu ().item())
        loss_list.append (loss.cpu().item())
        optimizer.step()
        bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, idx, np.mean(loss_list)))

def eval(model,test_dataloader):
    model.eval()
    loss_list = []
    eval_acc=0
    eval_total=0
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(config.device)
            target = target.to(config.device)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1) #返回最大值和对应的index
            pred = output_max[-1]  #最大值的index
            eval_acc+=pred.eq(target).cpu().float().sum().item()
            eval_total+=target.shape[0]
        acc=eval_acc/eval_total
        print("loss:{:.6f},acc:{}".format(np.mean(loss_list), acc))
    return acc

# 经过测试，该模型在测试集上的准确率达到了99%

def test(test_dataloader):
    model = LSTM_Model().to(device)
    model.load_state_dict(torch.load('model/model.pkl'))
    model.eval()
    loss_list = []
    test_acc=0
    test_total=0
    bar=tqdm(test_dataloader,total=len(test_dataloader))
    with torch.no_grad():
        for input, target in bar:
            input = input.to(config.device)
            target = target.to(config.device)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1) #返回最大值和对应的index
            pred = output_max[-1]  #最大值的index
            test_acc+=pred.eq(target).cpu().float().sum().item()
            test_total+=target.shape[0]
        print("test loss:{:.6f},acc:{}".format(np.mean(loss_list), test_acc/test_total))

if __name__ == '__main__':
    model = LSTM_Model().to(device)
    count_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters:,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = get_dataloader(model='train')
    val_dataloader = get_dataloader(model='val')
    best_acc=0
    early_stop_cnt=0
    for epoch in range(20):
        train(epoch,model,optimizer,train_dataloader)
        acc=eval(model,val_dataloader)
        if acc>best_acc:
            best_acc=acc
            torch.save(model.state_dict(), 'model/model.pkl')
            torch.save(optimizer.state_dict(), 'model/optimizer.pkl')
            print("save model,acc:{}".format(best_acc))
            early_stop_cnt=0
        else:
            early_stop_cnt+=1
        if early_stop_cnt>5:
            break
    plt.figure(figsize=(20, 8))
    plt.plot(range(len(loss_list)), loss_list)

    test_dataloader=get_dataloader(model='test')
    test(test_dataloader)

