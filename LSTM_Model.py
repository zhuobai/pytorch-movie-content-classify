"""
定义模型
"""
import torch
import torch.nn as nn
import lib
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_dataloader
import os
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(lib.ws),embedding_dim=100)
        self.lstm = nn.LSTM(input_size=100,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
                            batch_first=True,bidirectional=True,dropout=lib.dropout)
        self.fc1 = nn.Linear(lib.hidden_size * 2,64)
        self.fc2 = nn.Linear(64,2)


    def forward(self,input):
        """

        :param input: 形状[batch_size,max_len],其中max_len表示每个句子有多少单词
        :return:
        """
        x = self.embedding(input)   # 输出形状:[batch_size,seq_len,embedding_dim]
        # 经过lstm层，x:[batch_size,max_len,2*hidden_size]
        # h_n,c_n:[2*num_layers,batch_size,hidden_size]
        x, (h_n,c_n) = self.lstm(x)

        # 获取两个方向最后一次的output,进行concat
        output_fw = h_n[-2,:,:] # 正向最后一次输出
        output_bw = h_n[-1,:,:] # 反向最后一次输出

        output = torch.cat([output_fw,output_bw],dim=-1)

        out_fc1 = self.fc1(output)
        out_relu = F.relu(out_fc1)
        out = self.fc2(out_relu)
        return F.log_softmax(out,dim=-1)

model = MyModel().to(lib.device)
optimizer = optim.Adam(model.parameters(),lr=0.01)

if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))


def train(epoch):
    data_loader = get_dataloader(True)
    for idx, (input,target) in enumerate(data_loader):
        # 梯度清零
        optimizer.zero_grad()
        # 看能不能用gpu
        input = input.to(lib.device)
        target = target.to(lib.device)

        output = model(input)
        loss = F.nll_loss(output,target)
        print(epoch,idx,loss.item())
        # 反向传播
        loss.backward()
        # 梯度更新
        optimizer.step()

        if idx % 100 == 0:
            """
            pytorch有两种模型保存方式：
            一、保存整个神经网络的的结构信息和模型参数信息，save的对象是网络net

            二、只保存神经网络的训练模型参数，save的对象是net.state_dict()
            """
            torch.save(model.state_dict(),"./model/model.pkl")
            torch.save(optimizer.state_dict(),"./model/optimizer.pkl")

def evol():
    loss_list = []
    acc_list = []
    test_loader = get_dataloader(False)
    for idx,(input,target) in enumerate(test_loader):
        input = input.to(lib.device)
        target = target.to(lib.device)

        # with torch.no_grad:
        output = model(input)
        cur_loss = F.nll_loss(output,target)
        loss_list.append((cur_loss.cpu().item()))
        # 计算准确率
        pred = output.max(dim=-1)[1]
        cur_acc = pred.eq(target).float().mean()
        acc_list.append(cur_acc.cpu().item())

    print("total loss,acc:",np.mean(loss_list),np.mean(acc_list))



if __name__ == '__main__':
    # for i in range(8):
    #     train(i)
    evol()

