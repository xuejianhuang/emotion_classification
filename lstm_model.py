"""
定义模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from word_sequence import WordSequence
from dataset import ws


class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(ws),  # 词典的大小
                                      embedding_dim=config.embedding_dim)  # 256
        self.lstm = nn.LSTM(input_size=config.embedding_dim,  # 256
                            hidden_size=config.hidden_size,  # 128
                            num_layers=config.num_layers,  # 2
                            batch_first=True,
                            bidirectional=True,
                            dropout=config.dropout)  # 0.5
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size)  # [128*2,128]
        self.fc2 = nn.Linear(config.hidden_size, 2)  # [128,2]

    def forward(self, input):
        """
        :param input: [batch_size,max_len],其中max_len表示每个句子有多少单词
        :return:
        """
        x = self.embedding(input)  # [batch_size,max_len,embedding_dim]
        # 经过lstm层，x:[batch_size,max_len,2*hidden_size]
        # h_n,c_n:[2*num_layers,batch_size,hidden_size]
        out, (h_n, c_n) = self.lstm(x)

        # 获取两个方向最后一次的h，进行concat
        output_fw = h_n[-2, :, :]  # [batch_size,hidden_size]
        output_bw = h_n[-1, :, :]  # [batch_size,hidden_size]
        out_put = torch.cat([output_fw, output_bw], dim=-1)  # [batch_size,hidden_size*2]

        out_fc1 = F.relu(self.fc1(out_put))  # []
        out_put = self.fc2(out_fc1)
        return F.log_softmax(out_put, dim=-1)
