import pickle
import torch
from word_sequence import WordSequence

# load 从数据文件中读取数据，并转换为python的数据结构
# ws = pickle.load(open("./model/ws.pkl", "rb"))
embedding_dim = 256
hidden_size = 128
num_layers = 2
dropout = 0.5
train_batch_size = 1000
test_batch_size = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
