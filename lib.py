import pickle
import torch

# load 从数据文件中读取数据，并转换为python的数据结构
ws = pickle.load(open(".\model\ws.pkl","rb"))
hidden_size = 64
num_layers = 2
dropout = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")