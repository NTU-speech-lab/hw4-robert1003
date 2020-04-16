# main.py
import os
import torch
import argparse
import sys
import random
import torch
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from _data import TwitterDataset
from _preprocess import Preprocess
from _model import LSTM_Net
from _utils import load_training_data, load_testing_data, evaluation, training, testing

output_path_prefix = './'
train_label_fpath = sys.argv[1]
train_unlabel_fpath = sys.argv[2]

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 處理好各個data的路徑
train_with_label = os.path.join(train_label_fpath)
train_no_label = os.path.join(train_unlabel_fpath)

w2v_path = os.path.join('embedding/w2v_all.model') # 處理word to vec model的路徑

# 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
sen_len = 30
fix_embedding = True # fix embedding during training
batch_size = 64
epoch = 10
lr = 0.0005
# model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
model_dir = output_path_prefix # model directory for checkpoint model

print("loading data ...") # 把'training_label.txt'跟'training_nolabel.txt'讀進來
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)

# 對input跟labels做預處理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 製作一個model的對象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=250, num_layers=2, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

# 把data分為training data跟validation data(將一部份training data拿去當作validation data)
X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]

# 把data做成dataset供dataloader取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
