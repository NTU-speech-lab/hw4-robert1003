# main.py
import os
import torch
import argparse
import sys
import random
import torch
import numpy as np
import pandas as pd
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from _data import TwitterDataset
from _preprocess import Preprocess
from _model import LSTM_Net
from _utils import load_training_data, load_testing_data, evaluation, training, testing

output_path_prefix = './'
test_fpath = sys.argv[1]
output_fpath = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

w2v_path = os.path.join('embedding/w2v_all.model') # 處理word to vec model的路徑

sen_len = 30
fix_embedding = True # fix embedding during training
batch_size = 64

print("loading data ...") 
test_x = load_testing_data(test_fpath)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

model = torch.load('./model.pth')
model = model.to(device)

outputs = testing(batch_size, test_loader, model, device)

print('')
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(output_fpath, index=False)
print("Finish Predicting")
