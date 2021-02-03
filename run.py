import torch
from torch import nn, optim,cuda
from models import AutoRec, MRMSELoss
from data import MovielensDataset
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
AutoRec: Autoencoders Meet Collaborative Filtering implementation with PyTorch
"""
# Setting
def check_positive(val):
    val = int(val)
    if val <=0:
        raise argparse.ArgumentError(f'{val} is invalid value. epochs should be positive integer')
    return val

parser = argparse.ArgumentParser(description='matrix factorization with pytorch')
parser.add_argument('--epochs', '-e', type=check_positive, default=30)
parser.add_argument('--batch', '-b', type=check_positive, default=32)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-2)

args = parser.parse_args()

path = 'data/movielens/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_users, n_items = (9724, 610)

# Dataset & Dataloader

train_dataset = MovielensDataset(path=os.path.join(path,'ratings.csv'), index='movieId', columns='userId', train=True)
test_dataset = MovielensDataset(path=os.path.join(path,'ratings.csv'), index='movieId', columns='userId', train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch,shuffle=True)

# Model & Criterion
model = AutoRec(n_users=n_users, n_items=n_items, n_factors=200).to(device)

# Criterion & Optimizer
criterion = MRMSELoss().to(device)
optimizer = optim.Adam(model.parameters(), weight_decay= args.wd, lr=args.lr)

# Train & Test
def train(epoch):
    process = []
    for idx, (data,) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred,data)
        loss.backward()
        optim.step()
        process.append(loss.item())
        if idx % 100 == 0:
            print (f"[+] Epoch {epoch} [{idx * args.batch} / {len(train_loader.dataset)}] - RMSE {sum(process) / len(process)}")
    return torch.Tensor(sum(process) / len(process)).to(device)
def test():
    process = []
    for idx, (data,) in enumerate(test_loader):
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred,data)
        loss.backward()
        optim.step()
        process.append(loss.item())
    print (f"[*] Test RMSE {sum(process) / len(process)} ")
    return torch.Tensor(sum(process) / len(process)).to(device)
    
# Run
if __name__=="__main__":
    train_rmse = torch.Tensor([]).to(device)
    test_rmse = torch.Tensor([]).to(device)
    for epoch in range(args.epochs):
        train_rmse = torch.cat((train_rmse, train(epoch)), dim=0)
        test_rmse = torch.cat((test_rmse, test()), dim=0)
    plt.plot(range(args.epochs),train_rmse, range(args.epochs),test_rmse)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.xticks(range(0,args.epochs,2))
    plt.show()