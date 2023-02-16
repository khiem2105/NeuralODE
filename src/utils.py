import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(
    model: nn.Module,
    loader: DataLoader
):
    acc = 0
    
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        yhat = model(x)
        pred = torch.max(torch.softmax(yhat, dim=-1), dim=-1)[1]
        acc_ = torch.mean((pred == y).float()).item()
        
        acc += acc_

    return acc

