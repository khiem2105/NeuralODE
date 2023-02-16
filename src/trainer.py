from config import Config
from models import *
from utils import accuracy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm


class Trainer():
    def __init__(
        self,
        config: Config
    ):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = config
        
        self.downsampling = DownSampling(in_channels=1, out_channels=64)
        self.feature_extractor = [ODEBlock(ODEFunc(n_channels=64), tol=self.config.tol)]\
                                 if self.config.network == "odenet"\
                                 else [ResBlock(in_channels=64, out_channels=64) for _ in range(6)]
        self.fc = [
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        ]

        self.net = nn.Sequential(
            self.downsampling,
            *self.feature_extractor,
            *self.fc
        )
        self.net = self.net.to(self.device)
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = SGD(self.net.parameters(), lr=self.config.lr)
        # self.lr_scheduler = MultiStepLR(
        #     self.optimizer,
        #     milestones=[50, 100, 150],
        #     gamma=0.1
        # )

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        logger
    ):
        it = 0
        for epoch in tqdm(range(self.config.max_epochs)):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                yhat = self.net(x)
                loss = self.loss(yhat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                logger.log({"train_loss": loss}, step=it)
                it += 1

            # self.lr_scheduler.step()

            # Eval
            train_acc = accuracy(self.net, train_loader)
            test_acc = accuracy(self.net, test_loader)

            logger.log({"train acc": train_acc}, step=epoch)
            logger.log({"test acc": test_acc}, step=epoch)


        