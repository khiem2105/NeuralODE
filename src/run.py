from trainer import Trainer
from config import Config
from models import *
from dataset import load_MNIST
from utils import accuracy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, choices=["resnet", "odenet"], default="odenet")
parser.add_argument("--use_adjoint", type=eval, choices=[True, False], default=True)
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--data_aug", type=eval, choices=[True, False], default=True)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

config = Config(
    network=args.network,
    use_adjoint=args.use_adjoint,
    tol=args.tol,
    max_epochs=args.max_epochs,
    data_aug=args.data_aug,
    lr=args.lr,
    batch_size=args.batch_size
)

train_loader, test_loader = load_MNIST(
    batch_size=config.batch_size,
    data_aug=config.data_aug
)

trainer = Trainer(config)

if __name__ == "__main__":
    import wandb
    wandb.login()

    run_name = config.network + "_no_adjoint" if config.network == "odenet" and not config.use_adjoint\
               else config.network
    run = wandb.init(
        project="NeuralODE",
        name=run_name,
        config=config.__dict__
    )

    trainer.train(
        train_loader,
        test_loader,
        run
    )


