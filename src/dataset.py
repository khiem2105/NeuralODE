from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomCrop, ToTensor
from torch.utils.data import DataLoader

def load_MNIST(
    batch_size: int,
    data_aug: bool,
    num_workers: int=8
):

    if data_aug:
        transform_train = Compose([
            RandomCrop(size=28, padding=4),
            ToTensor()
        ])
    else:
        transform_train = ToTensor()

    transform_test = ToTensor()

    train_dataset = MNIST(
        root="../dataset",
        train=True,
        transform=transform_train
    )
    test_dataset = MNIST(
        root="../dataset",
        train=False,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )

    return train_loader, test_loader