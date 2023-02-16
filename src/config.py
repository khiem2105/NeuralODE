from dataclasses import dataclass

@dataclass
class Config:
    network: str
    tol: float
    max_epochs: int
    data_aug: bool
    lr: float
    batch_size: int