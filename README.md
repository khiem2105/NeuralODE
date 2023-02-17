**NeuralODE** for Dampled Pendulum and MNIST classification

**Repository structure**

- [Notebook for the Damped Pendulum problem](/src/Damped%20Pendulum/)
- [Classification problem on MNIST dataset](/src/run.py):
```--network```: "resnet" or "odenet"
```--use_adjoint```: whether to use adjoint method or not for NeuralODE
```--tol```: error tolerance for the ODE solver
```--max_epochs```: max epochs for training
```--data_aug``: whether to use data augmentation or not
```--lr```: learning rate
```--batch_size```: batch size

**Links**
- [Github](https://github.com/khiem2105/NeuralODE)
- [Wandb logs](https://wandb.ai/amal-project/NeuralODE)