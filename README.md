**NeuralODE** for Dampled Pendulum and MNIST classification

**Repository structure**

- [Notebook for the Damped Pendulum problem](/src/Damped%20Pendulum/)
- [Classification problem on MNIST dataset](/src/main.py):
```--network```: "resnet" or "odenet"
```--use_adjoint```: whether to use adjoint method or not for NeuralODE
```--tol```: error tolerance for the ODE solver
```--max_epochs```: max epochs for training
```--data_aug``: whether to use data augmentation or not
```--lr```: learning rate
```--batch_size```: batch size