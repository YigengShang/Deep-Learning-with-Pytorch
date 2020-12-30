import torch
from torch import autograd


x = torch.tensor(2.,requires_grad=True)
a = torch.tensor(3.)

y = a * x**3

grads = autograd.grad(y,[x])
print('results',grads[0])