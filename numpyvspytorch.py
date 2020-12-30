import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
    '\nnumpy',type(np_data),
    '\ntorch',type(torch_data),
    '\nt2a',type(tensor2array)
)


data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)
#abs
print("\nabs",torch.abs(tensor))


data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
#metric
print('\ntorchmetric',torch.mm(tensor,tensor))