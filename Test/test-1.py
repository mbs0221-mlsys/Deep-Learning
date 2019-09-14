import torch
import numpy as np

x = torch.randn(2, 2, requires_grad=True)
x = np.array([1., 2., 3.])
x = torch.from_numpy(x)
x.requires_grad_(True)

x = torch.tensor(1.0, requires_grad=True)
z = x ** 3
z.backward()
print(z.grad)

x = torch.tensor([0., 2., 8.], requires_grad=True)
y = torch.tensor([5., 1., 7.], requires_grad=True)
z = x * y
z.backward(torch.FloatTensor([1.0, 1.0, 1.0]))
print(z.grad)