import torch

def gradient_example():

    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(3.0, requires_grad=True)
    z = x ** 3 + w*x
    z.backward()  # Computes the gradient
    print(x.grad.data)  # Prints '3' which is dz/dx