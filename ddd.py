import torch

device = torch.device("cuda" if torch.cuba.is_available() else "cpu")
print("using device : ", device)

x = torch.tensor([1.0, 2.0, 3.0], device= device)         # shape: (3, )
w = torch.tensor([0.5, 0.2, -0.3], device= device, requires_grad= True)
b = torch.tensor(0.1, device= device, requires_grad= True)

y = x @ w + b                                             # shape: ()
print("result : ", y.item())