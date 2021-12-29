from re import S
from typing import Union, Optional,  Optional, Callable, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import random


# reproducibility stuff
if True:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module
  
  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lin_sys(x, A, b):
    y = torch.einsum("ik, k -> i", A, x)
    y = y + b
    return y


class ESN(nn.Module):
    def __init__(
        self, Win:torch.Tensor, W:torch.Tensor, activation:str, output_size: int
    ) -> None:
        
        super().__init__()
        #self.fci = nn.Linear(input_size, in_to_hidden)
        self.Win = Win
        self.W = W
        if activation == "LeakyReLU":
            self.act = torch.nn.LeakyReLU()
        elif activation == "Tanh":
            self.act = torch.nn.Tanh()
        # Default activation function
        else:   
            self.act = torch.nn.Tanh()
        self.fco = nn.Linear(W.shape[0], output_size)


    def forward(self, 
                u: torch.Tensor, 
                h_i: torch.Tensor
        ) -> torch.Tensor:

        #x = self.fci(u)
        x = torch.einsum("ik, k -> i", self.Win, u)
        #x, h_o = self.resvoir(x, h_i)
        x = x + torch.einsum("ij, j -> i", self.W, h_i)
        #h_o = torch.tanh(x)   # x(n+1) = tanh(Win*u(n) +  W*x(n))
        h_o = self.act(x)
        x = self.fco(h_o)

        return x, h_o


def get_model_optimizer(model: torch.nn.Module, opt_type:str) -> torch.optim.Optimizer:
    """
    Encapsulate the creation of the model's optimizer, to ensure that we use the
    same optimizer everywhere

    :param model: the model that contains the parameter to optimize

    :returns: the model's optimizer
    """
    if opt_type == "Adam":
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif opt_type == "SGD":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=1e-5)
    else:
        # default
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


##############################

########### ESN - Hyperparameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # TODO: move all tensors and model to device
os.system("cls")
H = 300
d = 6
Nt = 100
for_hor = 1      # This can be any integer, "n" for infinite horizon, "v" for variable
dym_sys = 3
epochs = 5000
activations = ["LeakyReLU", "Tanh"]
activation = activations[1]
sys_types = ["rectilinear", "sinusoidal"]
sys_type = sys_types[1]
opt_types = ["SGD", "Adam"]
opt_type = opt_types[1]


########### Internal Model (Reservoir)

W = torch.rand([H,H])*2 - 1
ind = np.diag_indices(W.shape[0])
W[ind[0],ind[1]] = 0
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        if torch.rand(1) > d/(H-1):
            W[i,j] = 0
cnt = 0
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        if abs(W[i,j]) > 0:
            cnt += 1

In_acc = torch.rand([H])*2 - 1
h_0 = torch.rand([H])
Win = torch.zeros([H,dym_sys])
for i in range(In_acc.shape[0]):
    Win[i,torch.randint(high=dym_sys-1, size=[1])] = In_acc[i]

#h_0.to(device)
#Win.to(device)
#W.to(device)
model = ESN(Win, W, activation, dym_sys)
#model.to(device)

print("########## ESN\n")
print(f'Using device: {device}') 
print(f"Hidden dimension: {H}")
print(f"AVG connectivity: {cnt/H}")
print(f"Training horizon: {Nt}")
print(f"System type: {sys_type}")
print(f"System dimension: {dym_sys}")
print(f"Trained forecasting horizon: {for_hor}")
print(f"Epochs: {epochs}")
print(f"Reservoir activation function: {model.act}")
print(f"Optimizer: {opt_type}")
print(f'ESN number of parameters: {count_parameters(model)}\n')


########### Ground Truth model

if sys_type == "rectilinear":

    A = torch.tensor(np.array([[1, 0, 0],[0, 1 ,0],[ 0, 0, 1]]), dtype=torch.float)
    x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float)
    b = torch.tensor(np.array([0.1,0,0]), dtype=torch.float)

elif sys_type == "sinusoidal":  

    A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float)
    x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float)
    b = torch.tensor(np.array([0,0,0]), dtype=torch.float)

else:
    # default: sinusoidal
    A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float)
    x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float)
    b = torch.tensor(np.array([0,0,0]), dtype=torch.float)


########### TRAINING Phase

optimizer = get_model_optimizer(model, opt_type)

num_print = 10      # Number of Loss values printed
buff_print = ""
loss_plotter = []
for epoch in trange(epochs, desc="train epoch"):
    model.train()

    h_i = h_0
    x_i = torch.rand([3], dtype=torch.float)
    #x_i.to(device)
    x_hat_i = x_i

    Ed = 0
    if for_hor == "v":
        for_hor_t = random.randint(a=1, b=Nt)
    else:
        for_hor_t = for_hor
    for i in range(1, Nt+1):
        if for_hor_t != "n" and i % for_hor_t == 0:
            x_hat_i = x_i
        x_hat_i, h_i = model(x_hat_i,h_i)
        x_i = lin_sys(x_i, A, b)
        Ed += (x_hat_i - x_i)**2

    Ed = torch.sum(Ed/Nt)/dym_sys
    if epoch % (epochs//num_print) == 0:
        buff_print += f"\nLoss [epoch {epoch}]: {Ed}"
    loss_plotter.append(Ed)

    Ed.backward()
    optimizer.step()
    optimizer.zero_grad()


if epoch % (epochs//num_print) != 0:
    buff_print += f"\nLoss [epoch {epoch}]: {Ed}"
print(buff_print)



########### TEST Phase

model.eval()
Nt_test = 200  # Horizon in test phase
k = 0          # Component of the system to plot 
x_sys = [x_0[k].numpy()]
x_for_1 = [x_0[k].numpy()]
x_for_n = [x_0[k].numpy()]
x_for_t = [x_0[k].numpy()]
x_i = x_0
xn_hat_i = x_0
xt_hat_i = x_0
h1_i = h_0
hn_i = h_0
ht_i = h_0

if for_hor == "v":
    #for_hor_t = random.randint(a=1, b=Nt)
    for_hor_t = Nt//2
else:
    for_hor_t = for_hor

for i in range(1, Nt_test+1):

    # 1 step forecasting
    x1_hat_i, h1_i = model(x_i, h1_i)

    # n step forecasting
    xn_hat_i, hn_i = model(xn_hat_i, hn_i)

    # trained t step forecasting
    if for_hor_t != "n" and i % for_hor_t == 0:
        xt_hat_i = x_i
    xt_hat_i, ht_i = model(xt_hat_i, ht_i)

    # ground truth system
    x_i = lin_sys(x_i, A, b)

    x_sys.append(x_i[k])
    x_for_1.append(x1_hat_i[k])
    x_for_n.append(xn_hat_i[k])
    x_for_t.append(xt_hat_i[k])



########### PLOTS

# Loss
plt.plot(loss_plotter)
plt.title(f"Loss value")
plt.xlabel("t")
plt.ylabel(f"L(t)")
plt.figure(2)


plt.title(f"Forecasting")
plt.plot(x_sys, color="blue", label="ground truth")
plt.plot(x_for_1, color="red", label="1-forecasting")
plt.plot(x_for_n, color="green",label="n-forecasting")
#if for_hor != "n" and for_hor != 1:
#    plt.plot(x_for_t, color="violet",label=f"{for_hor}-forecasting (trained)")
plt.plot(x_for_t, color="violet",label=f"{for_hor}-forecasting (trained)")
plt.xlabel("t")
plt.ylabel(f"x{k}(t)")
plt.legend()
plt.scatter(Nt, x_sys[Nt], color="black")

plt.show()