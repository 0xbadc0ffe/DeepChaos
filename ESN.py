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
        self.activation = activation
        self.ker_size = 10

        if activation == "LeakyReLU":
            self.act = torch.nn.LeakyReLU()
        elif activation == "Tanh":
            self.act = torch.nn.Tanh()
        elif  activation == "ELU":
            self.act = torch.nn.ELU(alpha=1.0, inplace=False)
        elif activation == "ModTanh":
            self.mod = nn.Linear(W.shape[0], 1)
            self.act = torch.nn.Tanh()
        elif activation == "PrModTanh":
            self.mod = nn.Linear(W.shape[0], 1)
            self.act = torch.nn.Tanh()
        elif activation == "ConvModTanh":
            self.mod2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(W.shape[0],1))
            self.mod3 = nn.Linear(W.shape[0],1)
            self.act = torch.nn.Tanh()
        else:   
            # Default activation function
            self.act = torch.nn.Tanh()

        self.fco = nn.Linear(W.shape[0], output_size)

        #self.mod2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.ker_size)#max(3,W.shape[0]//20))
        #self.mod3 = nn.Linear((W.shape[0]-self.ker_size+1),1)

        #self.mod4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(W.shape[0],1))
        #self.mod5 = nn.Linear(W.shape[0],1)

    def forward(self, 
                u: torch.Tensor, 
                h_i: torch.Tensor
        ) -> torch.Tensor:

        #x = self.fci(u)
        xu = torch.einsum("ik, k -> i", self.Win, u)
        #x, h_o = self.resvoir(x, h_i)
        x = xu + torch.einsum("ij, j -> i", self.W, h_i)
        #h_o = torch.tanh(x)   # x(n+1) = tanh(Win*u(n) +  W*x(n))
        h_o = self.act(x)

        if self.activation == "ModTanh":
            h_o = (torch.tanh(self.mod(h_o))+1)*h_o  

        elif self.activation == "PrModTanh":
            pr = torch.tanh(torch.einsum("ij, j -> i", self.W, h_o) +  torch.einsum("ik, k -> i", self.Win, self.fco(h_o)))
            h_o = (torch.tanh(self.mod(pr))+1)*h_o

        elif self.activation == "ConvModTanh":
            h_conv = torch.einsum("ij, j -> i", self.W, h_o) 
            h_conv = torch.einsum("i, ij, k-> jk", h_o, self.W, h_conv)
            h_conv = self.mod2(torch.reshape(h_conv,(1,1,1000,1000)))[0,0,0,...]
            h_o = (torch.tanh(self.mod3(h_conv))+1)*h_o

        #h_conv = torch.einsum("ij, j -> i", self.W, h_i) 
        #h_conv = torch.einsum("i, ij, k-> jk", h_i, self.W, h_conv)
        
        #h_conv = self.mod2(torch.reshape(h_conv,(1,1,1000,1000)))[0,0,...]
        #h_o = (torch.tanh(self.mod3(torch.diag(h_conv)))+1)*h_o

        #h_conv = self.mod2(torch.reshape(h_conv,(1,1,1000,1000)))[0,0,...]
        #h_conv = F.max_pool2d(h_conv, kernel_size=self.ker_size)
        #h_o = (torch.tanh(self.mod3(h_conv))+1)*h_o

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
d = 6  # 0.1*H
Nt = 100
for_hor = 4      # This can be any integer, "n" for infinite horizon, "v" for variable
dym_sys = 3
epochs = 500
activations = ["LeakyReLU", "Tanh", "ELU", "ModTanh", "PrModTanh", "ConvModTanh"]
activation = activations[0]
sys_types = ["dis_rectilinear", "dis_sinusoidal"]
sys_type = sys_types[0]
opt_types = ["SGD", "Adam"]
opt_type = opt_types[1]
early_stop = None #0.0005*Nt # 0.05  # None to deactive early stopping
tikhonov = 0  # lambda value, 0 to deactive tikhonov
p_tikhonov = 2


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

In_acc = torch.rand([H], device=device)*2 - 1
h_0 = torch.rand([H], device=device)
Win = torch.zeros([H,dym_sys], device=device)
for i in range(In_acc.shape[0]):
    Win[i,torch.randint(high=dym_sys-1, size=[1], device=device)] = In_acc[i]

W = W.to(device)
model = ESN(Win, W, activation, dym_sys).to(device)


print("########## ESN\n")
print(f'Using device: {device}') 
print(f"Hidden dimension: {H}")
print(f"AVG connectivity: {cnt/H}")
print(f"Training horizon: {Nt}")
print(f"System type: {sys_type}")
print(f"System dimension: {dym_sys}")
print(f"Trained forecasting horizon: {for_hor}")
print(f"Epochs: {epochs}")
print(f"Reservoir activation function: {activation}")
print(f"Optimizer: {opt_type}")
print(f"Ealry Stop: {early_stop}")
print(f"Tikhonov: {tikhonov}   [ p={p_tikhonov} ]")
print(f'ESN number of parameters: {count_parameters(model)}\n')


########### Ground Truth model

if sys_type == "dis_rectilinear":

    A = torch.tensor(np.array([[1, 0, 0],[0, 1 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
    x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
    b = torch.tensor(np.array([0.1,0,0]), dtype=torch.float, device=device)

elif sys_type == "dis_sinusoidal":  

    A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
    x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
    b = torch.tensor(np.array([0,0,0]), dtype=torch.float, device=device)

else:
    # default: dis_sinusoidal
    A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
    x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
    b = torch.tensor(np.array([0,0,0]), dtype=torch.float, device=device)


########### TRAINING Phase

optimizer = get_model_optimizer(model, opt_type)

num_print = 10      # Number of Loss values printed
buff_print = ""
loss_plotter = []
weigths_norm_plt = []
stopped = False
for epoch in trange(epochs, desc="train epoch"):
    model.train()

    h_i = h_0
    x_i = torch.rand([3], dtype=torch.float, device=device)
    x_hat_i = x_i

    Ed = 0
    if for_hor == "v":
        for_hor_t = random.randint(a=1, b=Nt)
    else:
        for_hor_t = for_hor
    for i in range(1, Nt+1):
        if for_hor_t != "n" and i % for_hor_t == 0:
            x_hat_i = x_i
        #try:
        x_hat_i, h_i = model(x_hat_i,h_i)
        #except Exception as e:
            #print(e)
        x_i = lin_sys(x_i, A, b)
        Ed += (x_hat_i - x_i)**2

    # p-regularization 
    Ed += tikhonov*torch.norm(model.fco.weight, p=p_tikhonov, dim=1)
    Ed = torch.sum(Ed/Nt)/dym_sys

    # updating print buffer
    if epochs < num_print or epoch % (epochs//num_print) == 0:
        buff_print += f"\nLoss [epoch {epoch}]: {Ed}"

    # updating buffers for plots    
    loss_plotter.append(Ed.detach())
    weigths_norm_plt.append(torch.sum(torch.norm(model.fco.weight, p=p_tikhonov, dim=1)).detach())

    # early stopping
    if early_stop != None and Ed < early_stop:
        stopped = True
        break

    # backpropagation and optimization
    Ed.backward()
    optimizer.step()
    optimizer.zero_grad()


if epochs < num_print or epoch % (epochs//num_print) != 0:
    buff_print += f"\nLoss [epoch {epoch}]: {Ed}"
print(buff_print)
if stopped:
    print(f"\nEarly Stop:  {Ed} < {early_stop}   [epoch: {epoch}]\n")


########### TEST Phase

model.eval()
Nt_test = Nt*2  # Horizon in test phase
k = 0           # Component of the system to plot 
x_sys = [x_0[k].cpu().numpy()]
x_for_1 = [x_0[k].cpu().numpy()]
x_for_n = [x_0[k].cpu().numpy()]
x_for_t = [x_0[k].cpu().numpy()]
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

    x_sys.append(x_i[k].cpu())
    x_for_1.append(x1_hat_i[k].cpu())
    x_for_n.append(xn_hat_i[k].cpu())
    x_for_t.append(xt_hat_i[k].cpu())



########### PLOTS

# Loss
plt.plot(loss_plotter)
plt.plot(weigths_norm_plt)
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


# IDEAS

# early stop
