import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.algorithms import SLE, FLE
import numpy as np

def build_ESN(H=200, d=3, lambda_coeff=0.4, dym_sys=3, sigma_in=0.15, activation="Tanh", output_size="3", device="cpu"):
    model = ESN()
    model.build_model(H, d, lambda_coeff, dym_sys, sigma_in, activation, output_size=output_size, device=device)
    return model


class ESN(nn.Module):
    def __init__(
        self, Win:torch.Tensor=None, W:torch.Tensor=None, activation:str = "Tanh", output_size: int = 3, h_0:torch.Tensor=None
    ) -> None:
        
        super().__init__()
        self.Win = Win
        self.W = W
        self.activation = activation
        self.h = h_0

        if W is None:
            return

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
        elif activation == "ConvModTanh1" or activation == "ConvModTanh2":
            self.mod2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(W.shape[0],1))
            self.mod3 = nn.Linear(W.shape[0],1)
            self.act = torch.nn.Tanh()
        else:   
            # Default activation function
            self.act = torch.nn.Tanh()

        self.fco = nn.Linear(W.shape[0], output_size)


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

        elif self.activation == "ConvModTanh1":
            h_conv = torch.einsum("ij, j -> i", self.W, h_o) 
            h_conv = torch.einsum("i, ij, k-> jk", h_o, self.W, h_conv)
            h_conv = self.mod2(torch.reshape(h_conv,(1,1,self.W.shape[0],self.W.shape[0])))[0,0,0,...]
            h_o = (torch.tanh(self.mod3(h_conv))+1)*h_o

        elif self.activation == "ConvModTanh2":
            h_conv= self.W*h_o
            h_conv = self.mod2(torch.reshape(h_conv,(1,1,self.W.shape[0],self.W.shape[0])))[0,0,0,...]
            h_o = (torch.tanh(self.mod3(h_conv))+1)*h_o


        x = self.fco(h_o)

        return x, h_o

    # like forward but handle the hidden state internnally
    def step(self, u: torch.Tensor )-> torch.Tensor:
        x, h_i = self.forward(u, self.h)
        self.h = h_i
        return x, h_i

    def init_reservoir(self, H=200, d=3, lambda_coeff=0.4, device="cpu"): 
        self.H = H         
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

        self.connectivity = cnt/H

        # Forcing largest eigenvalue norm to lambda to ensure ESP
        eig = SLE(W) #FLE(W)
        W = W*lambda_coeff/eig
        W.to(device)
        self.W = W

    # TODO: introduce probability for the i-th row of Win to depend on 1 output.
    def init_Win(self, H=None, dym_sys=3, sigma_in=0.15, device="cpu"):
        if H is None:
            H = self.H
        In_acc = (torch.rand([H], device=device)*2 - 1)*sigma_in
        if dym_sys == 1:
            Win = torch.ones([H,dym_sys], device=device)
        else:
            Win = torch.zeros([H,dym_sys], device=device)
            for i in range(In_acc.shape[0]):
                Win[i,torch.randint(high=dym_sys-1, size=[1], device=device)] = In_acc[i]
        self.Win = Win

    def set_activation(self, activation, H=None, device="cpu"): 

        if H is None:
            H = self.H

        self.activation = activation  
        if activation == "LeakyReLU":
            self.act = torch.nn.LeakyReLU()
        elif activation == "Tanh":
            self.act = torch.nn.Tanh()
        elif  activation == "ELU":
            self.act = torch.nn.ELU(alpha=1.0, inplace=False)
        elif activation == "ModTanh":
            self.mod = nn.Linear(H, 1, device=device)
            self.act = torch.nn.Tanh()
        elif activation == "PrModTanh":
            self.mod = nn.Linear(H, 1, device=device)
            self.act = torch.nn.Tanh()
        elif activation == "ConvModTanh1" or activation == "ConvModTanh2":
            self.mod2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(H,1), device=device)
            self.mod3 = nn.Linear(H,1, device=device)
            self.act = torch.nn.Tanh()
        else:   
            # Default activation function
            self.act = torch.nn.Tanh()

    def set_fc_output(self, output_size=3, H=None, device="cpu"):
        if H is None:
            H = self.H
        self.fco = nn.Linear(H, output_size, device=device)

    def build_model(self, H=200, d=3, lambda_coeff=0.4, dym_sys=3, sigma_in=0.15, activation="Tanh", output_size="3", device="cpu"):
        self.init_reservoir(H=H, d=d, lambda_coeff=lambda_coeff, device=device)
        self.init_Win(dym_sys=dym_sys, sigma_in=sigma_in, device=device)
        self.set_activation(activation, device=device)
        self.set_fc_output(output_size, device=device)

    def rand_h0(self, range=1,H=None, device="cpu"):
        if H is None:
            H = self.H
        self.h_0 = (torch.rand([H], device=device)*2-1)*range
        return self.h_0
