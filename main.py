from re import S
from typing import Union, Optional,  Optional, Callable, Dict
import numpy as np
import torch
from tqdm import tqdm, trange
import os
import random
import utils.plotter as plt
import utils.data_save_handler as save_handler
from utils.algorithms import FLE, SLE, lin_sys
from utils.nn_utils import count_parameters, get_model_optimizer, set_reproducibility
from models import Sys
from models.NNmodel import ESN, build_ESN
import platform

if platform.system() == 'Windows':
    CLEAR_STR = "cls" 
else:
    CLEAR_STR = "clear"


os.system(CLEAR_STR)

#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        
##############################

########### ESN - Hyperparameters

reproducible = True

if reproducible:
    set_reproducibility(seed=42)
    reproducible = 42

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
H = 200
d = 3              # 0.02*H
Nt = 1000          # paper: 1000
for_hor = 1        # This can be any integer, "n" for infinite horizon, "v" for variable
epochs = 10000
activations = ["LeakyReLU", "Tanh", "ELU", "ModTanh", "PrModTanh", "ConvModTanh1", "ConvModTanh2"]
activation = activations[6]

sys_types = {
            "discrete":     ["dis_rectilinear", "dis_sinusoidal", "messy_dis", "armonic_boom_dis", "collatz"], 
            "continuous":   ["Lorenz", "Elicoidal"]
        }
sys_type = list(sys_types.keys())[1]  # "discrete" #"continuous" 
sys_name = sys_types[sys_type][1]

opt_types = ["SGD", "Adam"]
opt_type = opt_types[1]

early_stop = None  # 0.0005*Nt  # None to deactive early stopping
tikhonov = 0.0001  # lambda value, 0 to deactive tikhonov
p_tikhonov = 2
sigma_in = 0.15
lambda_coeff = 0.4  # spectral radius. must be < 1 to ensure the Echo State Property
save_training = False         # save training
pre_training = False          # pre training (Ridge regression)
pre_training_horizon = 20     # pre training horizon 
alpha = 0                     # tempered Physical loss
basin_r = 0 #0.01             # radius of the n-sphere around x_0 form which initial states are randomly initialized during training
washout = 25                  # Steps after which the predictability is counted (the Reservoir do not depend almost anymore on h_0)                    


########### Ground Truth model

# Discrete
if sys_type == "discrete":

    if sys_name == "dis_rectilinear":

        A = torch.tensor(np.array([[1, 0, 0],[0, 1 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
        x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
        b = torch.tensor(np.array([0.1,0,0]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.linear_dis_sys(A,b)
        dym_sys = 3

    elif sys_name == "dis_sinusoidal":  

        #A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
        A = torch.tensor(np.array([[1, 0, 0],[0, 0 ,1],[ 0, -1, 0]]), dtype=torch.float, device=device)
        x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
        b = torch.tensor(np.array([0,0,0]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.linear_dis_sys(A,b)
        dym_sys = 3
    
    elif sys_name == "messy_dis":
        dym_sys = 1
        x_0 = torch.tensor(np.array([1]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.messy_dis_sys

    elif sys_name == "armonic_boom_dis":
        dym_sys = 2
        x_0 = torch.tensor(np.array([1,1]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.armonic_boom_dis

    elif sys_name == "collatz":
        dym_sys = 1
        x_0 = torch.tensor([27], dtype=torch.float, device=device)
        eps = 1
        df = Sys.collatz
    
    else:
        # default: discrete sinusoidal
        A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
        x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
        b = torch.tensor(np.array([0,0,0]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.linear_dis_sys(A,b)
        dym_sys = 3

# Continuous
elif sys_type == "continuous":

    if sys_name == "Lorenz":
        x_0 = torch.tensor([10, 20, 10], dtype=torch.float, device=device)
        #x_0 = torch.tensor([-2, -5, 25], dtype=torch.float, device=device)
        eps = 0.01
        df = Sys.Lorenz
        dym_sys = 3

    elif sys_name == "Elicoidal":
        x_0 = torch.tensor([5, 5, 5], dtype=torch.float, device=device)
        eps = 0.01
        df = Sys.Elicoidal
        dym_sys = 3


sys = Sys.Sys(x_0, df, eps, sys_type=sys_type)



########### Model Building


model = build_ESN(H, d, lambda_coeff, dym_sys, sigma_in, activation, output_size=dym_sys, device=device)
h_0 = model.rand_h0(device=device)
# since we are going to use the same model for different forecasting, we will save the internal state h_i externally



print("########## ESN\n")
print(f"Reproducibility:                {reproducible}")
print(f'Using device:                   {device}') 
print(f"Hidden dimension:               {H}")
print(f"AVG connectivity:               {model.connectivity}")
print(f"Sigma_in:                       {sigma_in}")
print(f"Lambda:                         {lambda_coeff}")
print(f"Alpha:                          {alpha}")
print(f"Training horizon:               {Nt}")
print(f"Washout:                        {washout}")
print(f"System type:                    {sys_name}   [{sys_type}]")
print(f"System dimension:               {dym_sys}")
print(f"Trained forecasting horizon:    {for_hor}")
print(f"Epochs:                         {epochs}")
print(f"Reservoir activation function:  {activation}")
print(f"Optimizer:                      {opt_type}")
print(f"Ealry Stop:                     {early_stop}")
print(f"Tikhonov:                       {tikhonov}   [ p={p_tikhonov} ]")
print(f"Basin radius:                   {basin_r}")
print(f"Save Training:                  {save_training}")
print(f"Pre-Trainig:                    {pre_training}")
print(f"Pre-Training horizon:           {pre_training_horizon}")
print(f'ESN number of parameters:       {count_parameters(model)}\n')



########### TRAINING Phase


optimizer = get_model_optimizer(model, opt_type)

## Pre-Training
#  Ridge regression: X*R'*(R*R' + tikhonov*I)^-1
if pre_training:
    h_i = h_0
    x_i = x_0 #(torch.rand([dym_sys], dtype=torch.float, device=device)*2-1)*basin_r + x_0
    R = h_0.clone().detach().unsqueeze(1)
    X = x_i.clone().detach().unsqueeze(1)
    x_hat_i = x_i

    sys.restart(x_i)

    model.train()
    if for_hor == "v":
        for_hor_t = random.randint(a=1, b=Nt)
    else:
        for_hor_t = for_hor
    x_prev = x_i
    for i in range(1, pre_training_horizon):
        if for_hor_t != "n" and i % for_hor_t == 0:
            x_hat_i = x_i

        # model prediction
        x_hat_i, h_i = model(x_hat_i,h_i)
         
        # ground truth model step
        sys.step()
        x_i = sys.x
        x_prev = x_hat_i

        X = torch.cat([X, x_i.clone().detach().unsqueeze(1)],dim=1)
        R = torch.cat([R, h_i.clone().detach().unsqueeze(1)],dim=1)

    # Wout = X*R'*(R*R' + tikhonov*I)^-1

    Wout = torch.einsum("dn, nh -> dh", X, R.t())
    R_inv = torch.einsum("hn, nk -> hk", R, R.t())

    Wout = torch.einsum("dh, hk -> dk", Wout, torch.inverse(R_inv + tikhonov*torch.eye(H, device=device)))
    #Wout.requires_grad = True

    with torch.no_grad():
        model.fco.weight = torch.nn.Parameter(Wout)




## Training

num_print = 10      # Number of Loss values printed
buff_print = ""
loss_plotter = []
weigths_norm_plt = []
stopped = False
for epoch in trange(epochs, desc="train epoch"):
    model.train()

    h_i = h_0
    x_i = (torch.rand([dym_sys], dtype=torch.float, device=device)*2-1)*basin_r + x_0
    x_hat_i = x_i

    sys.restart(x_i)

    Ed = 0
    Ep = 0
    x_prev = x_hat_i
    if for_hor == "v":
        for_hor_t = random.randint(a=1, b=Nt)
    else:
        for_hor_t = for_hor
    for i in range(1, Nt+1):
        if for_hor_t != "n" and i % for_hor_t == 0:
            x_hat_i = x_i

        x_hat_i, h_i = model(x_hat_i,h_i)

        sys.step()
        x_i = sys.x

        Ep += ((x_hat_i - x_prev) - df(sys.t0+sys.clock*sys.eps, x_hat_i)*sys.eps)**2
        Ed += (x_hat_i - x_i)**2        
        x_prev = x_hat_i


    # p-regularization 
    Ed += tikhonov*torch.norm(model.fco.weight, p=p_tikhonov, dim=1)
    Ed = torch.sum(Ed/Nt)/dym_sys

    # Physical constraint
    Ep = torch.sum(Ep/Nt)/dym_sys
    Ed += alpha*Ep

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
threshold = 0.2 # Error threshold for predicatbility horizon
sys.restart(x_0)


x_i = x_0
xn_hat_i = x_0
xt_hat_i = x_0
h1_i = h_0
hn_i = h_0
ht_i = h_0

x_sys = x_0.cpu().unsqueeze(0)
x_for_1 = x_0.cpu().unsqueeze(0)
x_for_n = x_0.cpu().unsqueeze(0)
x_for_t = x_0.cpu().unsqueeze(0)


time_avg=0
error_plot_1for = torch.zeros([1])
error_plot_nfor = torch.zeros([1])
error_plot_tfor = torch.zeros([1])

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
    sys.step()
    x_i = sys.x

    time_avg += torch.norm(x_i)**2

    error_plot_1for=torch.cat([error_plot_1for,torch.norm(x_i-x1_hat_i).detach().unsqueeze(0).cpu()],dim=0)
    error_plot_nfor=torch.cat([error_plot_nfor,torch.norm(x_i-xn_hat_i).detach().unsqueeze(0).cpu()],dim=0)
    error_plot_tfor=torch.cat([error_plot_tfor,torch.norm(x_i-xt_hat_i).detach().unsqueeze(0).cpu()],dim=0)


    x_sys = torch.cat([x_sys, x_i.unsqueeze(0).cpu()],dim=0)
    x_for_1 = torch.cat([x_for_1, x1_hat_i.unsqueeze(0).detach().cpu()],dim=0)
    x_for_n = torch.cat([x_for_n, xn_hat_i.unsqueeze(0).detach().cpu()],dim=0)
    x_for_t = torch.cat([x_for_t, xt_hat_i.unsqueeze(0).detach().cpu()],dim=0)

time_avg = (time_avg/Nt_test)**0.5
error_plot_1for = error_plot_1for/time_avg
error_plot_nfor = error_plot_nfor/time_avg
error_plot_tfor = error_plot_tfor/time_avg

print("\n\n\n########## Testing Phase:\n")
print(f"Tested horizon:                 {Nt_test}")
print(f"Predictability threshold:       {threshold}")
leng = len(error_plot_1for)-1
for k,v in enumerate(error_plot_1for):
    if (k>=washout and v>threshold) or k==leng:
        print(f"Predictability Horizon (1-for): {k} ({k*sys.eps}) | washout: {washout}")
        break
leng = len(error_plot_nfor)-1
for k,v in enumerate(error_plot_nfor):
    if (k>=washout and v>threshold) or k==leng:
        print(f"Predictability Horizon (n-for): {k} ({k*sys.eps}) | washout: {washout}")
        break
leng = len(error_plot_tfor)-1
for k,v in enumerate(error_plot_tfor):
    if (k>=washout and v>threshold) or k==leng:
        print(f"Predictability Horizon (t-for): {k} ({k*sys.eps}) | washout: {washout}")
        break


########### SAVE

plots = {}
data = {}
plots["loss"] = loss_plotter
plots["weigths_norm"] = weigths_norm_plt
plots["error_1-for"] = error_plot_1for
plots["error_n-for"] = error_plot_nfor
plots["error_t-for"] = error_plot_tfor
plots["ground_truth_sys"] = x_sys
plots["1-forecasting"] = x_for_1
plots["n-forecasting"] = x_for_n
plots["trained-forecasting"] = x_for_t

if save_training:

    # Saving plots and configs

    data["reproducible"] = reproducible
    data["device"] = str(device)
    data["H"] = H
    data["connectivity"] = model.connectivity
    data["sigma_in"] = sigma_in
    data["lambda_coeff"] = lambda_coeff
    data["alpha"] = alpha
    data["Nt"] = Nt
    data["sys_name"] = sys_name
    data["sys_type"] = sys_type
    data["dym_sys"] = dym_sys
    data["for_hor"] = for_hor
    data["epochs"] = epochs
    data["activation"] = activation
    data["opt_type"] = opt_type
    data["early_stop"] = early_stop
    data["tikhonov"] = tikhonov
    data["p_tikhonov"] = p_tikhonov
    data["basin_r"] = basin_r
    data["washout"] = washout
    data["pre_training"] = pre_training
    data["pre_training_horizon"] = pre_training_horizon
    data["parameters count"] = count_parameters(model)
    data["Nt_test"] = Nt_test
    data["threshold"] = threshold 


    data = {
        "reproducible": 42,
        "device": "cpu",
        "H": 200,
        "connectivity": 2.985,
        "sigma_in": 0.15,
        "lambda_coeff": 0.4,
        "alpha": 0,
        "Nt": 200,
        "sys_name": "Elicoidal",
        "sys_type": "continuous",
        "dym_sys": 3,
        "for_hor": 1,
        "epochs": 10000,
        "activation": "ConvModTanh2",
        "opt_type": "Adam",
        "early_stop": None,
        "tikhonov": 0.0001,
        "p_tikhonov": 2,
        "basin_r": 0,
        "washout": 25,
        "pre_training": False,
        "pre_training_horizon": pre_training_horizon,
        "parameters count": 1005,
        "Nt_test": 400,
        "threshold": 0.2
    }

    save_handler.save_cfgs(data)

    # save_handler.save_hidden(h_0.detach().cpu())
    # save_handler.save_initial(x_0.detach().cpu())
    # save_handler.save_W(model.W.detach().cpu())
    # save_handler.save_Win(model.Win.detach().cpu())
    # save_handler.save(model, data, plots)


########### PLOTS

if not save_training:
    data = {}
    data["Nt"] = Nt
    data["for_hor"] = for_hor


plt.plot_all(components=dym_sys, plots=plots,data=data)

