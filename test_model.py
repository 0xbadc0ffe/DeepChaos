import torch
from torch.nn.modules import activation
import utils.data_save_handler as save_handler
from models.NNmodel import ESN, build_ESN
from models import Sys
from utils.nn_utils import set_reproducibility
import numpy as np
import utils.plotter as plt


model_dict = save_handler.load_model_dict()
cfgs = save_handler.load_cfgs()

sys_type = cfgs["sys_type"]
sys_name = cfgs["sys_name"]
Nt = cfgs["Nt"]
device = cfgs["device"]
activation = cfgs["activation"]
H = cfgs["H"]
dym_sys = cfgs["dym_sys"]
sigma_in =  cfgs["sigma_in"]
lambda_coeff =  cfgs["lambda_coeff"]
connectivity = cfgs["connectivity"]
d = round(connectivity)
for_hor =  cfgs["for_hor"]

if type(cfgs["reproducible"]) is int:
    set_reproducibility(seed=cfgs["reproducible"])


########### Ground Truth model


# Discrete
if sys_type == "discrete":

    if sys_name == "dis_rectilinear":

        A = torch.tensor(np.array([[1, 0, 0],[0, 1 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
        x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
        b = torch.tensor(np.array([0.1,0,0]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.linear_dis_sys(A,b)

    elif sys_name == "dis_sinusoidal":  

        #A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
        A = torch.tensor(np.array([[1, 0, 0],[0, 0 ,1],[ 0, -1, 0]]), dtype=torch.float, device=device)
        x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
        b = torch.tensor(np.array([0,0,0]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.linear_dis_sys(A,b)

    else:
        # default: discrete sinusoidal
        A = torch.tensor(np.array([[0, 1, 0],[-1, 0 ,0],[ 0, 0, 1]]), dtype=torch.float, device=device)
        x_0 = torch.tensor(np.array([1,1,-1.]), dtype=torch.float, device=device)
        b = torch.tensor(np.array([0,0,0]), dtype=torch.float, device=device)
        eps = 1
        df = Sys.linear_dis_sys(A,b)

# Continuous
elif sys_type == "continuous":

    if sys_name == "Lorenz":
        x_0 = torch.tensor([10, 20, 10], dtype=torch.float, device=device)
        #x_0 = torch.tensor([-2, -5, 25], dtype=torch.float, device=device)
        eps = 0.01
        df = Sys.Lorenz

    elif sys_name == "Elicoidal":
        x_0 = torch.tensor([5, 5, 5], dtype=torch.float, device=device)
        eps = 0.01
        df = Sys.Elicoidal

try:
    load_x0 = save_handler.load_initial()
    x_0 = load_x0
except:
    pass  

sys = Sys.Sys(x_0, df, eps, sys_type=sys_type)

model = build_ESN(H, d, lambda_coeff, dym_sys, sigma_in, activation, output_size=3, device=device)
model.load_state_dict(model_dict)
model.to(device)


try:
    W = save_handler.load_W()
    model.W = W.to(device)
except:
    pass

try:
    Win = save_handler.load_Win()
    model.Win = Win.to(device)
except:
    pass

try:
    h_0 = save_handler.load_hidden()
except:
    h_0 = model.rand_h0(device=device)


model.eval()
Nt_test = cfgs["Nt_test"]  # Horizon in test phase
k = cfgs["k"]           # Component of the system to plot 
threshold = cfgs["threshold"] # Error threshold for predicatbility horizon
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
print(f"Plotted component:              {k}")
print(f"Predictability threshold:       {threshold}")
for k,v in enumerate(error_plot_1for):
    if v>threshold:
        print(f"Predictability Horizon (1-for): {k} ({k*sys.eps})")
        break
for k,v in enumerate(error_plot_nfor):
    if v>threshold:
        print(f"Predictability Horizon (n-for): {k} ({k*sys.eps})")
        break
for k,v in enumerate(error_plot_tfor):
    if v>threshold:
        print(f"Predictability Horizon (t-for): {k} ({k*sys.eps})")
        break

plots = {}
plots["error_1-for"] = error_plot_1for
plots["error_n-for"] = error_plot_nfor
plots["error_t-for"] = error_plot_tfor
plots["ground_truth_sys"] = x_sys
plots["1-forecasting"] = x_for_1
plots["n-forecasting"] = x_for_n
plots["trained-forecasting"] = x_for_t

plt.plot_all(plots=plots,data=cfgs)