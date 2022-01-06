import torch
import numpy as np

def Lorenz(t, x, sigma=10, rho=28, beta=8/3):
    return torch.tensor([sigma*(x[1]-x[0]), rho*x[0] -x[1] - x[0]*x[2], x[0]*x[1]-beta*x[2]], device=x.device)

def Elicoidal(t, x, a=0.6, b=-0.1):
    return torch.tensor([-a*x[1], a*x[0], b*x[2]], device=x.device)

def linear_dis_sys(A:torch.Tensor, b:torch.Tensor):
    y = lambda t,x : torch.einsum("ik, k -> i", A, x) + b
    return y

def messy_dis_sys(t, x):
    return x*torch.sin(x)**2 + 0.2*x

def armonic_boom_dis(t,x, delT=50, delX=1, delS=0.4):
    return torch.tensor([x[0]*(delS+np.sin(t/delT)), -x[0]*(delS + np.cos(t/delT))])*delX


class Sys():

    def __init__(self, x_0, f, eps, t0=0, steps=1, sys_type="continuous") -> None:
        self.x_0 = x_0          # initial state
        self.f = f              # transition matrix
        self.x = x_0            # state
        self.t0 = t0            # initial time
        self.clock = 0          # clock
        self.eps = eps          # epsilon (time step)
        self.steps = steps      # RK steps
        self.sys_type=sys_type  # system type (discrete/continuous)

    # System step, executes self.steps RK4
    def step(self):
        if self.sys_type == "continuous":
            for i in range(self.steps):
                self.RK4()
                self.clock +=1
        elif self.sys_type == "discrete":
            for i in range(self.steps):
                self.dis_step()
                self.clock +=1
        else:
            raise Exception("Unknwon System Type")

    def dis_step(self):
        t = self.clock*self.eps + self.t0
        self.x = self.f(t, self.x)

    # Runge-Kutta 4
    def RK4(self):
        eps = self.eps
        x = self.x
        f = self.f

        t = self.clock*eps + self.t0
        k1 = eps * f(t, x)
        k2 = eps * f(t + 0.5 * eps, x + 0.5 * k1)
        k3 = eps * f(t + 0.5 * eps, x + 0.5 * k2)
        k4 = eps * f(t + eps, x + k3)
        self.x = x + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

    # System evolves until t_end
    def step_t(self,t_end):
        t = self.clock*self.eps + self.t0
        if t > t_end:
            raise Exception("End time cannot be lower then self.t")
        n = int((t_end -t)/self.eps)
        if self.sys_type == "continuous":
            for i in range(n):
                self.RK4()
                self.clock +=1
        elif self.sys_type == "discrete":
            for i in range(n):
                self.dis_step()
                self.clock +=1
        else:
            raise Exception("Unknwon System Type")


    def restart(self, x0, t0=0):
        self.x0 = x0
        self.t0 = t0
        self.x = x0
        self.clock = 0

