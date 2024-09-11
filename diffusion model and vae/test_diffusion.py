import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

def fwd(x_0,t):
    eps = torch.randn_like(x_0)
    a_bar = alpha_bar[t]*torch.ones_like(x_0)
    x_t = torch.sqrt(a_bar)*x_0 + torch.sqrt(1-a_bar)*eps
    return x_t, eps

#sinusoidal positional encoding
def encoding(T):
    d = 4
    n = 10000
    P = torch.zeros((T, d))
    for k in range(T):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P

def np2torch(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).reshape(n_samples, -1)
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(11, 20)
        self.linear2 = nn.Linear(20,8)
        self.linear3 = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x1 = func.relu(self.linear1(x))
        x2 = func.relu(self.linear2(x1))
        z =  self.linear3(x2)
        return z
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 20)
        self.linear3 = nn.Linear(20,7)

    def forward(self, z):
        z1 = func.relu(self.linear1(z))
        z2 = func.relu(self.linear2(z1))
        x = self.linear3(z2)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, t, n):
        t_bar = posn[t,:]
        z = self.encoder(torch.cat([x,torch.tile(t_bar,(n,1))],dim=1))
        return self.decoder(z)
    
def predict(model, aux, X):
    model.eval()
    aux.eval()
    Xt = np2torch(X)
    out1 = model.forward(Xt)
    out2 = aux.forward(torch.cat((Xt,out1),dim=1))
    return np.hstack((out2.detach().numpy(),out1.detach().numpy()))

def trainfn(auto, X, learn_rate = 0.002):
    Xt = np2torch(X)
    ploss=nn.MSELoss()
    optimiser = optim.Adam(list(auto.parameters()), lr = learn_rate) #list(model.parameters()) + list(aux.parameters())

    auto.train()
    losses = [2e-05,1e05]
    nbatch = 10
    ep = 0
    epochs = 50
    while (abs(losses[-2]-losses[-1]) > 1e-09) and (ep < epochs):
        for i in range(int(Xt.shape[0]/nbatch)):
            optimiser.zero_grad()

            t = torch.randint(0, T, (1,))

            x_0 = Xt[i*nbatch:(i+1)*nbatch,:]

            x_t, eps = fwd(x_0, t)

            eps_hat = auto(x_t,t,nbatch)

            loss = ploss(eps, eps_hat)

            loss.backward()
            optimiser.step()
        losses.append(loss.item())
        if ep % int(epochs / 10) == 0:
            print("Epoch", ep, "| loss:", losses[-1])
        ep += 1

    return losses[2:]

@torch.no_grad()
def sample_step(x_t,t,autoencoder):
    eps_pred = autoencoder(x_t,torch.tensor([t]))[0]
    a = alpha[t]*torch.ones_like(x_t)
    a_bar = alpha_bar[t]*torch.ones_like(x_t)
    b = beta[t]*torch.ones_like(x_t)
    variance_t = variance[t]*torch.ones_like(x_t)
    #convert alpha bars to correct dims
    mean = 1/torch.sqrt(a)*(x_t-b/torch.sqrt(1-a_bar)*eps_pred[:,:5])
    
    if t == 0:
        return mean
    else:
        eps = torch.randn_like(x_t)
        return mean + torch.sqrt(variance_t) * eps 

@torch.no_grad()
def sampling(T,autoencoder): #start with x_t = x_T (normally distributed)
    xdist = torch.distributions.Normal(0, 1)
    x_t = xdist.sample([1,7])
    for t in range(0,T):
        x_t = sample_step(x_t,t,autoencoder)
    return x_t

global alpha_bar, alpha, beta, variance, posn, weight, T

# Number of timesteps
T = 80

# Linear beta schedule
b1 = 1e-04
bT = 5e-01
beta = torch.linspace(b1, bT, T)

weight = torch.linspace(0.1,0.9,T)

# alpha = 1-beta
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, axis = 0)

plt.plot(torch.sqrt(alpha_bar), label = "sqrt alpha prod")
plt.plot(torch.sqrt(1- alpha_bar), label = "1 - sqrt alpha prod")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.show()
breakpoint()

posn = encoding(T)

# variance
variance = beta * (1. - torch.cat((torch.tensor([1.]),alpha_bar[:-1]))) / (1. - alpha_bar)

# Make training data
# Normalise

x_values = np.genfromtxt('CLenv_popn.csv', dtype= float, delimiter = ",")

x_mean = np.mean(x_values,axis=0)
x_std = np.std(x_values,axis=0)
x_norm = np.divide((x_values-x_mean),x_std)
#[iaspect,ispan,itaper,iareaH,isweep,ispeed,ialt]

auto = Autoencoder()
lr = 0.002
losses = trainfn(auto, x_norm, lr)
breakpoint()