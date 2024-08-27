import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils
import torch.distributions
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#import matplotlib as mpl
import matplotlib.pyplot as plt

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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(9, 4) # 4 inputs, one layer of 4 nodes
        self.linear2 = nn.Linear(4, 2) # 2 latent dims, one layer of 4 nodes: mean

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x1 = func.relu(self.linear1(x)) # pass through ReLU activation and linear 1
        z =  func.relu(self.linear2(x1))
        return z
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4) # same structure as encoder, just not variational, and with y input
        self.linear2 = nn.Linear(4, 9)

    def forward(self, z):
        z1 = func.relu(self.linear1(z)) # pass through ReLU activation and linear 1
        x = torch.sigmoid(self.linear2(z1)) # pass through sigmoid activation and linear 2
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, t):
        t_bar = posn[t,:]
        z = self.encoder(torch.cat([x,t_bar],dim=1))
        return self.decoder(z), torch.cat([x,t_bar],dim=1)
    
def train(autoencoder, data, epochs=25):
    opt = torch.optim.Adam(autoencoder.parameters())
    autoencoder.train() # set to training mode
    for epoch in range(epochs):
        for x_0, y in data:
            t = torch.randint(0, T, (1,))
            opt.zero_grad() #reset gradients
            x_t, eps = fwd(x_0, t)
            eps_hat, x_bar = autoencoder(x_t,t) #fetch output
            y_hat = autoencoder.encoder(x_bar) # fetch latent space
            w = weight[t]
            loss = w*func.l1_loss(eps[0], eps_hat[0,:5]) + (1-w)*func.l1_loss(y, y_hat)
            loss.backward() # backwards step
            opt.step() # optimiser step
        print("End of epoch: " + str(epoch))
    autoencoder.eval() # set to evaluation mode
    return autoencoder

@torch.no_grad()
def sample_step(x_t,t,autoencoder):
    eps_pred, x_hat = autoencoder(x_t,torch.tensor([t]))
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
    breakpoint()
    x_t = xdist.sample([1,5]) #torch.cat([xdist.sample([1,5]),torch.tensor(np.array([posn[T-1,:]]))],dim=1)
    for t in range(0,T):
        x_t = sample_step(x_t,t,autoencoder)
    t_bar = posn[0,:]
    physics = autoencoder.encoder(torch.cat([x_t,torch.tensor(np.array([t_bar]))],dim=1))
    return x_t, physics

def optfn(x_norm,y_norm):
    mse_value = np.zeros((5,2))
    for kk in range(5):
        test1 = int(np.ceil(kk*len(y_norm)/5))
        test2 = int(np.ceil((kk+1)*len(y_norm)/5))
        x_train = np.vstack((x_norm[0:test1,:],x_norm[test2:,:]))
        y_train = np.vstack((y_norm[0:test1,:],y_norm[test2:,:]))
        x_test = x_norm[test1:test2,:]
        y_test = y_norm[test1:test2,:]

        #split training and testing data
        combined_data = totalData(x_train, y_train)
        traindata = DataLoader(combined_data)
        diff = Autoencoder()
        diff = train(diff, traindata)
        torch.save(diff, 'complete_model.pth')

        breakpoint()

        # Test the data
        test_pred = torch.zeros((len(y_test),2))
        t_bar = torch.tensor(np.array([[0.,1.,0.,1.]]),dtype=torch.float32)
        with torch.no_grad():
            for ii in range(len(y_test)):
                test_pred[ii,:] = diff.encoder(torch.cat([torch.tensor(np.array([x_test[ii,:]]),dtype=torch.float32),t_bar],dim=1)) # need to add extra dim ugh
        #NEED TO USE SOME OTHER WAY, AS THIS JUST PREDICT AFTER UNDONE
        # MSE
        mse_value[kk,:] = ((y_test - test_pred.detach().numpy())**2).mean(axis=0)

        print(sampling(T,diff))
        breakpoint()

    return mse_value


global alpha_bar, alpha, beta, variance, posn, weight, T

# Number of timesteps
T = 20

# Linear beta schedule
b1 = 1e-04
bT = 2e-02
beta = torch.linspace(b1, bT, T)

weight = torch.linspace(0.1,0.9,T)

# alpha = 1-beta
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, axis = 0)

posn = encoding(T)

# variance
variance = beta * (1. - torch.cat((torch.tensor([1.]),alpha_bar[:-1]))) / (1. - alpha_bar)

class totalData(Dataset):

    def __init__(self, inputData, physicsData, dev="cpu"):

        self.dataLen = len(physicsData)
        
        self.input = inputData
        self.physics = physicsData


    def __len__(self):

        return self.dataLen

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.input[idx])
        y = torch.FloatTensor(self.physics[idx])

        return x, y

device = "cpu"

# Normalise
x_values = np.genfromtxt('aero_vals.csv', dtype= float, delimiter = ",")
y_values = np.genfromtxt('aero_train.csv', dtype= float, delimiter = ",")
x_values =  x_values[1:,:]
y_values =  y_values[1:,:]
#min_x = np.min(x_values,axis=0)
min_y = np.min(y_values,axis=0)
#range_x = np.max(x_values,axis=0)-min_x
range_y = np.max(y_values,axis=0)-min_y

x_norm = np.divide((x_values-np.mean(x_values,axis=0)),np.std(x_values,axis=0))
y_norm = np.divide((y_values-min_y),range_y)

# HOW TO KNOW WHEN TO STOP TRAINING
breakpoint()
mse = optfn(x_norm,y_norm).mean()
print(mse)