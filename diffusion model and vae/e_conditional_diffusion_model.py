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
    def __init__(self,n):
        super().__init__()
        self.linear1 = nn.Linear(n, 4) # 4 inputs, one layer of 4 nodes
        self.linear2 = nn.Linear(4, 2) # 2 latent dims, one layer of 4 nodes: mean

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x1 = func.relu(self.linear1(x)) # pass through ReLU activation and linear 1
        z =  func.relu(self.linear2(x1))
        return z
    
class Decoder(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.linear1 = nn.Linear(2, 4) # same structure as encoder, just not variational, and with y input
        self.linear2 = nn.Linear(4, n)

    def forward(self, z):
        z1 = func.relu(self.linear1(z)) # pass through ReLU activation and linear 1
        x = torch.sigmoid(self.linear2(z1)) # pass through sigmoid activation and linear 2
        return x
    
class Autoencoder(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.encoder = Encoder(n)
        self.decoder = Decoder(n)

    def forward(self, x, t):
        t_bar = posn[t,:]
        z = self.encoder(torch.cat([x,t_bar],dim=1))
        return self.decoder(z), torch.cat([x,t_bar],dim=1)
    
def physicsLoss(in_val,out_val):
    gradient1 = grad(out_val[0][0], in_val)
    gradient2 = grad(out_val[0][1], in_val)
    loss = 1-(1.0*(gradient1[0][0][4]>=0)+1.0*(gradient1[0][0][5]>=0)+1.0*(gradient2[0][0][4]<=0)+1.0*(gradient2[0][0][5]<=0))/4
    return loss

def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input."""
    return torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs=torch.ones_like(outputs), 
        create_graph=True
    )

def train(autoencoder, conditional, data, epochs=40):
    opt = torch.optim.Adam(autoencoder.parameters())
    opt2 = torch.optim.Adam(conditional.parameters())
    autoencoder.train() # set to training mode
    conditional.train() # set to training mode
    for epoch in range(epochs):
        for x_0, y, c in data:
            t = torch.randint(0, T, (1,))
            opt.zero_grad() #reset gradients
            x_t, eps = fwd(x_0, t)
            eps_hat, x_bar = autoencoder(x_t,t) #fetch output
            y_hat = autoencoder.encoder(x_bar) # fetch latent space
            w = weight[t]
            loss = w*func.l1_loss(eps[0], eps_hat[0,:4]) + (1-w)*func.l1_loss(y, y_hat)
            loss.backward() # backwards step
            opt.step() # optimiser step

            opt2.zero_grad() #reset gradients
            x_0_c = torch.cat([x_0,c],dim=1)
            x_t_c, eps_c = fwd(x_0_c, t)
            eps_hat_c, x_bar_c = conditional(x_t_c,t) #fetch output
            x_bar_c.requires_grad_(requires_grad=True)
            y_hat_c = conditional.encoder(x_bar_c) # fetch latent space
            physics_loss = physicsLoss(x_bar_c,y_hat_c)
            x_bar_c.requires_grad_(False) #stop tracking gradients
            loss2 = 0.5*(w*func.l1_loss(eps_c[0], eps_hat_c[0,:6]) + (1-w)*func.l1_loss(y, y_hat_c)) + 0.5*physics_loss
            loss2.backward() # backwards step
            opt2.step() # optimiser step

        print("End of epoch: " + str(epoch))
    autoencoder.eval() # set to evaluation mode
    conditional.eval()
    return autoencoder, conditional

@torch.no_grad()
def sample_step(x_t,c,t,autoencoder,conditional):
    eps_pred, x_hat = autoencoder(x_t,torch.tensor([t]))
    eps_cond, x_cond = conditional(torch.cat([x_t,c.unsqueeze(0)],dim=1),torch.tensor([t]))
    eps_pred = (1+cW)*eps_cond[:,:8] - cW*eps_pred
    a = alpha[t]*torch.ones_like(x_t)
    a_bar = alpha_bar[t]*torch.ones_like(x_t)
    b = beta[t]*torch.ones_like(x_t)
    variance_t = variance[t]*torch.ones_like(x_t)
    #convert alpha bars to correct dims
    mean = 1/torch.sqrt(a)*(x_t-b/torch.sqrt(1-a_bar)*eps_pred[:,:4])
    
    if t == 0:
        return mean
    else:
        eps = torch.randn_like(x_t)
        return mean + torch.sqrt(variance_t) * eps 

@torch.no_grad()
def sampling(T,c,autoencoder,conditional): #start with x_t = x_T (normally distributed)
    xdist = torch.distributions.Normal(0, 1)
    x_t = xdist.sample([1,4]) #torch.cat([xdist.sample([1,5]),torch.tensor(np.array([posn[T-1,:]]))],dim=1)
    for t in range(0,T):
        x_t = sample_step(x_t,c,t,autoencoder,conditional)
    t_bar = posn[0,:]
    physics = autoencoder.encoder(torch.cat([x_t,torch.tensor(np.array([t_bar]))],dim=1))
    return x_t, physics

def optfn(x_norm,y_norm,c_norm):
    mse_value = np.zeros((5,2))
    for kk in range(5):
        test1 = int(np.ceil(kk*len(y_norm)/5))
        test2 = int(np.ceil((kk+1)*len(y_norm)/5))
        x_train = np.vstack((x_norm[0:test1,:],x_norm[test2:,:]))
        y_train = np.vstack((y_norm[0:test1,:],y_norm[test2:,:]))
        c_train = np.vstack((c_norm[0:test1,:],c_norm[test2:,:]))
        x_test = x_norm[test1:test2,:]
        y_test = y_norm[test1:test2,:]
        c_test = c_norm[test1:test2,:]

        #split training and testing data
        combined_data = totalData(x_train, y_train, c_train)
        traindata = DataLoader(combined_data)
        n = x_train.shape[1] + posn.shape[1]
        diff = Autoencoder(n)
        conditional = Autoencoder(n + c_train.shape[1])
        diff, conditional = train(diff, conditional, traindata)
        torch.save(diff, 'complete_model.pth')
        torch.save(conditional, 'conditional_model.pth')
        breakpoint()

        print(sampling(T,torch.tensor([0.5,0.5]),diff,conditional))
        print(sampling(T,torch.tensor([1,0]),diff,conditional))
        print(sampling(T,torch.tensor([0,1]),diff,conditional))
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

    return mse_value

global alpha_bar, alpha, beta, variance, posn, weight, T, cW

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

# conditional weight
cW = 0.5

class totalData(Dataset):

    def __init__(self, inputData, physicsData, constraintData, dev="cpu"):

        self.dataLen = len(physicsData)
        
        self.input = inputData
        self.physics = physicsData
        self.constraints = constraintData


    def __len__(self):

        return self.dataLen

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.input[idx])
        y = torch.FloatTensor(self.physics[idx])
        c = torch.FloatTensor(self.constraints[idx])

        return x, y, c

device = "cpu"

def load_data(filename):
    values = np.genfromtxt(filename, dtype= float, delimiter = ",")
    values =  values[1:,:]
    min_v = np.min(values,axis=0)
    range_v = np.max(values,axis=0)-min_v
    norm_v = np.divide((values-min_v),range_v)
    return norm_v

# Normalise
x_norm = load_data('x_vals.csv')
y_norm = load_data('y_vals.csv')

# HOW TO KNOW WHEN TO STOP TRAINING
breakpoint()
mse = optfn(x_norm,y_norm[:,:2],y_norm[:,2:]).mean()
print(mse)