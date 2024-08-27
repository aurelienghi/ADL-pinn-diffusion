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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 4) # 4 inputs, one layer of 4 nodes
        self.linear2 = nn.Linear(4, 2) # 2 latent dims, one layer of 4 nodes: mean
        self.linear3 = nn.Linear(4, 2) # 2 latent dims, one layer of 4 nodes: variance

        self.dist = torch.distributions.Normal(0, 1) # Normal distribution over the input parameters
        self.elbo = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x1 = func.relu(self.linear1(x)) # pass through ReLU activation and linear 1
        mu =  self.linear2(x1) # determine the mean
        sigma = torch.exp(self.linear3(x1)) # determine the standard deviation
        z = mu + sigma*self.dist.sample(mu.shape) # obtain the latent space z = mu + sigma * eps
        self.elbo = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() #loss: ELBO = p(theta) - D_KL (the KL divergence is what is tracked here)
        return z
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 6) # same structure as encoder, just not variational, and with y input
        self.linear2 = nn.Linear(6, 4)

    def forward(self, z):
        z1 = func.relu(self.linear1(z)) # pass through ReLU activation and linear 1
        x = torch.sigmoid(self.linear2(z1)) # pass through sigmoid activation and linear 2
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, y):
        z = self.encoder(torch.cat([x,y],dim=1))
        return self.decoder(torch.cat([z,y],dim=1))
    
def train(autoencoder, data, epochs=25):
    opt = torch.optim.Adam(autoencoder.parameters())
    autoencoder.train() # set to training mode
    for epoch in range(epochs):
        for x, y in data:
            opt.zero_grad() #reset gradients
            x_hat = autoencoder(x,y) #fetch output
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.elbo #loss
            loss.backward() # backwards step
            opt.step() # optimiser step
        print("End of epoch: " + str(epoch))
    #autoencoder.eval() # set to evaluation mode
    return autoencoder

def predict(autoencoder, y):
    #zdist = torch.distributions.Normal(0, 1)
    z = torch.as_tensor(np.float32(np.zeros((20,2)))) #zdist.sample([100,2]) #y.shape])
    x = autoencoder.decoder(torch.cat([z,torch.as_tensor(np.float32(y))],dim=1))
    x = x.to('cpu').detach().numpy()
    return x

class totalData(Dataset):

    def __init__(self, inputData, constraintData, dev="cpu"):

        self.dataLen = len(constraintData)
        
        self.input = inputData
        self.constraint = constraintData


    def __len__(self):

        return self.dataLen

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.input[idx])
        y = torch.FloatTensor(self.constraint[idx])

        return x, y

device = "cpu"
vae = Autoencoder()
x_values = np.genfromtxt('x_vals.csv', dtype= float, delimiter = ",")
y_values = np.genfromtxt('y_vals.csv', dtype= float, delimiter = ",")
combined_data = totalData(x_values, y_values[:,2:])
data = DataLoader(combined_data)

vae = train(vae, data)

torch.save(vae, 'complete_model.pth')

speed = np.array([238,  239.53846153846155,  245.3846153846154,  247.76923076923077,  249.53846153846155,  \
                250.76923076923077, 253,  251.3846153846154,  255.84615384615384,  256.7692307692308,  \
                257.2307692307692,  260.61538461538464,  263.61538461538464,  263.6923076923077,  \
                267.15384615384613,  267.84615384615387,  268.0769230769231,  271.0769230769231,  \
                272.46153846153845,  273.0769230769231])

alt = np.array([11.829787234042552,  11.627659574468085,  11.691489361702128,  11.696808510638299,  \
                11.696808510638299,  11.48936170212766,  11.48404255319149,  9.664893617021276,  \
                9.670212765957446,  9.670212765957446, 9.01063829787234,  9.579787234042554,  \
                9.553191489361701,  9.095744680851064,  9.553191489361701, 9.01063829787234,  \
                8.696808510638299,  9.430851063829786,  9.212765957446809, 8.48404255319149])

lbound = np.array([90, 7, 10, 0.2,150, 6.5])
ubound = np.array([150, 24, 34, 0.5, 273, 13.1])
diff = ubound-lbound
speed = np.divide(speed-lbound[4],diff[4])
alt = np.divide(alt-lbound[5],diff[5])

prediction = predict(vae, np.transpose(np.vstack((speed,alt))))
np.savetxt("pred2.csv",prediction,delimiter=",")
#prediction = predict(vae, np.tile(np.array([0,0]),(100,1)))
breakpoint()