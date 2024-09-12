import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


def np2torch(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).reshape(n_samples, -1)

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(9, 20) # 4 inputs, one layer of 6 nodes
        self.linear2 = nn.Linear(20, 30) # 4 inputs, one layer of 6 nodes
        self.linear3 = nn.Linear(30, 10) # 6 nodes, one layer of 4 nodes
        self.linear4 = nn.Linear(10, 2) # 2 latent dims, one layer of 4 nodes

    def forward(self, x):
        x1 = func.relu(self.linear1(x)) # pass through ReLU activation and linear 1
        x2 =  func.relu(self.linear2(x1))
        x3 =  func.relu(self.linear3(x2))
        z =  self.linear4(x3)
        return z

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7, 20) # 4 inputs, one layer of 6 nodes
        self.linear2 = nn.Linear(20, 30) # 4 inputs, one layer of 6 nodes
        self.linear3 = nn.Linear(30, 10) # 6 nodes, one layer of 4 nodes
        self.linear4 = nn.Linear(10, 2) # 2 latent dims, one layer of 4 nodes

    def forward(self, x):
        x1 = func.relu(self.linear1(x)) # pass through ReLU activation and linear 1
        x2 =  func.relu(self.linear2(x1))
        x3 =  func.relu(self.linear3(x2))
        z =  self.linear4(x3)
        return z
    
def predict(model, aux, X):
    model.eval()
    aux.eval()
    Xt = np2torch(X)
    out1 = model.forward(Xt)
    out2 = aux.forward(torch.cat((Xt,out1),dim=1))
    return np.hstack((out2.detach().numpy(),out1.detach().numpy()))

def trainfn(model, aux, X, y,physin_t, phys_pred_t, learn_rate = 0.002, physweight = 0.01):
    Xt = np2torch(X)
    yt = np2torch(y)
    ploss=nn.MSELoss()
    optimiser = optim.Adam(list(model.parameters()) + list(aux.parameters()), lr = learn_rate)

    model.train()
    aux.train()
    losses = [2e-05,1e05]
    nbatch = 10
    ep = 0
    while (abs(losses[-2]-losses[-1]) > 1e-07) and (ep < 500):
        for i in range(int(Xt.shape[0]/nbatch)):
            optimiser.zero_grad()
            outputs = model.forward(Xt[i*nbatch:(i+1)*nbatch,:])
            phys_outputs = model.forward(physin_t)
            outputs2 = aux.forward(torch.cat((Xt[i*nbatch:(i+1)*nbatch,:],outputs),dim=1))
            physics_inputs = torch.cat((physin_t,phys_outputs),dim=1).requires_grad_(True)
            result = aux.forward(physics_inputs)
            phys_outputs2_env = grad(result[:,0],physics_inputs)[0][:,:7]
            phys_outputs2_cost = grad(result[:,1],physics_inputs)[0][:,:7]
            loss = ploss(yt[i*nbatch:(i+1)*nbatch,:2], outputs) + ploss(yt[i*nbatch:(i+1)*nbatch,2:], outputs2)
            physlossval = physweight*ploss(phys_pred_t[:,0:7], phys_outputs2_env) + physweight*ploss(phys_pred_t[:,7:14], phys_outputs2_cost)
            loss = loss + physlossval
            #print(physlossval)
            loss.backward()
            optimiser.step()
        losses.append(loss.item())
        #if ep % int(epochs / 10) == 0:
            #print(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.4f}")
        ep += 1

    return losses[2:]

def make_pinn(x_norm,y_norm,phys_in,phys_pred,numbers,bb,lr):
    mse_cost = np.array([])
    mse_env = np.array([])
    hyperparams = np.array([0,0,0])

    for physweight in [1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.0]:
        net = MLP()
        aux = MLP2()
        losses = trainfn(net, aux, x_norm[numbers[:bb],:], y_norm[numbers[:bb],:],np2torch(phys_in), np2torch(phys_pred), lr, physweight)
        pred = predict(net, aux, x_norm[numbers[bb:],:])
        mse_env = np.append(mse_env,((pred[:,0]-y_norm[numbers[bb:],0])**2).mean())
        mse_cost = np.append(mse_cost,((pred[:,1]-y_norm[numbers[bb:],1])**2).mean())
        hyperparams = np.vstack((hyperparams,np.array([bb,lr,physweight])))
        del net, aux

        #plt.semilogy(losses)
        #plt.show()

    np.savetxt("MSEConditions.txt",hyperparams)
    np.savetxt("MSEenv.txt",mse_env)
    np.savetxt("MSEcost.txt",mse_cost)

    print("PINN training complete")