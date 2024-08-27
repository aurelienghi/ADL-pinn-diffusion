import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from phys_model_jax import physics_grads_nocl, physics_grads, physics_pred


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

def predict2(aux, X,y):
    aux.eval()
    out2 = aux.forward(torch.cat((np2torch(X),np2torch(y)),dim=1))
    return np.hstack((out2.detach().numpy(),y))

def trainfn2(model, aux, X, y,physin_t, phys_outputs, phys_pred_t, epochs = 1000, learn_rate = 0.002, physweight = 0.01):
    Xt = np2torch(X)
    yt = np2torch(y)
    ploss=nn.MSELoss()
    optimiser = optim.Adam(aux.parameters(), lr = learn_rate)

    model.eval()
    aux.train()
    losses2 = []
    nbatch = 30
    #loss = 0.
    for ep in range(epochs):
        for i in range(int(Xt.shape[0]/nbatch)):
            optimiser.zero_grad()
            outputs = model.forward(Xt[i*nbatch:(i+1)*nbatch,:]).detach()
            outputs2 = aux.forward(torch.cat((Xt[i*nbatch:(i+1)*nbatch,:],outputs),dim=1))
            physics_inputs = torch.cat((physin_t,phys_outputs),dim=1).requires_grad_(True)
            result = aux.forward(physics_inputs)
            phys_outputs2_env = grad(result[:,0],physics_inputs)[0][:,5:7] #grad(aux.forward(physics_inputs)[:,0],physics_inputs)[0][:,5:]
            phys_outputs2_cost = grad(result[:,1],physics_inputs)[0][:,5:7]
            loss = ploss(yt[i*nbatch:(i+1)*nbatch,:2], outputs2)
            physlossval = physweight*ploss(phys_pred_t[:,0:2], phys_outputs2_env) + physweight*ploss(phys_pred_t[:,2:4], phys_outputs2_cost)
            loss = loss + physlossval
            #print(physlossval)
            loss.backward()
            optimiser.step()
        losses2.append(loss.item())
        if ep % int(epochs / 10) == 0:
            print(f"Epoch {ep}/{epochs}, loss: {losses2[-1]:.4f}")

    return losses2

def trainfn1(model, X, y, epochs = 100, learn_rate = 0.002):
    Xt = np2torch(X)
    yt = np2torch(y)
    closs=nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = learn_rate)
    model.train()
    losses1 = []
    nbatch = 30

    for ep in range(epochs):
        for i in range(int(Xt.shape[0]/nbatch)):
            optimiser.zero_grad()
            outputs = model.forward(Xt[i*nbatch:(i+1)*nbatch,:])
            loss = closs(yt[i*nbatch:(i+1)*nbatch,2:], outputs)
            loss.backward()
            optimiser.step()
        losses1.append(loss.item())
        if ep % int(epochs / 10) == 0:
            print(f"Epoch {ep}/{epochs}, loss: {losses1[-1]:.5f}")

    return losses1

def trainfn(model, aux, X, y,physin_t, phys_pred_t, epochs = 1000, learn_rate = 0.002, physweight = 0.01):
    Xt = np2torch(X)
    yt = np2torch(y)
    ploss=nn.MSELoss()
    optimiser = optim.Adam(list(model.parameters()) + list(aux.parameters()), lr = learn_rate)

    model.train()
    aux.train()
    losses = []
    nbatch = 30

    for ep in range(epochs):
        for i in range(int(Xt.shape[0]/nbatch)):
            optimiser.zero_grad()
            outputs = model.forward(Xt[i*nbatch:(i+1)*nbatch,:])
            phys_outputs = model.forward(physin_t)
            outputs2 = aux.forward(torch.cat((Xt[i*nbatch:(i+1)*nbatch,:],outputs),dim=1))
            physics_inputs = torch.cat((physin_t,phys_outputs),dim=1).requires_grad_(True)
            result = aux.forward(physics_inputs)
            phys_outputs2_env = grad(result[:,0],physics_inputs)[0][:,:7]
            phys_outputs2_cost = grad(result[:,1],physics_inputs)[0][:,:7]
            loss = ploss(yt[i*nbatch:(i+1)*nbatch,:2], outputs2) + ploss(yt[i*nbatch:(i+1)*nbatch,2:], outputs)
            physlossval = physweight*ploss(phys_pred_t[:,0:7], phys_outputs2_env) + physweight*ploss(phys_pred_t[:,7:14], phys_outputs2_cost)
            loss = loss + physlossval
            #print(physlossval)
            loss.backward()
            optimiser.step()
        losses.append(loss.item())
        if ep % int(epochs / 10) == 0:
            print(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.4f}")

    return losses

# Make training data
# Normalise

x_values = np.genfromtxt('CLenv_popn.csv', dtype= float, delimiter = ",")
y_values = physics_pred(x_values) #np.genfromtxt('CLenv_fn.csv', dtype= float, delimiter = ",")
min_y = np.min(y_values,axis=0)
range_y = np.max(y_values,axis=0)-min_y

x_mean = np.mean(x_values,axis=0)
x_std = np.std(x_values,axis=0)
x_norm = np.divide((x_values-x_mean),x_std)
#[iaspect,ispan,itaper,iareaH,isweep,ispeed,ialt]

y_norm = np.divide((y_values-min_y),range_y)
#[RFimpact,casm,cl,cd]

rng = default_rng()
numbers = rng.choice(x_norm.shape[0], size = x_norm.shape[0], replace=False)

bb = 30 #number of datapoints to include for training

cotrain = 1 # flag for cotraining or sequential training

if cotrain == 0:
    net = MLP()
    losses1 = trainfn1(net, x_norm, y_norm)
    plt.semilogy(losses1)
    plt.show()

    # Physics
    print("Phys start")
    ee = 3
    mesh0,mesh1,mesh2,mesh3,mesh4,mesh5,mesh6 = np.meshgrid(np.linspace(7,17,ee),np.linspace(15,35,ee),np.linspace(0.1,0.9,ee),np.linspace(0.2,0.5,ee),np.linspace(1,30,ee),np.linspace(150,273,ee),np.linspace(6.5,13.1,ee))
    physin = np.vstack((mesh0.reshape(-1),mesh1.reshape(-1),mesh2.reshape(-1),mesh3.reshape(-1),mesh4.reshape(-1),mesh5.reshape(-1),mesh6.reshape(-1))).T
    min_p = np.min(physin,axis=0)
    range_p = np.max(physin,axis=0)-min_p
    p_norm = np.divide((physin-min_p),range_p)
    physin_t = np2torch(p_norm)
    net.eval()
    phys_outputs = net.forward(physin_t).detach() #with torch.no_grad()

    phys_outputs_n = np.multiply(phys_outputs.numpy(),range_y[2:])+min_y[2:]

    phys_pred = physics_grads_nocl(physin,phys_outputs_n)
    phys_pred_t = np2torch(np.divide(phys_pred,x_std[5:]))
    print("Phys end")

    breakpoint()

    mse_cost = np.array([])
    mse_env = np.array([])

    for ep in [500, 1000, 4000]:
        for lr in [0.02,0.002,0.0002]:
            for physweight in [0.005,0.001,0.0005,0.0001,0.0]: #[0.1,0.01,0.001,0.0001,0.0]:
                aux = MLP2()
                losses2 = trainfn2(net, aux, x_norm[numbers[:bb],:], y_norm[numbers[:bb],:],physin_t, phys_outputs, phys_pred_t, ep, lr, physweight)
                pred = predict(net, aux, x_norm[numbers[bb:],:])
                mse_env = np.append(mse_env,((pred[:,0]-y_norm[numbers[bb:],0])**2).mean())
                mse_cost = np.append(mse_cost,((pred[:,1]-y_norm[numbers[bb:],1])**2).mean())
                del aux

                plt.semilogy(losses2)
                plt.show()

    np.savetxt("MSEenv.txt",mse_env)
    np.savetxt("MSEcost.txt",mse_cost)

elif cotrain == 1:

    # Physics
    print("Phys start")
    ee = 3
    mesh0,mesh1,mesh2,mesh3,mesh4,mesh5,mesh6 = np.meshgrid(np.linspace(7,17,ee),np.linspace(15,35,ee),np.linspace(0.1,0.9,ee),np.linspace(0.2,0.5,ee),np.linspace(1,30,ee),np.linspace(150,273,ee),np.linspace(6.5,13.1,ee))
    physin = np.vstack((mesh0.reshape(-1),mesh1.reshape(-1),mesh2.reshape(-1),mesh3.reshape(-1),mesh4.reshape(-1),mesh5.reshape(-1),mesh6.reshape(-1))).T
    min_p = np.min(physin,axis=0)
    range_p = np.max(physin,axis=0)-min_p
    p_norm = np.divide((physin-min_p),range_p)
    physin_t = np2torch(p_norm)

    phys_pred = physics_grads(physin)
    phys_pred_t = np2torch(np.divide(phys_pred,x_std))
    print("Phys end")

    breakpoint()

    mse_cost = np.array([])
    mse_env = np.array([])

    for ep in [500, 1000, 4000]:
        for lr in [0.02,0.002,0.0002]:
            for physweight in [0.005,0.001,0.0005,0.0001,0.0]: #[0.1,0.01,0.001,0.0001,0.0]:
                net = MLP()
                aux = MLP2()
                losses = trainfn(net, aux, x_norm[numbers[:bb],:], y_norm[numbers[:bb],:],physin_t, phys_pred_t, ep, lr, physweight)
                pred = predict(net, aux, x_norm[numbers[bb:],:])
                mse_env = np.append(mse_env,((pred[:,0]-y_norm[numbers[bb:],0])**2).mean())
                mse_cost = np.append(mse_cost,((pred[:,1]-y_norm[numbers[bb:],1])**2).mean())
                del net, aux

                plt.semilogy(losses)
                plt.show()

    np.savetxt("MSEenv.txt",mse_env)
    np.savetxt("MSEcost.txt",mse_cost)

breakpoint()

plt.scatter(x_norm[numbers[bb:],5],y_norm[numbers[bb:],1],label="Truth")
plt.scatter(x_norm[numbers[:bb],5],y_norm[numbers[:bb],1])
plt.scatter(x_norm[numbers[bb:],5],pred[:,1],label="Prediction",marker="+")
plt.legend()
plt.show()

breakpoint()