from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
import scipy
import MLP_NN
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

class envcostprob(Problem):

    def __init__(self):
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=0, xl=np.array([0, 0, 0, 0]), xu=np.array([1, 1, 1, 1]))

    def _evaluate(self, x, out, *args, **kwargs):
        input = np.concatenate((x,altspeedval),axis=1)
        out["F"] = surrogate(input) #predict(input)

def surrogate(x,pop=100):
    val = model(torch.as_tensor(np.float32(x)))
    crit = mitchMor(x)
    return val.detach().numpy() + (weight/(100*np.sqrt(6)))*np.tile(crit,(pop,2)) #population

def surrogatetrue(x):
    val = model(torch.as_tensor(np.float32(x)))
    return val.detach().numpy()

def optStep(pop,vis=0):
    problem = envcostprob()
    #print(altspeedval)
    algorithm = NSGA2(pop_size=pop)
    res = minimize(problem,
        algorithm,
        seed=1,
        verbose=False,
        save_history=True,
        termination=get_termination("n_gen", 300))
    if vis == 1:
        visualize(problem,res)
    return res

def predict(xin,lb,diff,minf,range):
    xs = np.divide((xin-lb),diff)
    out = SurrogateModel(xs)
    outsc = np.multiply(out,range)+minf
    return outsc

def mitchMor(x):
    dist = scipy.spatial.distance.pdist(x, 'euclidean')

    # In ascending order
    q = [1,2,5,10] #ignore 20,50
    phi_i = 0
    for mm in range(4):
        phi = np.sum(np.power(dist,-q[mm]))**(1/q[mm])
        if phi>phi_i:
            phi_i = phi

    return phi

def visualize(problem,res):
    #plot = Scatter()
    #plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    #plot.add(res.F, facecolor="none", edgecolor="red")
    #plot.show()
    mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 14
    })
    max = np.array([0.27581882, 22.01293723])
    min = np.array([0.11410737, 7.16842504])
    range = max-min
    breakpoint()
    plt.scatter((res.F[:,0]*range[0]+min[0])*1000,res.F[:,1]*range[1]+min[1],"r+",markersize='10')
    plt.xlabel("$CO_2$ emission equivalent [g per available seat mile]")
    plt.ylabel("Cost per available seat mile [$\mbox{\\textcent}$]")
    plt.show()

global altspeedval
global model
global weight

#onnx_model = onnx.load("ML_surrogate.onnx")
#SurrogateModel = lambda input: onnx2xla.interpret_onnx(onnx_model.graph, input)[0]

input_size = 6 #size of information for each flow condition
out_size = 2 #size of coeff
hidden_layer_size = 8 #size of hidden layer
hidden_layer_2nd = 10 #size of 2nd hidden layer
epochs = 50
lr = 5e-03
weight = 0.09

model = MLP_NN.WPSpredict(out_size,input_size,hidden_layer_size,hidden_layer_2nd)
model.load_state_dict(torch.load("savefile.pth"))

lbound = np.array([90, 7, 10, 0.2, 150, 6.5])
ubound = np.array([150, 24, 34, 0.5, 273, 13.1])
diff = ubound-lbound
#maxf = 
#minf = 
#range = maxf-minf
population = 100
nn = 30
alt = np.linspace(0,0.99,nn)
speed = np.linspace(0,0.99,nn)
iter = 0

#maxMitchMor = ?
#in our case, all distances are between zero and 1, so the largest possible pairwise distance is 1/sqrt(6)
# max mitch morris = sum_pop(sqrt(6)^q)**(1/q) = (100*sqrt(6)^q)**(1/q) = 100^1/q * sqrt(6) = 100*sqrt(6)
# note this maximum is not actually feasible.

for iter_s in range(nn):
    for iter_a in range(nn):

        # use fixed altitude and speed values
        altspeedval = np.tile(np.array([alt[iter_a],speed[iter_s]]), (population,1))
        results = optStep(population,1)
        breakpoint()
        if iter == 0:
            normpt = results.X
            popleft = len(results.X)
            altleft = altspeedval[:popleft,:]
            fvals = np.concatenate((surrogatetrue(np.concatenate((results.X,altleft),axis=1)),altleft),axis=1)
        else:
            normpt = np.append(normpt,results.X,axis=0)
            popleft = len(results.X)
            altleft = altspeedval[:popleft,:]
            fvals = np.append(fvals,np.concatenate((surrogatetrue(np.concatenate((results.X,altleft),axis=1)),altleft),axis=1),axis=0)
        iter +=1

normpts = np.unique(normpt,return_index = True, axis=0)

np.savetxt("x_vals.csv", normpts[0], delimiter=",")
np.savetxt("z_vals.csv", fvals[normpts[1]], delimiter=",")

#Maximum:
#[ 0.27581882 22.01293723]
#Minimum:
#[0.11410737 7.16842504]