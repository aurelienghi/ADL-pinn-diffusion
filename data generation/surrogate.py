from simplelhs import LatinHypercubeSampling
from wing_env_CL import despt
import scipy
import numpy as np

def mitchMor(x):

    dist = scipy.spatial.distance.pdist(x, 'euclidean')

    # In ascending order
    q = [1,2,5,10,20,50]
    phi_i = 0
    for mm in range(6):
        phi = np.sum(np.power(dist,-q[mm]))**(1/q[mm])
        if phi>phi_i:
            phi_i = phi

    return phi

initpop = 1500
dim = 7
lbound = np.array([7, 15, 0.1, 0.2, 0, 150, 6.5])
ubound = np.array([17, 35, 0.9, 0.5, 30, 273, 13.1])
#iaspect,ispan,itaper,iareaH,isweep
diff = ubound-lbound

# Initial Latin Hypercube sample
lhs = LatinHypercubeSampling(dim)  # inputs for surrogate
hc_i = np.zeros([initpop,dim])
crit_i = 1e05
for kk in range(100): # choose the best in 100 random samples
    hc = lhs.random(initpop)  # initial points
    crit = mitchMor(hc)
    print(crit)
    if crit<crit_i:
        crit_i = crit
        hc_i = hc

print("points done")
npts = initpop
normpts = hc_i

filestr = "allpop"+".csv"

evalpts = np.multiply(normpts,diff)+lbound

allpts = evalpts
allnormpts = normpts

np.savetxt("points"+filestr, allpts, delimiter=",")

funcval = np.zeros([int(npts),4])
for jj in range(int(npts)):
    funcval[jj,:] = despt(evalpts[jj,:])
    if jj % 100 == 0:
        np.savetxt("evals_up_to_"+str(jj)+filestr, funcval, delimiter=",")

np.savetxt("evals"+filestr, funcval, delimiter=",")

# Train the surrogate - gotta test it, and retrain every time to prevent becoming too biased
#surrogatemodel = trainfn(allnormpts,allfunc)
    

