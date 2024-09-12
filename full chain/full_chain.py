import numpy as np

print("----- STEP 1: GENERATING OPTIMAL AIRCRAFT -----")

pop = int(input("How many aircraft to generate? >> "))
divyn = input("Include population diversity enhancement (y/n)? >> ")

if divyn == "y":
    div = 1
    weight = float(input("Diversity enhancement weight? >> "))
elif divyn == "n":
    div = 0
    weight = 0
else:
    print("input only y/n")

from opt_sim_phys import opt

lb = np.array([7, 15, 0.1, 0.2, 0, 150, 6.5])
ub = np.array([17, 35, 0.9, 0.5, 30, 273, 13.1])

x_values, y_values, dy_values = opt(pop,lb,ub,div,weight)

np.savetxt("population.txt",x_values)
np.savetxt("objectives.txt",y_values)
np.savetxt("derivatives.txt",dy_values.reshape(dy_values.shape[0], -1))

print("Population generated, saved results. Continue?")
breakpoint()

print("----- STEP 2: POPULATION AND DERIVATIVES IN NORMALIZED COORDINATES -----")

# Normalise
min_y = np.min(y_values,axis=0)
range_y = np.max(y_values,axis=0)-min_y

x_mean = np.mean(x_values,axis=0)
x_std = np.std(x_values,axis=0)
x_norm = np.divide((x_values-x_mean),x_std)
#[iaspect,ispan,itaper,iareaH,isweep,ispeed,ialt]

y_norm = np.divide((y_values-min_y),range_y)
#[RFimpact,casm,cl,cd]

dy_norm = np.multiply(np.divide(dy_values,np.transpose(np.tile(range_y,(y_norm.shape[0],len(ub),1)),(0,2,1))),x_std)

print("Normalization finished. Continue?")
breakpoint()

print("----- STEP 3: TRAINING THE PINN -----")

from numpy.random import default_rng

bb = int(round(float(input("Input the test/train split as train/total. >> "))*pop))
rng = default_rng()
numbers = rng.choice(x_norm.shape[0], size = x_norm.shape[0], replace=False)

lr = float(input("Learning rate >> "))

from pinn_module import make_pinn

physin = x_values[numbers[:bb],:]
min_p = np.min(physin,axis=0)
range_p = np.max(physin,axis=0)-min_p
p_norm = np.divide((physin-min_p),range_p)

make_pinn(x_norm,y_norm,p_norm,dy_norm[numbers[:bb],:],numbers,bb,lr)

breakpoint()