import jax.numpy as np
from jax import jit, vmap, jacrev
from functools import partial
from jax.lax import dynamic_slice
import numpy as nnp
import matplotlib.pyplot as plt

def physics(x_in): #[iaspect,ispan,itaper,iareaH,isweep,ispeed,ialt]
    AR, b, taper, tailArea, sweep, U, H = x_in

    rho = 1.225*(20-H)/(20+H); # Estimate of air density
    a = 295.1 # approx - to change
    M = U/a

    W = 70875*9.81 # Estimate of cruise weight
    range = 2700*1.852*1000 # range in m
    af_cost = 112e06 # Airframe base cost
    seats = 170# Seats
    fp = 735/1000 # Fuel price
    #thrust = 27000/2
    Isp = 3000 # thrust/(g0 * mass flow)
    flight_hours_per_year = 4000 # from flying hours #https://www.bitre.gov.au/statistics/aviation/general_aviation
    lf = 38.02 # fuselage length
    df = 3.74 # fuselage diameter
    tc = 0.1 # thickness to chord ratio

    # https://www.iata.org/contentassets/4a4b100c43794398baf73dcea6b5ad42/airline-disclosure-guide-aircraft-acquisition.pdf

    S = 124 #(b**2/AR)*(1+tailArea); # Estimate of wing area

    CL = 2*W/(rho*S*U**2) # Estimate of lift coefficient

    k_ef = 1-2*(df/lf)**2
    #k_em = -0.00152*(M/0.3-1)**10.82 + 1
    k_em = np.sqrt(1-M**2)
    f = 0.005*(1+1.5*(taper-0.6)**2)
    e = (k_ef*k_em)/((1+0.12*M**6)*(1+(0.142*f*AR*(10*tc)**0.33)/(np.cos(np.deg2rad(sweep))**2))+0.7/((4+AR)**0.8))

    Swet = np.pi*df*lf*(1+(df/lf)**2)*(1-2*df/lf)**(2/3) + 2*S*(1+0.25*tc)
    CD0 = 0.0035*Swet/S # 0.0030

    CD = CD0 + (CL**2)/(np.pi*AR*e) # Estimate of drag with compressibility correction

    # Breguet range: estimate of fuel: speed, CL, CD, (which dep on alt and other parameters) which captures all inputs from before
    w_frac = np.exp(range/(U*(CL/CD)*Isp))-1
    fuel = (w_frac*W)/9.81

    # Cost estimated from fuel: simplified version of ATR cost model
    flights_per_year = flight_hours_per_year*3600/(range/U)
    airframe_cost =  (0.06+0.1)*af_cost # 6% depreciation, 10% parts per year
    people_cost = 1500*flight_hours_per_year
    fuel_cost = fp*fuel*flights_per_year
    cost = (airframe_cost+people_cost+fuel_cost)/(seats*range/1000*flights_per_year) 

    # Environmental impact estimated from fuel burn, speed and altitude: simplified version of weighed GWP model
    # NEED TO DO AS A FOR LOOP
    env = envcalc(H*3281,fuel,30,100,flight_hours_per_year,flights_per_year,range/1000)/(seats*flights_per_year*range)

    return np.array([env,cost*100,CL,CD]) #CL,CD, env, cost

def physics_nocl(x_in, CLin): #[iaspect,ispan,itaper,iareaH,isweep,ispeed,ialt]
    AR, b, taper, tailArea, sweep, U, H = x_in
    CL, CD = CLin

    W = 70875*9.81 # Estimate of cruise weight
    range = 2700*1.852*1000 # range in m
    af_cost = 112e06 # Airframe base cost
    seats = 170# Seats
    fp = 735/1000 # Fuel price
    #thrust = 27000/2
    Isp = 3000 # thrust/(g0 * mass flow)
    flight_hours_per_year = 4000 # from flying hours #https://www.bitre.gov.au/statistics/aviation/general_aviation

    # Breguet range: estimate of fuel: speed, CL, CD, (which dep on alt and other parameters) which captures all inputs from before
    w_frac = np.exp(range/(U*(CL/CD)*Isp))-1
    fuel = (w_frac*W)/9.81

    # Cost estimated from fuel: simplified version of ATR cost model
    flights_per_year = flight_hours_per_year*3600/(range/U)
    airframe_cost =  (0.06+0.1)*af_cost # 6% depreciation, 10% parts per year
    people_cost = 1500*flight_hours_per_year
    fuel_cost = fp*fuel*flights_per_year
    cost = (airframe_cost+people_cost+fuel_cost)/(seats*range/1000*flights_per_year) 

    # Environmental impact estimated from fuel burn, speed and altitude: simplified version of weighed GWP model
    env = envcalc(H*3281,fuel,30,100,flight_hours_per_year,flights_per_year,range/1000)/(seats*flights_per_year*range)

    return np.array([env,cost*100]) #CL,CD, env, cost

## ENVIRONMENTAL IMPACT MODEL (SHORTENED)

class emittor:
    # Class attributes
    EI: float
    RF: staticmethod
    s: np.array
    ls: float # long or short or contrail, -1,0,1

    # Constructor
    def __init__(self, EI, RF, s, ls):
        self.EI = EI
        self.RF = RF
        self.s = s
        self.ls = ls

        return

    # Methods
    @partial(jit, static_argnums=(0,))
    def RFcalc_s(self,alt,E):
        xalt = np.array([17509.15750915751,  19523.809523809523,  21538.46153846154,  23516.483516483517,  25531.13553113553,  27509.157509157507,  29487.179487179485, 31501.8315018315,  33516.48351648352,  35494.505494505494,  37509.15750915751,  39523.80952380953,  41501.831501831504])
        sAlt = np.interp(alt,xalt,self.s)
        RFval = np.multiply(np.multiply(sAlt,self.RF(1)),E)

        return RFval
    
    @partial(jit, static_argnums=(0,))
    def RFcalc_l(self,alt,span,E):
        xalt = np.array([17509.15750915751,  19523.809523809523,  21538.46153846154,  23516.483516483517,  25531.13553113553,  27509.157509157507,  29487.179487179485, 31501.8315018315,  33516.48351648352,  35494.505494505494,  37509.15750915751,  39523.80952380953,  41501.831501831504])
        sAlt = np.interp(alt,xalt,self.s)
        RFval = np.zeros(len(span))
        mat1 = np.tile(span,(len(span),1))
        tmint = np.transpose(mat1) - np.tile(span,(len(span),1))
        RFfn = self.RF(tmint)*(tmint>0)
        RFval = vmap(lambda RF, sAlt, E, span: sAlt*np.trapezoid(RF*E,span), (0,None,None,None))(RFfn,sAlt,E,span)
        #for t in span:
        #    RFval = RFval.at[ii].set(sAlt*jnp.trapz(self.RF(t-span[:t])*E,span[:t]))
        #    ii+=1

        return RFval
    
    @partial(jit, static_argnums=(0,))
    def RFcalc_c(self,alt,dist):
        xalt = np.array([17509.15750915751,  19523.809523809523,  21538.46153846154,  23516.483516483517,  25531.13553113553,  27509.157509157507,  29487.179487179485, 31501.8315018315,  33516.48351648352,  35494.505494505494,  37509.15750915751,  39523.80952380953,  41501.831501831504])
        sAlt = np.interp(alt,xalt,self.s)
        RFval = np.multiply(np.multiply(sAlt,self.RF(1)),dist)

        return RFval

def envcalc(H,fuel,tlife,tmax,flight_hours_per_year,flights_per_year,range,r=0.03):
    RF_sum = np.zeros(tmax)
    years = np.arange(tmax)+1

    # Add in the emittors
    GCO2 = lambda t: 1.80e-15*(1+0.259*(np.exp(-t/172.9)-1)+0.338*(np.exp(-t/18.51)-1)+0.186*(np.exp(-t/1.186)-1))
    CO2 = emittor(3.16, GCO2, np.ones(13), -1.)

    NOxEI = 15.14e-03 # can change with engine-dependent model later
    GCH4 = lambda t: -5.16e-13*np.exp(np.divide(-t,12.0))
    GO3L = lambda t: -1.21e-13*np.exp(np.divide(-t,12.0))

    sCH4_O3L = np.array([0.869281045751634, 0.9248366013071896, 0.9575163398692811, 0.9673202614379085, 0.9477124183006537, 0.934640522875817, 0.9281045751633987, 0.9411764705882353, 0.977124183006536, 1.1372549019607843, 1.212418300653595, 1.2026143790849673, 1.2026143790849673])
    CH4 = emittor(NOxEI, GCH4, sCH4_O3L, -1.)
    O3L = emittor(NOxEI, GO3L, sCH4_O3L, -1.)

    sO3s = np.array([0.470588235294117, 0.558823529411764, 0.620915032679738, 0.712418300653594, 0.712418300653594, 0.813725490196078, 0.931372549019607, 1.00653594771241, 1.13071895424836, 1.4313725490196, 1.62745098039215, 1.80065359477124, 1.93464052287581])
    NOxShort = emittor(NOxEI, lambda t: 1.01e-11, sO3s, 0.)

    sAIC = np.array([0.032679738562091505, 0, 0, 0.17320261437908496, 0.4019607843137255, 0.7973856209150327, 1.2549019607843137, 1.7091503267973858, 2.104575163398693, 1.8202614379084967, 1.5359477124183007, 0.9673202614379085, 0.7941176470588236])
    AIC = emittor(1.245, lambda t: 2.21e-12, sAIC, 1.)

    soot = emittor(4e-05, lambda t: 5e-10, np.ones(13), 0.)

    SO4 = emittor(2e-04, lambda t: -1e-10, np.ones(13), 0.)

    H2O = emittor(1.26, lambda t: 7.14e-15, np.ones(13), 0.)

    emittorList = [CO2, CH4, O3L, NOxShort, AIC, soot, SO4, H2O]

    # Year by year integration: depends on how integrate into SUAVE
    for i in emittorList:
        Eval = fuel*i.EI*flight_hours_per_year
        RF_sum += i.RFcalc_s(H,E=Eval)*np.ones(tmax)*(i.ls==0.) + \
            i.RFcalc_l(H,years,E=Eval)*np.ones(tmax)*(i.ls==-1.) + \
            i.RFcalc_c(H,dist=range*flights_per_year)*np.ones(tmax)*(i.ls==1.)

    wr = np.ones(tmax)*np.array(years<=tlife) + (np.divide(1,((1+r)**(years-tlife))))*np.array(years>tlife)
    ATRh = np.trapezoid(np.multiply(RF_sum,wr),x=years)/tlife

    # Convert to GWP
    GWPeq = (ATRh/1.80e-15)

    return GWPeq

def physics_grads_nocl(x_in, CLin):
    jacobianfn = jacrev(physics_nocl)
    y_diff = vmap(jacobianfn,(0,0))(np.array(x_in), np.array(CLin))
    return nnp.array(y_diff[:,:,5:])

def physics_grads(x_in):
    jacobianfn = jacrev(physics)
    y_diff = vmap(jacobianfn,0)(np.array(x_in))
    return nnp.array(y_diff)

def physics_pred(x_values):
    return nnp.array(vmap(physics,0)(np.array(x_values)))


# Reformulated version - this is where do derivatives and various other things

# Test with actual data

folder = "./Images/PhysModelTest/"

# Make training data
# Normalise
x_values = np.array(nnp.genfromtxt('CLenv_popn.csv', dtype= float, delimiter = ","))
y_values = np.array(nnp.genfromtxt('CLenv_fn.csv', dtype= float, delimiter = ","))
min_y = np.min(y_values,axis=0)
range_y = np.max(y_values,axis=0)-min_y
min_x = np.min(x_values,axis=0)
range_x = np.max(x_values,axis=0)-min_x

#x_norm = np.divide((x_values-np.mean(x_values,axis=0)),np.std(x_values,axis=0))
x_norm = np.divide((x_values-min_x),range_x)
xlab = ["AR","Span","Taper","Tail area","Sweep","Speed","Altitude"]
y_norm = np.divide((y_values-min_y),range_y)
ylab = ["wGWP","CASM","CL","CD"]
y_pred = vmap(physics,0)(x_values)
y_pred_norm = np.divide((y_pred-np.min(y_pred,axis=0)),(np.max(y_pred,axis=0)-np.min(y_pred,axis=0)))

jacobianfn = jacrev(physics)

y_diff = vmap(jacobianfn,0)(x_values)

breakpoint()

abs_err = np.round(abs(y_pred-y_values)*100/(y_values),2)
print(abs_err.mean(axis=0))
rel_err = np.round(abs(y_pred_norm-y_norm)*100/(y_norm),2)

Rval = np.corrcoef(x_norm.T,y_norm.T)

for jj in range(4):
    fig,axs = plt.subplots(3,3,layout="constrained")
    for ii in range(7):
        ax = axs.flat[ii]
        ax.scatter(x_values[:,ii],y_pred[:,jj],marker="x",label="Simple model")
        ax.scatter(x_values[:,ii],y_values[:,jj],marker="+",label="SUAVE")
        ax.set_xlabel(xlab[ii])
        ax.set_ylabel(ylab[jj])
        ax.set_title(str(round(Rval[ii,7+jj], 2)))
    #plt.savefig(folder+ylab[jj]+".png")
        ax.set_xlim([np.quantile(x_values[:,ii], 0.01), np.quantile(x_values[:,ii], 0.99)])
        ax.set_ylim([np.quantile(y_pred[:,jj], 0.01), np.quantile(y_pred[:,jj], 0.99)])
    plt.show()

for jj in range(4):
    fig,axs = plt.subplots(2,2,layout="constrained")
    for ii in range(4):
        ax = axs.flat[ii]
        ax.scatter(y_values[:,ii],y_values[:,jj],marker="+",label="SUAVE")
        ax.scatter(y_pred[:,ii],y_pred[:,jj],marker="x",label="Simple model")
        ax.set_xlabel(ylab[ii])
        ax.set_ylabel(ylab[jj])
        ax.set_title(str(round(Rval[ii+7,jj+7], 2)))
    #plt.savefig(folder+ylab[jj]+"_auto.png")
        ax.set_xlim([np.quantile(y_pred[:,ii], 0.01), np.quantile(y_pred[:,ii], 0.99)])
        ax.set_ylim([np.quantile(y_pred[:,jj], 0.01), np.quantile(y_pred[:,jj], 0.99)])
    plt.show()
    fig,axs = plt.subplots(2,2,layout="constrained")
    for ii in range(4):
        ax = axs.flat[ii]
        ax.scatter(y_pred_norm[:,ii],y_pred_norm[:,jj],marker="x",label="Simple model")
        ax.scatter(y_norm[:,ii],y_norm[:,jj],marker="+",label="SUAVE")
        ax.set_xlabel(ylab[ii])
        ax.set_ylabel(ylab[jj])
        ax.set_title(str(round(Rval[ii+7,jj+7], 2)))
    #plt.savefig(folder+ylab[jj]+"_auto.png")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    plt.show()
