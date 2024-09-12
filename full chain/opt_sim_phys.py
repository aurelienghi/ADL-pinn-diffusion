from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from phys_model_jax import physics_grads, physics_pred
import matplotlib.pyplot as plt
import numpy as np
import mitchMor

class envcostprob(Problem):

    def __init__(self, lb, ub):
        super().__init__(n_var=len(ub), n_obj=4, n_ieq_constr=0, xl=lb, xu=ub)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = physics_pred(x)

class envcostprob_div(Problem):

    def __init__(self, weight, lb, ub):
        super().__init__(n_var=len(ub), n_obj=4, n_ieq_constr=0, xl=lb, xu=ub)
        self.weight = weight

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = physics_pred(x) + (self.weight/(20*np.sqrt(6)))*np.tile(mitchMor(x),(x.shape[0],2))


def opt(pop,lb,ub,div=0,weight=0):
    if div == 0:
        problem = envcostprob(lb, ub)
    else:
        problem = envcostprob_div(weight, lb, ub)
    algorithm = NSGA2(pop_size=pop)
    res = minimize(problem,
        algorithm,
        seed=1,
        verbose=False,
        save_history=True,
        termination=get_termination("n_gen", 30))
    
    return res.X, res.F, physics_grads(res.X)