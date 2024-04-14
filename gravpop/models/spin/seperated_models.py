import jax
import jax.numpy as jnp
from ..generic import *

class MixtureTruncatedGaussian1DFixedZero(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1'], hyper_var_names=['mu_chi_a', 'sigma_chi_a', 'sigma_chi_b', 'eta']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.eta_variable = self.hyper_var_names[-1]
        kwargs = {'a' : a, 'b' : b}
        self.models = [TruncatedGaussian1DAnalytic(var_names=[var_names[0]], 
                                                   hyper_var_names=[hyper_var_names[0], hyper_var_names[1]], 
                                                   **kwargs),
                       TruncatedGaussian1DAnalytic(var_names=[var_names[0]], 
                                                   hyper_var_names=['mu_zero_spin_1d_fixed', hyper_var_names[2]], 
                                                   **kwargs),]

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.eta_variable]  * self.models[0](data, params)
        result += (1 - params[self.eta_variable]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.eta_variable]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.eta_variable]) * self.models[1].evaluate(data, params)
        return result



class TruncatedGaussian1DMixtureChi1StandardChi2(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1_a', 'sigma_chi_1_a', 'sigma_chi_1_b', 'eta_1', 'mu_chi_2', 'sigma_chi_2']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b}
        chi_1_var_names = [var_names[0]]
        chi_2_var_names = [var_names[1]]

        chi_1_hyper_var_names = self.hyper_var_names[0:4]
        chi_2_hyper_var_names = self.hyper_var_names[4:6]

        self.models = [MixtureTruncatedGaussian1DFixedZero(var_names=chi_1_var_names, hyper_var_names=chi_1_hyper_var_names, a=a[0], b=b[0]),
                       TruncatedGaussian1DAnalytic(var_names=chi_2_var_names, hyper_var_names=chi_2_hyper_var_names, a=a[1], b=b[1])]

    def __call__(self, data, params):
        result = self.models[0](data, params)
        for i in range(1,len(self.models)):
            result *= self.models[i](data, params)
        return result

    def evaluate(self, data, params):
        result = self.models[0].evaluate(data, params)
        for i in range(1,len(self.models)):
            result *= self.models[i].evaluate(data, params)
        return result