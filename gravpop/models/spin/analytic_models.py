import jax
import jax.numpy as jnp
from ..utils import *
from ..generic import *
from .truncated_gaussian import *


class GaussianIsotropicSpinOrientationsIIDAnalytic(AnalyticPopulationModel, SpinPopulationModel):
	r"""
	Mixture of gaussian and isotropic distribution over spin orientations.
	Performs a monte carlo estimate of the population likelihood. 

	Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
	Population Level Parameters:    :math:`\xi, \sigma` 

	.. math::
	
		P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
	"""
	def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','sigma_spin'], a=-1, b=1):
		self.a = a
		self.b = b
		self.var_names = var_names
		self.hyper_var_names = hyper_var_names
		self.models = [TruncatedGaussian1DAnalytic(a=a, b=b, var_names=[var_names[i]], hyper_var_names=['mu_spin', hyper_var_names[1]]) for i in range(len(var_names))]

	def evaluate(self, data, params):
		xi_spin = params[self.hyper_var_names[0]]
		sigma_spin = params[self.hyper_var_names[1]]
		prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
		prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
		prob *= xi_spin
		prob += (1-xi_spin)/4
		return prob
	
	def __call__(self, data, params):
		xi_spin = params[self.hyper_var_names[0]]
		sigma_spin = params[self.hyper_var_names[1]]
		new_params = {**params, 'mu_spin':1}
		prob  = self.models[0](data, new_params)           #truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
		prob *= self.models[1](data, new_params)
		prob *= xi_spin
		prob += (1-xi_spin)/4
		return prob


class IIDTruncatedGaussian1DAnalytic(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['mu', 'sigma']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b, 'hyper_var_names' : hyper_var_names}
        self.models = [TruncatedGaussian1DAnalytic(var_names=[var_name], **kwargs) for var_name in self.var_names]

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


class TruncatedGaussian1DIndependentAnalytic(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b}
        self.models = [TruncatedGaussian1DAnalytic(var_names=[var_names[i]], 
                                                   hyper_var_names=[hyper_var_names[2*i], hyper_var_names[2*i + 1]], 
                                                   **kwargs) for i in range(len(self.var_names))]

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
