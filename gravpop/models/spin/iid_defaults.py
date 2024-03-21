import jax
import jax.numpy as jnp
from ..utils import *
from ..generic import *



def alpha_beta_max_to_mu_var_max(alpha, beta, amax):
    mu = alpha / (alpha + beta) * amax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax ** 2
    return mu, var, amax


def mu_var_max_to_alpha_beta_max(mu, var, amax):
    mu /= amax
    var /= amax ** 2
    alpha = (mu ** 2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta, amax


class GaussianIsotropicSpinOrientationsIID(AbstractPopulationModel):
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

	
	def __call__(self, data, params):
		xi_spin = params[self.hyper_var_names[0]]
		sigma_spin = params[self.hyper_var_names[1]]
		prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
		prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
		prob *= xi_spin
		prob += (1-xi_spin)/4
		return prob



class BetaSpinMagnitudeIID(AbstractPopulationModel):
	r"""
	Mixture of gaussian and isotropic distribution over spin orientations.
	Performs a monte carlo estimate of the population likelihood. 

	Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
	Population Level Parameters:    :math:`\xi, \sigma` 

	.. math::
	
		P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
	"""
	def __init__(self, var_names=['a_1', 'a_2'], hyper_var_names=['alpha_chi','beta_chi','amax'],parameterization="mu_sigma"):
		self.parameterization = parameterization
		self.converter = lambda x,y,z: (x,y,z)
		self.var_names = var_names
		self.hyper_var_names = hyper_var_names

		hyper_var_names_are_alpha_beta = (("alpha" in self.hyper_var_names[0]) or (("beta" in self.hyper_var_names[1])))

		if (parameterization in ("mu_var")) and hyper_var_names_are_alpha_beta:
			self.converter = lambda mu,var,amax=1 : mu_var_max_to_alpha_beta_max(mu, var, amax)
		if (parameterization in ("mu_sigma")) and hyper_var_names_are_alpha_beta:
			self.converter = lambda mu,sigma,amax=1 : mu_var_max_to_alpha_beta_max(mu, sigma**2, amax)
	
	def __call__(self, data, params):
		amax = params.get(self.hyper_var_names[2], 1)
		alpha_chi, beta_chi, amax = self.converter(params[self.hyper_var_names[0]], params[self.hyper_var_names[1]], amax)
		prob  = beta(data[self.var_names[0]], alpha=alpha_chi, beta=beta_chi, scale=amax)
		prob *= beta(data[self.var_names[1]], alpha=alpha_chi, beta=beta_chi, scale=amax)
		return prob
