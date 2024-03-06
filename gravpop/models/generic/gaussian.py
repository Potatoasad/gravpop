import jax
import jax.numpy as jnp
from .abstract import *
from ..utils import *

class Gaussian1D(AbstractPopulationModel):
    r"""
    Gaussian Distribution. Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`x`
    Population Level Parameters:    :math:`\mu, \sigma` 

    .. math::
    
        P(x | \mu, \sigma) = \mathcal{N}_{[a,b]}))(x | \mu, \sigma) 
    """
    def __init__(self, var_name='x', mu_name='mu', sigma_name='sigma'):
        self.var_name = var_name
        self.mu_name = mu_name
        self.sigma_name = sigma_name
        
    def get_data(self, data, params):
        Xs          = data[self.var_name];
        mu          = params[self.mu_name]
        sigma       = params[self.sigma_name]
        return Xs, mu, sigma
    
    def __call__(self, data, params):
        Xs, mu, sigma = self.get_data(data, params);
        pdf = jax.scipy.stats.norm.pdf(Xs, loc=mu, scale=sigma).mean(axis=-1)
        return pdf
