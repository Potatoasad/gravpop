import jax
import jax.numpy as jnp
from .abstract import *
from ..utils import *

class Gaussian1D(SampledPopulationModel):
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


@jit
def transform_gaussians_gauss(x,dx,mu,sigma):
    #x = K.x; Δx = K.Δx;
    denom = dx**2 + sigma**2
    x_ = (dx**2 * mu + sigma**2 * x)/denom;
    sigma_ = jnp.sqrt((sigma**2 * dx**2)/denom);
    return x_, sigma_

@jit
def logweight_gauss(x,dx,mu,sigma,a,b):
    x_, sigma_ = transform_gaussians(x,dx,mu,sigma);
    
    logC_ = logC(x_, sigma_, a, b); #print(logC_)
    logCmu = logC(mu ,sigma , a, b);# print(logCmu)
    #logCx = logC(x ,dx , a, b); #print(logCx)
    logA = logC_ - logCmu# - logCx
    return logA

@jit
def loglikelihood_kernel1d_gauss(x, dx, mu, sigma, a, b):
    logA = logweight_gauss(x,dx,mu,sigma,a,b)
    logB = jax.scipy.stats.norm.logpdf(x,loc=mu, scale=jnp.sqrt(dx**2 + sigma**2))     #normlogpdf(x,jnp.sqrt(Δx^2 + σ^2), μ)
    return logA + logB


class Gaussian1DAnalytic1D(AnalyticPopulationModel):
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



class Gaussian1DAnalytic(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['mu', 'sigma']):
        self.a = a
        self.b = b
        self.var_name = var_names[0]
        self.mu_name = hyper_var_names[0]
        self.sigma_name = hyper_var_names[1]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}
        
    def get_data(self, data, params):
        X_locations = data[self.var_name + '_mu_kernel'];
        X_scales    = data[self.var_name + '_sigma_kernel'];
        mu          = params[self.mu_name]
        sigma       = params[self.sigma_name]
        return X_locations, X_scales, mu, sigma

    def evaluate(self, data, params):
        """
        Evaluates the value of the *integrand* and not the integral
        """
        Xs          = data[self.var_name];
        mu          = params[self.mu_name]
        sigma       = params[self.sigma_name];
        #loglikes = jax.scipy.special.logsumexp( jnp.log(truncnorm(Xs, mu, sigma, low=a, high=b)) , axis=-1) - jnp.log(Xs.shape[-1])
        prob = truncnorm(Xs, mu, sigma, low=self.a, high=self.b)
        return prob
        
    def __call__(self, data, params):
        X_locations, X_scales, mu, sigma = self.get_data(data, params);
        loglikes = loglikelihood_kernel1d_gauss(X_locations, X_scales, mu, sigma, self.a, self.b)
        return jnp.exp(loglikes)

    def sample(self, df_hyper_samples, oversample=1):
        return ppd_truncCorrelatedanalytic(self, df_hyper_samples, oversample=oversample)




class Gaussian1DAnalytic(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['mu', 'sigma']):
        self.a = a
        self.b = b
        self.var_name = var_names[0]
        self.mu_name = hyper_var_names[0]
        self.sigma_name = hyper_var_names[1]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}
        
    def get_data(self, data, params):
        X_locations = data[self.var_name + '_mu_kernel'];
        X_scales    = data[self.var_name + '_sigma_kernel'];
        mu          = params[self.mu_name]
        sigma       = params[self.sigma_name]
        return X_locations, X_scales, mu, sigma

    def evaluate(self, data, params):
        """
        Evaluates the value of the *integrand* and not the integral
        """
        Xs          = data[self.var_name];
        mu          = params[self.mu_name]
        sigma       = params[self.sigma_name];
        #loglikes = jax.scipy.special.logsumexp( jnp.log(truncnorm(Xs, mu, sigma, low=a, high=b)) , axis=-1) - jnp.log(Xs.shape[-1])
        prob = truncnorm(Xs, mu, sigma, low=self.a, high=self.b)
        return prob
        
    def __call__(self, data, params):
        X_locations, X_scales, mu, sigma = self.get_data(data, params);
        loglikes = loglikelihood_kernel1d_gauss(X_locations, X_scales, mu, sigma, self.a, self.b)
        return jnp.exp(loglikes)

    def sample(self, df_hyper_samples, oversample=1):
        return ppd_truncCorrelatedanalytic(self, df_hyper_samples, oversample=oversample)