import jax
import jax.numpy as jnp
from ..utils import *
from ..generic import *

class TruncatedGaussian1D(SampledPopulationModel):
    r"""
    Truncated Gaussian Distribution. 

    Event Level Parameters:         :math:`x`
    Population Level Parameters:    :math:`\mu, \sigma` 

    .. math::
    
        P(x | \mu, \sigma) = \mathcal{N}_{[a,b]}))(x | \mu, \sigma) 
    """
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['mu', 'sigma']):
        self.a = a
        self.b = b
        self.var_name = var_names[0]
        self.mu_name = hyper_var_names[0]
        self.sigma_name = hyper_var_names[1]
        
    def get_data(self, data, params):
        Xs          = data[self.var_name];
        mu          = params[self.mu_name]
        sigma       = params[self.sigma_name]
        return Xs, mu, sigma
    
    def __call__(self, data, params):
        Xs, mu, sigma = self.get_data(data, params);
        #loglikes = jax.scipy.special.logsumexp( jnp.log(truncnorm(Xs, mu, sigma, low=a, high=b)) , axis=-1) - jnp.log(Xs.shape[-1])
        prob = truncnorm(Xs, mu, sigma, low=self.a, high=self.b)
        return prob



from jax import jit
@jit
def logC(mu, sigma, a , b):
    return jax.scipy.special.logsumexp(
                jnp.stack([jax.scipy.stats.norm.logcdf(b, loc=mu, scale=sigma),
                           jax.scipy.stats.norm.logcdf(a, loc=mu, scale=sigma)],
                          axis=-1),
                b=jnp.array([1,-1]),
                axis=-1
            )

@jit
def transform_gaussians(x,dx,mu,sigma):
    #x = K.x; Δx = K.Δx;
    denom = dx**2 + sigma**2
    x_ = (dx**2 * mu + sigma**2 * x)/denom;
    sigma_ = jnp.sqrt((sigma**2 * dx**2)/denom);
    return x_, sigma_

@jit
def logweight(x,dx,mu,sigma,a,b):
    x_, sigma_ = transform_gaussians(x,dx,mu,sigma);
    
    logC_ = logC(x_, sigma_, a, b); #print(logC_)
    logCmu = logC(mu ,sigma , a, b);# print(logCmu)
    logCx = logC(x ,dx , a, b); #print(logCx)
    logA = logC_ - logCmu - logCx
    return logA

@jit
def loglikelihood_kernel1d(x, dx, mu, sigma, a, b):
    logA = logweight(x,dx,mu,sigma,a,b)
    logB = jax.scipy.stats.norm.logpdf(x,loc=mu, scale=jnp.sqrt(dx**2 + sigma**2))     #normlogpdf(x,jnp.sqrt(Δx^2 + σ^2), μ)
    return logA + logB


class TruncatedGaussian1DAnalytic(AnalyticPopulationModel):
    r"""
    Truncated Gaussian Distribution. Evaluates an analytical expression for the population likelihood of a truncated
    gaussian distribution evaluated over data distributed as a truncated gaussian as well. Useful when our data is represented as a truncated gaussian mixture. 
    This class evaluated over the components of such a mixture, would allow us to evaluate the population likelihood of arbitrary data distributions represented using truncated gaussian mixtures. 

    Event Level Parameters:         :math:`x \sim \mathcal{N}_{[a,b]}(x_0, \Delta x)` 
    Population Level Parameters:    :math:`\mu, \sigma` 

    .. math::
    
        P(x | \mu, \sigma) = \mathcal{N}_{[a,b]}))(x | \mu, \sigma) \textrm{ where } x \sim \mathcal{N}_{[a,b]}(x_0, \Delta x)
    """
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['mu', 'sigma']):
        self.a = a
        self.b = b
        self.var_name = var_names[0]
        self.mu_name = hyper_var_names[0]
        self.sigma_name = hyper_var_names[1]
        
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
        loglikes = loglikelihood_kernel1d(X_locations, X_scales, mu, sigma, self.a, self.b)
        return jnp.exp(loglikes)



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

