#########################################################################
### Truncated 2D Gaussian
########################################################################

from ..utils import *
from .abstract import *
from ..sample import *

import jax
import jax.numpy as jnp
from typing import Union, List, Dict
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CovarianceMatrix2D:
    s1  : Union[jax.Array,float]
    s2  : Union[jax.Array,float]
    rho : Union[jax.Array,float]
    
    def inv(self):
        rhom1 = jnp.sqrt(1-self.rho**2)
        s1_ = 1 / ( self.s1 * rhom1 )
        s2_ = 1 / ( self.s2 * rhom1 )
        rho_ = -self.rho
        return CovarianceMatrix2D(s1_, s2_, rho_)
    
    def __add__(self, other):
        s1_ = jnp.sqrt(self.s1**2 + other.s1**2)
        s2_ = jnp.sqrt(self.s2**2 + other.s2**2)
        rho_ =  self.rho * self.s1 * self.s2
        rho_ += other.rho * other.s1 * other.s2
        rho_ /= s1_ * s2_
        return CovarianceMatrix2D(s1_, s2_, rho_)
    
    def unpack(self):
        return self.s1, self.s2, self.rho
    
    @property
    def value(self):
        s1,s2,rho = self.unpack()
        return jnp.array([[s1**2, s1*s2*rho],[s1*s2*rho, s2**2]])
    
    
@dataclass(frozen=True)
class Vector2D:
    x1  : Union[jax.Array,float]
    x2  : Union[jax.Array,float]
    
    def __rmul__(self, cov_matrix):
        s1, s2, rho = cov_matrix.unpack()
        x1_ = s1**2 * self.x1 + rho * s1 * s2 * self.x2
        x2_ = rho * s1 * s2 * self.x1 + s2**2 * self.x2
        return Vector2D(x1_, x2_)
    
    def __add__(self, vector):
        return Vector2D(self.x1 + vector.x1, self.x2 + vector.x2)
    
    @property
    def value(self):
        x1 = self.x1
        x2 = self.x2
        return jnp.array([x1, x2])
    
    
    
def probability_mass(Mu, Sigma, a=[0,0], b=[1,1]):
    return mvnorm2d(Mu.x1, Mu.x2, Sigma.s1, Sigma.s2, a[0], a[1], b[0], b[1], Sigma.rho)

@dataclass(frozen=True)
class TruncatedNormal2D:
    mu : Vector2D
    sigma : CovarianceMatrix2D
    a : List[float] = field(default_factory=lambda: [0.0, 0.0])
    b : List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    def probability_mass(self):
        return probability_mass(self.mu, self.sigma, a=self.a, b=self.b)



def transform_component(mu_1, sigma_1, mu_2, sigma_2):
    Lambda_1 = sigma_1.inv()
    Lambda_2 = sigma_2.inv()
    sigma_ = (Lambda_1 + Lambda_2).inv()    
    mu_ = sigma_ * ( Lambda_1 * mu_1 + Lambda_2 * mu_2 )
    return mu_, sigma_

@jit                  
def pack_values(mu_11=0.0, mu_12=0.0, mu_21=0.0, mu_22=0.0,
                       sigma_11=1.0, sigma_12=1.0, sigma_21=1.0, sigma_22=1.0,
                       rho_1=0.0, rho_2=0.0):
    mu_1 = Vector2D(mu_11, mu_12)
    sigma_1 = CovarianceMatrix2D(sigma_11, sigma_12, rho_1)
    mu_2 = Vector2D(mu_21, mu_22)
    sigma_2 = CovarianceMatrix2D(sigma_21, sigma_22, rho_2)
    return mu_1, sigma_1, mu_2, sigma_2

def make_cov(sig_1, sig_2, rho):
    return jnp.stack([jnp.stack([sig_1**2, sig_1*sig_2*rho],axis=-1)
                     ,jnp.stack([sig_1*sig_2*rho, sig_2**2],axis=-1)], axis=-1)


multivariate_normal_pdf = jax.vmap(jax.scipy.stats.multivariate_normal.pdf, in_axes=(0, 0, 0))

def compute_normal_pdf(mu_11, mu_12, sig_11, sig_12, rho_1,
                        mu_21, mu_22, sig_21, sig_22, rho_2):
    #argss = (mu_11, mu_12, sig_11, sig_12, rho_1, mu_21, mu_22, sig_21, sig_22, rho_2)
    #for a in argss:
    #    print(a)
    mu_1 = jnp.stack([mu_11, mu_12], axis=-1);
    mu_2 = jnp.stack([mu_21, mu_22], axis=-1);
    sig_1 = make_cov(sig_11, sig_12, rho_1)#jnp.stack([jnp.stack([sig_11**2, sig_11*sig_12*rho_1]),jnp.stack([sig_11*sig_12*rho_1, sig_12**2])])
    sig_2 = make_cov(sig_21, sig_22, rho_2)#jnp.stack([jnp.stack([sig_21**2, sig_21*sig_22*rho_2]),jnp.stack([sig_21*sig_22*rho_2, sig_22**2])])
    return jax.scipy.stats.multivariate_normal.pdf(mu_1, mu_2, sig_1 + sig_2)


@jit
def compute_likelihood(mu_11, mu_12, mu_21, mu_22,
                       sigma_11, sigma_12, sigma_21, sigma_22,
                       rho_1, rho_2, a=[0.0, 0.0], b=[1.0, 1.0]):
    mu_1 = Vector2D(mu_11, mu_12)
    sigma_1 = CovarianceMatrix2D(sigma_11, sigma_12, rho_1)
    mu_2 = Vector2D(mu_21, mu_22)
    sigma_2 = CovarianceMatrix2D(sigma_21, sigma_22, rho_2)
    mu_, sigma_ = transform_component(mu_1, sigma_1, mu_2, sigma_2)
    final_prob  = probability_mass(mu_, sigma_, a=a, b=b)
    final_prob /= probability_mass(mu_1, sigma_1, a=a, b=b)
    final_prob /= probability_mass(mu_2, sigma_2, a=a, b=b)
    ## add part to compute loglikelihood here 
    final_prob *= compute_normal_pdf(mu_1.x1, mu_1.x2, sigma_1.s1, sigma_1.s2, sigma_1.rho,
                                     mu_2.x1, mu_2.x2, sigma_2.s1, sigma_2.s2, sigma_2.rho)
    return final_prob

@jit
def compute_likelihood_fixed_data_limits(mu_11, mu_12, mu_21, mu_22,
                       sigma_11, sigma_12, sigma_21, sigma_22,
                       rho_1, rho_2, a=[0.0, 0.0], b=[1.0, 1.0], a_data=[0.0, 0.0], b_data=[1.0, 1.0]):
    mu_1 = Vector2D(mu_11, mu_12)
    sigma_1 = CovarianceMatrix2D(sigma_11, sigma_12, rho_1)
    mu_2 = Vector2D(mu_21, mu_22)
    sigma_2 = CovarianceMatrix2D(sigma_21, sigma_22, rho_2)
    mu_, sigma_ = transform_component(mu_1, sigma_1, mu_2, sigma_2)
    final_prob  = probability_mass(mu_, sigma_, a=a, b=b)
    final_prob /= probability_mass(mu_1, sigma_1, a=a_data, b=b_data)
    final_prob /= probability_mass(mu_2, sigma_2, a=a, b=b)
    ## add part to compute loglikelihood here 
    final_prob *= compute_normal_pdf(mu_1.x1, mu_1.x2, sigma_1.s1, sigma_1.s2, sigma_1.rho,
                                     mu_2.x1, mu_2.x2, sigma_2.s1, sigma_2.s2, sigma_2.rho)
    return final_prob



@jit
def compute_normal_pdf_on_samples(x_1, x_2, mu_1, mu_2, sigma_1, sigma_2, rho):
    Z = jax.scipy.stats.multivariate_normal.pdf( 
                    jnp.stack([x_1, x_2], axis=-1),
                    jnp.stack([mu_1, mu_2], axis=-1),
                    jnp.stack([jnp.stack([sigma_1**2, sigma_1*sigma_2*rho]),jnp.stack([sigma_1*sigma_2*rho, sigma_2**2])], axis=-1))
    return Z


@jit
def compute_truncated_normal_pdf_on_samples(x_1, x_2, mu_1, mu_2, sigma_1, sigma_2, rho, a=[0.0, 0.0], b=[1.0, 1.0]):
    Z = jax.scipy.stats.multivariate_normal.pdf( 
                    jnp.stack([x_1, x_2], axis=-1),
                    jnp.stack([mu_1, mu_2], axis=-1),
                    jnp.stack([jnp.stack([sigma_1**2, sigma_1*sigma_2*rho]),jnp.stack([sigma_1*sigma_2*rho, sigma_2**2])], axis=-1))
    
    Z = Z / mvnorm2d(mu_1, mu_2, sigma_1, sigma_2, a[0], a[1], b[0], b[1], rho)
    return Z
    

class TruncatedGaussian2DAnalytic(AnalyticPopulationModel):
    def __init__(self, a=[0.0, 0.0], b=[1.0, 1.0], var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'rho_chi']):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.var_name_1 = var_names[0]
        self.mu_name_1 = hyper_var_names[0]
        self.sigma_name_1 = hyper_var_names[1]
        self.var_name_2 = var_names[1]
        self.mu_name_2 = hyper_var_names[2]
        self.sigma_name_2 = hyper_var_names[3]
        self.rho_name = hyper_var_names[4]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}
        
    def get_data(self, data, params, component):
        var_name = self.var_name_1 if (component == 1) else self.var_name_2
        mu_name = self.mu_name_1 if (component == 1) else self.mu_name_2
        sigma_name = self.sigma_name_1 if (component == 1) else self.sigma_name_2
        X_locations = data[var_name + '_mu_kernel'];
        X_scales    = data[var_name + '_sigma_kernel'];
        mu          = params[mu_name]
        sigma       = params[sigma_name]
        return X_locations, X_scales, mu, sigma
    
    def evaluate(self, data, params):
        x_1, x_2 = data[self.var_name_1], data[self.var_name_2]
        mu_1, mu_2 = params[self.mu_name_1], params[self.mu_name_2]
        sigma_1, sigma_2 = params[self.sigma_name_1], params[self.sigma_name_2]
        rho = params.get(self.rho_name, 1e-6)
        return compute_truncated_normal_pdf_on_samples(x_1, x_2, mu_1, mu_2, sigma_1, sigma_2, rho, a=self.a, b=self.b)
    
    def __call__(self, data, params):
        X_locations_1, X_scales_1, mu_1, sigma_1 = self.get_data(data, params, component=1);
        X_locations_2, X_scales_2, mu_2, sigma_2 = self.get_data(data, params, component=2);
        rho = params.get(self.rho_name, 1e-6)
        rho_kernel = data.get(self.var_name_1 + "_rho_kernel", 1e-6 * jnp.ones_like(data[self.var_name_1 + "_mu_kernel"]))
        return compute_likelihood(mu_11=X_locations_1, mu_12=X_locations_2, 
                                   mu_21=mu_1, mu_22=mu_2,
                                   sigma_11=X_scales_1, sigma_12=X_scales_2, 
                                   sigma_21=sigma_1, sigma_22=sigma_2,
                                   rho_1=rho_kernel, rho_2=rho, a=self.a, b=self.b)

    def sample(self, df_hyper_samples, oversample=1):
        return ppd_truncCorrelatedanalytic(self, df_hyper_samples, oversample=oversample)


class TruncatedGaussian2DAnalyticLimits(AnalyticPopulationModel):
    def __init__(self, a=[0.0, 0.0], b=[1.0, 1.0], var_names=['chi_1', 'chi_2'], 
                 hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'rho_chi',
                                  'chi_1_min', 'chi_1_max', 'chi_2_min', 'chi_2_max']):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.var_name_1 = var_names[0]
        self.mu_name_1 = hyper_var_names[0]
        self.sigma_name_1 = hyper_var_names[1]
        self.var_name_2 = var_names[1]
        self.mu_name_2 = hyper_var_names[2]
        self.sigma_name_2 = hyper_var_names[3]
        self.rho_name = hyper_var_names[4]
        self.a_name_0 = hyper_var_names[5]
        self.b_name_0 = hyper_var_names[6]
        self.a_name_1 = hyper_var_names[7]
        self.b_name_1 = hyper_var_names[8]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}
        
    def get_data(self, data, params, component):
        var_name = self.var_name_1 if (component == 1) else self.var_name_2
        mu_name = self.mu_name_1 if (component == 1) else self.mu_name_2
        sigma_name = self.sigma_name_1 if (component == 1) else self.sigma_name_2
        X_locations = data[var_name + '_mu_kernel'];
        X_scales    = data[var_name + '_sigma_kernel'];
        mu          = params[mu_name]
        sigma       = params[sigma_name]
        return X_locations, X_scales, mu, sigma
    
    def evaluate(self, data, params):
        x_1, x_2 = data[self.var_name_1], data[self.var_name_2]
        mu_1, mu_2 = params[self.mu_name_1], params[self.mu_name_2]
        sigma_1, sigma_2 = params[self.sigma_name_1], params[self.sigma_name_2]
        rho = params.get(self.rho_name, 0)
        a = [params.get(self.a_name_0, self.a[0]), params.get(self.a_name_1, self.a[1])];
        b = [params.get(self.b_name_0, self.b[0]), params.get(self.b_name_1, self.b[1])];
        return compute_truncated_normal_pdf_on_samples(x_1, x_2, mu_1, mu_2, sigma_1, sigma_2, rho, a=a, b=b)
    
    def __call__(self, data, params):
        X_locations_1, X_scales_1, mu_1, sigma_1 = self.get_data(data, params, component=1);
        X_locations_2, X_scales_2, mu_2, sigma_2 = self.get_data(data, params, component=2);
        rho = params.get(self.rho_name, 1e-6)
        rho_kernel = data.get(self.var_name_1 + "_rho_kernel", 1e-6 * jnp.ones_like(data[self.var_name_1 + "_mu_kernel"]))
        a = [params.get(self.a_name_0, self.a[0]), params.get(self.a_name_1, self.a[1])];
        b = [params.get(self.b_name_0, self.b[0]), params.get(self.b_name_1, self.b[1])];
        return compute_likelihood_fixed_data_limits(mu_11=X_locations_1, mu_12=X_locations_2, 
                                   mu_21=mu_1, mu_22=mu_2,
                                   sigma_11=X_scales_1, sigma_12=X_scales_2, 
                                   sigma_21=sigma_1, sigma_22=sigma_2,
                                   rho_1=rho_kernel, rho_2=rho, a=a, b=b, a_data=self.a, b_data=self.b)

    def sample(self, df_hyper_samples, oversample=1):
        return ppd_truncCorrelatedanalytic(self, df_hyper_samples, oversample=oversample)





