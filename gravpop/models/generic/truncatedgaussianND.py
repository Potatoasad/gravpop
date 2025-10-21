from ..utils import *
from .abstract import *
from ..sample import *
from .truncatedgaussian2D import compute_likelihood

import jax
import jax.numpy as jnp
from typing import Union, List, Dict
from dataclasses import dataclass, field
from functools import partial

key = jax.random.key(0)

@partial(jnp.vectorize, signature='(d),(d,d),(d),(d),()->()')
def mvn_vec(mu, Sigma, a, b, key):
    return mvn_box_prob(mu, Sigma, a, b, key=key)

def compute_truncated_mv_normal_overlap(mu_data, Sigma_data, a_data, b_data, mu_pop, Sigma_pop, a_pop, b_pop, key=key):
    C_data = mvn_vec(mu_data, Sigma_data, a_data, b_data, key);
    C_pop = mvn_vec(mu_pop, Sigma_pop, a_pop, b_pop, key);

    a_3 = jnp.maximum(a_data, a_pop);
    b_3 = jnp.minimum(b_data, b_pop);

    Lambda_data = jnp.linalg.inv(Sigma_data);
    Lambda_pop = jnp.linalg.inv(Sigma_pop);
    
    Sigma_3 = jnp.linalg.inv(Lambda_data + Lambda_pop);
    #print(Sigma_3.shape, Lambda_data.shape, mu_data.shape, Lambda_pop.shape, mu_pop.shape)
    pop_mul = jnp.einsum("...ij,...j->...i", Lambda_pop, mu_pop)
    data_mul = jnp.einsum("...ij,...j->...i", Lambda_data, mu_data)
    #print(data_mul.shape, pop_mul.shape)
    #pop_plus_data_mul = jnp.einsum("...j,j->...j", data_mul, pop_mul)
    pop_plus_data_mul = data_mul + pop_mul
    #print(Sigma_3.shape, pop_plus_data_mul.shape)
    mu_3 = jnp.einsum("...ij,...j->...i", Sigma_3, pop_plus_data_mul) #Sigma_3 @ ( Lambda_data @ mu_data + Lambda_pop @ mu_pop );

    #print(mu_3.shape, Sigma_3.shape)
    C_3 = mvn_vec(mu_3, Sigma_3, a_3, b_3, key);

    phi = jnp.exp(jax.scipy.stats.multivariate_normal.logpdf(mu_data, mu_pop, Sigma_data + Sigma_pop))

    return phi * C_3 / (C_data * C_pop)


from gravpop import *


class TruncatedGaussianNDAnalytic(AnalyticPopulationModel):
    def __init__(self, 
                 a=[0.0, 0.0, 0.0], 
                 b=[1.0, 1.0, 1.0], 
                 var_names=['chi'], 
                 hyper_var_names=['mu_chi_var', 'Sigma_chi_var']):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.key = jax.random.key(0)
        
    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        var_name = self.var_names[0];
        mu_data, Sigma_data =  data[f'{var_name}_mu_kernel'], data[f'{var_name}_sigma_kernel']
        mu_pop, Sigma_pop = params[self.hyper_var_names[0]], params[self.hyper_var_names[1]];
        
        probss = compute_truncated_mv_normal_overlap(
            mu_data, Sigma_data, self.a, self.b,
            mu_pop, Sigma_pop, self.a, self.b,
            key=self.key
        )
        return probss

class TruncatedGaussian2DAnalyticFull(AnalyticPopulationModel):
    def __init__(self, 
                 a=[0.0, 0.0], 
                 b=[1.0, 1.0], 
                 var_names=['chi'], 
                 hyper_var_names=['mu_chi_var', 'Sigma_chi_var']):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.key = jax.random.key(0)
        
    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        var_name = self.var_names[0];
        mu_data, Sigma_data =  data[f'{var_name}_mu_kernel'], data[f'{var_name}_sigma_kernel']
        mu_pop, Sigma_pop = params[self.hyper_var_names[0]], params[self.hyper_var_names[1]];

        s0_data = jnp.sqrt(Sigma_data[...,0,0])
        s1_data = jnp.sqrt(Sigma_data[...,1,1])
        rho_data = Sigma_data[...,0,1] /(s0_data * s1_data) 

        s0_pop = jnp.sqrt(Sigma_pop[...,0,0])
        s1_pop = jnp.sqrt(Sigma_pop[...,1,1])
        rho_pop = Sigma_pop[...,0,1] /(s0_pop * s1_pop) 
        
        probss = compute_likelihood(mu_11=mu_data[...,0], mu_12=mu_data[...,1], 
                                   mu_21=mu_pop[...,0], mu_22=mu_pop[...,1],
                                   sigma_11=s0_data, sigma_12=s1_data, 
                                   sigma_21=s0_pop, sigma_22=s1_pop,
                                   rho_1=rho_data, rho_2=rho_pop, a=self.a, b=self.b)
        return probss