from dataclasses import dataclass, field
from .custompriors import DiracDelta

from typing import List, Dict, Any, Union, Tuple, Optional

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

import pandas as pd

import numpyro
import numpy as np

class PriorBlock:
    pass

class CovarianceMatrix(PriorBlock):
    def __init__(self, ndims, var_name, sigma_distribution, corr_cholesky_distribution):
        self.var_name = var_name
        self.ndims = ndims
        self.sigma_distribution = sigma_distribution
        self.corr_cholesky_distribution = corr_cholesky_distribution
        
    def sample(self, x):
        L = numpyro.sample(self.var_name + "_corr_", self.corr_cholesky_distribution)
        s = numpyro.sample(self.var_name + "_sig_", self.sigma_distribution)
        C = L * s[..., None]  # (..., d, d)
        x[self.var_name] = C @ jnp.swapaxes(C, -1, -2)
        numpyro.deterministic(self.var_name, x[self.var_name])

    def cleanup(self, samples):
        for i in range(self.ndims):
            for j in range(self.ndims):
                samples[self.var_name + f"_{i}_{j}"] = samples[self.var_name][...,i,j]

        del samples[self.var_name]
        del samples[self.var_name  + "_corr_"]
        del samples[self.var_name + "_sig_"]

class UniformVector(PriorBlock):
    def __init__(self, ndims, var_name, vector_distribution):
        self.var_name = var_name
        self.ndims = ndims
        self.vector_distribution = vector_distribution
        
    def sample(self, x):
        vecs = numpyro.sample(self.var_name, self.vector_distribution)
        x[self.var_name] = vecs

    def cleanup(self, samples):
        for i in range(self.ndims):
            samples[self.var_name + f"_{i}"] = samples[self.var_name][...,i]

        del samples[self.var_name]

class LowerTriangularUniform(PriorBlock):
    def __init__(self, var_names, limits=None):
        self.x1 = var_names[0];
        self.x2 = var_names[1];
        if limits is not None:
            self.limits = dict(zip(var_names, limits))
        else:
            self.limits = {k : (0,1) for k in var_names}
    
    def sample(self, x):
        x1 = numpyro.sample(self.x1, dist.Uniform(*self.limits[self.x1]))
        x2 = numpyro.sample(self.x2, dist.Uniform(self.limits[self.x2][0], x1))
        
        x[self.x1] = x1;
        x[self.x2] = x2;
        
        numpyro.factor('lowertr_jac', jnp.log(x1-self.limits[self.x1][0]))
        
class ConditionalPriorTriangular(PriorBlock):
    def __init__(self, var_names, limits=None):
        self.x1 = var_names[0];
        self.x2 = var_names[1];
        if limits is not None:
            self.limits = dict(zip(var_names, limits))
        else:
            self.limits = {k : (0,1) for k in var_names}
    
    def sample(self, x):
        x1 = numpyro.sample(self.x1, dist.Uniform(*self.limits[self.x1]))
        x2 = numpyro.sample(self.x2, dist.Uniform(self.limits[self.x2][0], x1))
        
        x[self.x1] = x1;
        x[self.x2] = x2;
        
        
import random
import string

def create_random_variable_name(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

class DirchletPrior(PriorBlock):
    def __init__(self, var_names, concentration=None, dirchlet_name=None):
        self.var_names = var_names
        if concentration is None:
            self.concentration = jnp.ones(len(var_names))
        else:
            self.concentration = jnp.array(concentration)
        self.dirchlet_name = dirchlet_name or "DirchletPriorBlock_"+create_random_variable_name()
        
    def sample(self, x):
        xs = numpyro.sample(self.dirchlet_name, dist.Dirichlet(self.concentration))
        for i in range(len(self.var_names)):
            x[self.var_names[i]] = xs[i]
    
    def cleanup(self, samples):
        for i in range(len(self.var_names)):
            samples[self.var_names[i]] = samples[self.dirchlet_name][..., i]
            
        del samples[self.dirchlet_name]
        

class DiracDelta:
    def __init__(self, value):
        self.value = value
    

@dataclass
class Sampler:
    priors : Dict[str, Union[dist.Distribution, PriorBlock]]
    latex_symbols : Dict[str, str]
    likelihood : Any = field(repr=False)
    constraints : List = field(default_factory=(lambda : []))
    num_samples : int = 2000
    num_warmup : int = 1000
    seed : Union[None, int] = None
    target_accept_prob : Union[None, float] = 0.7
    summary : bool = True
    return_samples : bool = False
    max_tree_depth : Union[None, int, Tuple] = 6
    just_prior : bool = False
    N_eff : bool = False
    #validation : bool = True
    
    def __post_init__(self):
        self.x = {}
        self._samples = None
        self.samples = None
        if self.latex_symbols is None:
            self.latex_symbols = {k : k for k in self.priors.keys()}
        if self.constraints is None:
            self.constraints = []
    
    def model(self):
        for var,dist in self.priors.items():
            if type(dist) == DiracDelta:
                self.x[var] = dist.value
            elif isinstance(dist, PriorBlock):
                dist.sample(self.x)
            else:
                self.x[var] = numpyro.sample(var, dist)

        if len(self.constraints) != 0:
            for i in range(len(self.constraints)):
                numpyro.factor('user_defined_constraint_' + str(i), self.constraints[i].logpdf(self.x))


        if self.N_eff:
            numpyro.deterministic('Selection N_eff', self.likelihood.compute_selection_N_eff_only(self.x))
            numpyro.deterministic('Min Event N_eff', jnp.min(self.likelihood.compute_event_N_eff_only(self.x)))


        if self.just_prior:
            return None

        numpyro.factor('logP', self.likelihood.logpdf(self.x))
        
    def sample(self):
        # Start from this source of randomness. We will split keys for subsequent operations.
        if self.seed is None:
            self.seed = np.random.randint(2*30)
        rng_key = jax.random.PRNGKey(self.seed)
        rng_key, rng_key_ = jax.random.split(rng_key)

        #numpyro.enable_validation(True)

        # Run NUTS.
        kernel = NUTS(self.model, target_accept_prob=self.target_accept_prob, max_tree_depth=self.max_tree_depth)
        num_samples = self.num_samples
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=num_samples, jit_model_args=True)
        mcmc.run(rng_key_)
        if self.summary:
            mcmc.print_summary()
        self._samples = mcmc.get_samples()
        
        ## Clean up the multidimenional samples returned using
        ## clean up methods in particular prior blocks
        for p in self.priors.values():
            cleanup_func = getattr(p, "cleanup", None)
            if cleanup_func is not None:
                cleanup_func(self._samples)
        
        self.samples = pd.DataFrame(self._samples)
        #if self.latex_symbols:
        #    self.samples.rename(self.latex_symbols, inplace=True, axis=1)
        if self.return_samples:
            return self.samples
        
    def corner(self, color='k', truth=None, truth_color='b'):
        import corner
        figure = corner.corner(self.samples.values,
                                labels=[self.latex_symbols.get(col, col) for col in self.samples.columns],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                color=color,
                                title_kwargs={"fontsize": 12})

        if truth is not None:
            corner.overplot_points(figure, truth[None], marker="s", color=truth_color)
        return figure
        
    def corner_on_fig(self, fig, color='k', truth=None, truth_color='b'):
        import corner
        figure = corner.corner(self.samples.values,
                                labels=[self.latex_symbols.get(col, col) for col in self.samples.columns],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                color=color,
                                title_kwargs={"fontsize": 12}, 
                                fig=fig)
        if truth is not None:
            corner.overplot_points(figure, truth[None], marker="s", color=truth_color)
        return figure