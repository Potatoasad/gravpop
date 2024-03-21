from dataclasses import dataclass, field

from typing import List, Dict, Any, Union, Tuple

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

import corner

import numpyro
import numpy as np

@dataclass
class Sampler:
    priors : Dict[str, dist.Distribution]
    latex_symbols : Dict[str, str]
    likelihood : Any = field(repr=False)
    num_samples : int = 2000
    num_warmup : int = 1000
    seed : Union[None, int] = None
    target_accept_prob : Union[None, float] = 0.7
    summary : bool = True
    return_samples : bool = False
    max_tree_depth : Union[None, int, Tuple] = 6
    just_prior : bool = False
    #validation : bool = True
    
    def __post_init__(self):
        self.x = {}
        self._samples = None
        self.samples = None
    
    def model(self):
        for var,dist in self.priors.items():
            self.x[var] = numpyro.sample(var, dist)

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
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=num_samples)
        mcmc.run(rng_key_)
        if self.summary:
            mcmc.print_summary()
        self._samples = mcmc.get_samples()
        
        self.samples = pd.DataFrame(self._samples)
        #if self.latex_symbols:
        #    self.samples.rename(self.latex_symbols, inplace=True, axis=1)
        if self.return_samples:
            return self.samples
        
    def corner(self, color='k', truth=None, truth_color='b'):
        figure = corner.corner(self.samples.values,
                                labels=[self.latex_symbols[col] for col in self.samples.columns],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                color=color,
                                title_kwargs={"fontsize": 12})

        if truth is not None:
            corner.overplot_points(figure, truth[None], marker="s", color=truth_color)
        return figure
        
    def corner_on_fig(self, fig, color='k', truth=None, truth_color='b'):
        figure = corner.corner(self.samples.values,
                                labels=[self.latex_symbols[col] for col in self.samples.columns],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                color=color,
                                title_kwargs={"fontsize": 12}, 
                                fig=fig)
        if truth is not None:
            corner.overplot_points(figure, truth[None], marker="s", color=truth_color)
        return figure