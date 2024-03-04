import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import numpyro.distributions as dist
from typing import Dict
import jax
from gravpop import * 
import matplotlib.pyplot as plt

def test_sampler():

    @dataclass
    class TestLikelihood:
        data: Dict[str, jax.Array]
        var_name : str = 'x'
        mu_name : str = 'mu'
        sigma_name : str = 'sigma'
        
        def logpdf(self, x):
            mu = x['mu']
            sigma = x['sigma']
            return jnp.sum( (-0.5 * (self.data[self.var_name] - mu)**2 / sigma**2)  - jnp.log(sigma) )

    like = TestLikelihood({'x' : jnp.array(np.random.randn(1000))})

    Samp = Sampler(
    priors = {'mu' : dist.Uniform(-3,3),  
              'sigma' : dist.Uniform(0,4)}, 
    latex_symbols = {'mu' : r'$\mu$', 
                     'sigma' : r'$\sigma$'} , 
    likelihood=like)

    Samp.sample()

    fig1 = Samp.corner(color='r', truth=jnp.array([0,1]));
    Samp.corner_on_fig(fig1, color='k', truth=jnp.array([0,1]));
    plt.savefig("./test/test_sampler_plot.png")
