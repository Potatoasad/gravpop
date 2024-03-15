import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import numpyro.distributions as dist
from typing import Dict
import jax
from gravpop import * 
import matplotlib.pyplot as plt
import scipy


def test_likelihood_sampled():
		TG = Gaussian1D(var_name='x', mu_name='mu', sigma_name='sigma')
		mu = 0.0
		sigma = 0.2
		likelihood_sigma = 1.0

		# Number of Events, Kernels & Samples
		E, K, N = 10, 1, 1000  

		events = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=E)
		x_data = jnp.stack([scipy.stats.norm.rvs(loc=x, scale=likelihood_sigma, size=N) for x in events]).reshape(E,K,N)


		event_data = {'x': x_data, # 10 events, 1 kernels, 1000 points each
									'weights': jnp.ones((E,K))/K}

		HL = HybridPopulationLikelihood(sampled_models=[TG], event_data=event_data, analytic_models=[], selection_data={})



		Samp = Sampler(priors = {'mu' : dist.Normal(0,3),  
															'sigma' : dist.Uniform(0,4)}, 
										latex_symbols = {'mu' : r'$\mu$', 
																		 'sigma' : r'$\sigma$'} , 
										likelihood=HL, num_samples=2000, num_warmup=1000)

		Samp.sample();

		fig = Samp.corner(truth=np.array([mu,sigma]));
		fig.savefig("./test/test_likelihood_plot.png")


def test_constructed_full_likelihood():
		from gravpop_pipe import *




