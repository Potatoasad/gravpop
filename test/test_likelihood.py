import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import numpyro.distributions as dist
from typing import Dict
import jax
from gravpop import * 
import matplotlib.pyplot as plt
import scipy
import pytest


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

	HL = HybridPopulationLikelihood(sampled_models=[TG], event_data=event_data, analytic_models=[])



	Samp = Sampler(priors = {'mu' : dist.Normal(0,3), 'sigma' : dist.Uniform(0,4)}, 
				   latex_symbols = {'mu' : r'$\mu$','sigma' : r'$\sigma$'} , 
				   likelihood=HL, num_samples=2000, num_warmup=1000)

	Samp.sample();

	fig = Samp.corner(truth=np.array([mu,sigma]));
	fig.savefig("./test/test_likelihood_plot.png")


def test_constructed_full_likelihood():
	filename = 'test/testdata/test_event_data.h5'
	selection_filename = 'test/testdata/test_selection_function.h5'

	SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name="mass_1_source")
	R = PowerLawRedshift(z_max=1.9)

	HL = PopulationLikelihood.from_file(
	            event_data_filename = filename,
	            selection_data_filename = selection_filename,
	            models = [SM,R]
	            )

	Lambda_0 = dict(alpha = 3.5,lam = 0.04,mmin = 5,mmax = 96,beta = 1.1,mpp = 35,sigpp = 4,delta_m = 3,lamb = 2.9)
	HL.logpdf(Lambda_0)

	Lambda_0 = dict(alpha = [3.5],lam = [0.04],mmin = [5],mmax = [96],beta = [1.1],mpp = [35],sigpp = [4],delta_m = [3],lamb = [2.9])
	Lambda_0 = {key: jnp.array(value) for key, value in Lambda_0.items()}
	HL.selection_data.calculate_N_eff_chunked(HL, Lambda_0, chunk=100)
		

@pytest.mark.slow
def test_posterior_pulling_data():
	filename = 'test/testdata/test_event_data.h5'
	selection_filename = 'test/testdata/test_selection_function.h5'
	posterior_file = "/Users/asadh/Documents/GitHub/gravpop_pipe/dev/samples_output.csv"

	SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name="mass_1_source")
	R = PowerLawRedshift(z_max=1.9)

	HP = HyperPosterior.from_file(posterior_file, filename, selection_filename, models=[SM,R])
	#HP.calculate_rates()
	HP.N_effective_cuts()
	HP.posterior_with_cuts.to_csv("/Users/asadh/Documents/GitHub/gravpop_pipe/dev/samples_output_w_cuts.csv",index=False)





