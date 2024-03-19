from gravpop import *
import gravpop
from gwpopulation.models.redshift import PowerLawRedshift
from bilby.core.result import read_in_result
from scipy.interpolate import interp1d
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

PP_path = '/Users/asadh/Documents/Data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json'
PP_result = read_in_result(PP_path)

PP_hyperposterior_samples = PP_result.posterior.copy() # making a copy is best practice here so you don't accidentally modify things in-place
PP_hyperposterior_samples


class TestAgainstGWPop:

	def setup_method(self, method):
		gwpop = GWPopLoader(posterior_file = "/Users/asadh/Documents/Data/posteriors.pkl",
							prior_file = "/Users/asadh/Documents/Data/production.prior",
							vt_file = "/Users/asadh/Downloads/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5")
		gwpop.load_prior();
		gwpop.load_model();
		gwpop.load_vt();
		gwpop.load_posteriors();
		gwpop.create_likelihood();
		self.gwpop = gwpop

		SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name='mass_1')
		R = gravpop.PowerLawRedshift(z_max=1.9)

		event_data = gwpop.likelihood.data.copy()
		event_data['prior'] = gwpop.likelihood.sampling_prior
		selection_data = {k:v for k,v in gwpop.selection.data.items()}

		self.gravpop_likelihood = PopulationLikelihood(models=[SM,R], 
													   event_data=event_data,
													   selection_data=selection_data,
													   analysis_time=selection_data['analysis_time'],
													   total_generated=selection_data['total_generated'])
		#loglikes = HL.sampled_compute_log_weights(selection_data, param2)
		#HL.compute_selection_N_eff(loglikes, N=HL.selection_data.total_generated)[0]
		#HL.compute_selection_N_eff(loglikes)

	def evaluate_gwpop_mass_model_at(self, param):
		model = self.gwpop.likelihood.hyper_prior.models[0]
		self.gwpop.likelihood.hyper_prior.parameters.update(param)
		params = self.gwpop.likelihood.hyper_prior._get_function_parameters(model)
		return jnp.log(model(self.gwpop.likelihood.data, **params))

	def evaluate_gravpop_mass_model_at(self, param):
		model = self.gravpop_likelihood.models[0]
		return jnp.log(model(self.gravpop_likelihood.event_data, param))

	def evaluate_gwpop_redshift_model_at(self, param):
		model = self.gwpop.likelihood.hyper_prior.models[1]
		self.gwpop.likelihood.hyper_prior.parameters.update(param)
		params = self.gwpop.likelihood.hyper_prior._get_function_parameters(model)
		return jnp.log(model(self.gwpop.likelihood.data, **params))

	def evaluate_gravpop_redshift_model_at(self, param):
		model = self.gravpop_likelihood.models[1]
		return jnp.log(model(self.gravpop_likelihood.event_data, param))

	def test_compare_redshift(self):
		cols = ['alpha','beta','mmax','mmin','lam','mpp','sigpp','delta_m','lamb']
		param = PP_hyperposterior_samples[cols].iloc[0,:].to_dict()

		gwpop_red = jnp.nan_to_num(self.evaluate_gwpop_redshift_model_at(param))
		gravpop_red = jnp.nan_to_num(self.evaluate_gravpop_redshift_model_at(param))

		rel = jnp.abs((gwpop_red - gravpop_red))/(jnp.abs(gwpop_red) + 1e-16)
		#assert rel.max() < 1e-2

	def test_compare_mass(self):
		cols = ['alpha','beta','mmax','mmin','lam','mpp','sigpp','delta_m','lamb']
		param = PP_hyperposterior_samples[cols].iloc[0,:].to_dict()

		gwpop_mass = jnp.nan_to_num(self.evaluate_gwpop_mass_model_at(param), posinf = 0, neginf = 0, nan=0)
		gravpop_mass = jnp.nan_to_num(self.evaluate_gravpop_mass_model_at(param), posinf = 0, neginf = 0, nan=0)

		rel = jnp.abs((gwpop_mass - gravpop_mass))/(jnp.abs(gwpop_mass) + 1e-16)
		#assert rel.max() < 1e-2

	def teardown_method(self, method):
		pass

	def test_mass_model_matches_gwpop(self):
		pass
