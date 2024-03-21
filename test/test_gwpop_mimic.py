from gravpop import *
import gravpop
from gwpopulation.models.redshift import PowerLawRedshift
from bilby.core.result import read_in_result
from scipy.interpolate import interp1d
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import trange
from matplotlib.gridspec import GridSpec

PP_path = '/Users/asadh/Documents/Data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json'
PP_result = read_in_result(PP_path)

PP_hyperposterior_samples = PP_result.posterior.copy().sample(100) # making a copy is best practice here so you don't accidentally modify things in-place
PP_hyperposterior_samples = BETA_CONVERTER.convert_parameters(PP_hyperposterior_samples, remove=False)


class TestAgainstGWPop:
	def setup_method(self, method):
		models = dict(
						mass="SmoothedMassDistribution",#"two_component_primary_mass_ratio",
						mag="iid_spin_magnitude",
						tilt="iid_spin_orientation",
						redshift="gwpopulation.models.redshift.PowerLawRedshift",
					)
		gwpop = GWPopLoader(posterior_file = "/Users/asadh/Documents/Data/posteriors.pkl",
							prior_file = "/Users/asadh/Documents/Data/production.prior",
							vt_file = "/Users/asadh/Downloads/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5",
							enforce_minimum_neffective_per_event = "injection_resampling_vt",
							models = models, vt_models=models,
							max_redshift = 2.3, samples_per_posterior=5000)
		gwpop.load_prior();
		gwpop.load_model();
		gwpop.load_vt();
		gwpop.load_posteriors();
		gwpop.create_likelihood();
		self.gwpop = gwpop

		SM = SmoothedTwoComponentPrimaryMassRatio(primary_mass_name='mass_1')
		R = gravpop.PowerLawRedshift(z_max=2.3)
		S_mag = BetaSpinMagnitudeIID(var_names = ['a_1', 'a_2'])
		S_tilt = GaussianIsotropicSpinOrientationsIID(var_names = ['cos_tilt_1', 'cos_tilt_2'])


		event_data = gwpop.likelihood.data.copy()
		event_data['prior'] = gwpop.likelihood.sampling_prior
		selection_data = {k:v for k,v in gwpop.selection.data.items()}

		self.gravpop_likelihood = PopulationLikelihood(models=[SM,R,S_mag,S_tilt], 
													   event_data=event_data,
													   selection_data=selection_data,
													   analysis_time=selection_data['analysis_time'],
													   total_generated=selection_data['total_generated'])

		self.HP = HyperPosterior(posterior = PP_hyperposterior_samples,
								 likelihood = self.gravpop_likelihood,
								 mass_model = SM,
								 redshift_model = R,
								 rate = False)

		self.HP.posterior = self.HP.posterior.rename({'min_event_n_effective' : 'min_event_n_effective_lvk',
													  'pdet_n_effective' : 'pdet_n_effective_gwpop_lvk'}, axis=1)

		self.HP.calculate_rates()
		self.HP.N_effective_cuts()

	def get_model_like(self, model_list, target_name):
		names = [obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__ for obj in model_list]
		return [model for name, model in zip(names, model_list) if target_name in name.lower()][0]

	def evaluate_gwpop_mass_model_at(self, param):
		#model = self.gwpop.models['mass']
		#model = [x for x in self.gwpop.likelihood.hyper_prior.models if "mass" in x.__class__.__name__.lower()][0]
		model = self.get_model_like(self.gwpop.likelihood.hyper_prior.models, "mass")
		self.gwpop.likelihood.hyper_prior.parameters.update(param)
		params = self.gwpop.likelihood.hyper_prior._get_function_parameters(model)
		return model(self.gwpop.likelihood.data, **params)

	def evaluate_gravpop_mass_model_at(self, param):
		model = self.gravpop_likelihood.models[0]
		return model(self.gravpop_likelihood.event_data, param)

	def evaluate_gwpop_redshift_model_at(self, param):
		#model = self.gwpop.models['redshift']
		#model = [x for x in self.gwpop.likelihood.hyper_prior.models if "redshift" in x.__class__.__name__.lower()][0]
		model = self.get_model_like(self.gwpop.likelihood.hyper_prior.models, "redshift")
		self.gwpop.likelihood.hyper_prior.parameters.update(param)
		params = self.gwpop.likelihood.hyper_prior._get_function_parameters(model)
		return model(self.gwpop.likelihood.data, **params)

	def evaluate_gravpop_spin_orientation_model_at(self, param):
		model = self.gravpop_likelihood.models[3]
		return model(self.gravpop_likelihood.event_data, param)

	def evaluate_gwpop_spin_orientation_model_at(self, param):
		#model = self.gwpop.models['redshift']
		#model = [x for x in self.gwpop.likelihood.hyper_prior.models if "spin_orientation" in x.__class__.__name__.lower()][0]
		model = self.get_model_like(self.gwpop.likelihood.hyper_prior.models, "spin_orientation")
		self.gwpop.likelihood.hyper_prior.parameters.update(param)
		params = self.gwpop.likelihood.hyper_prior._get_function_parameters(model)
		return model(self.gwpop.likelihood.data, **params)

	def evaluate_gravpop_spin_magnitude_model_at(self, param):
		model = self.gravpop_likelihood.models[2]
		return model(self.gravpop_likelihood.event_data, param)

	def evaluate_gwpop_spin_magnitude_model_at(self, param):
		#model = self.gwpop.models['redshift']
		#model = [x for x in self.gwpop.likelihood.hyper_prior.models if "spin_magnitude" in x.__class__.__name__.lower()][0]
		model = self.get_model_like(self.gwpop.likelihood.hyper_prior.models, "spin_magnitude")
		self.gwpop.likelihood.hyper_prior.parameters.update(param)
		params = self.gwpop.likelihood.hyper_prior._get_function_parameters(model)
		return model(self.gwpop.likelihood.data, **params)

	def evaluate_gravpop_redshift_model_at(self, param):
		model = self.gravpop_likelihood.models[1]
		return model(self.gravpop_likelihood.event_data, param)

	def test_compare_redshift(self):
		cols = ['alpha','beta','mmax','mmin','lam','mpp','sigpp','delta_m','lamb']
		param = PP_hyperposterior_samples[cols].iloc[0,:].to_dict()

		Z_gwpop = jnp.nan_to_num(self.evaluate_gwpop_redshift_model_at(param), posinf=0, neginf=0, nan=0)
		Z_gravpop = jnp.nan_to_num(self.evaluate_gravpop_redshift_model_at(param), posinf=0, neginf=0, nan=0)

		rel = jnp.abs((Z_gravpop))/(jnp.abs(Z_gwpop) + 1e-16)

		fig = plt.figure(layout="constrained")
		gs = GridSpec(1, 1, figure=fig)
		ax3 = fig.add_subplot(gs[0, 0])
		ax3.hist(rel.flatten(), bins=100);
		ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
		plt.suptitle("Probability comparison for redshift magnitude models")
		fig.savefig("./test/images/data_redshift_magnitude_model_comparison.png")
		assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs


	def test_compare_redshift(self):
		cols = ['alpha','beta','mmax','mmin','lam','mpp','sigpp','delta_m','lamb']
		param = PP_hyperposterior_samples[cols].iloc[0,:].to_dict()

		Z_gwpop = jnp.nan_to_num(self.evaluate_gwpop_redshift_model_at(param), posinf=0, neginf=0, nan=0)
		Z_gravpop = jnp.nan_to_num(self.evaluate_gravpop_redshift_model_at(param), posinf=0, neginf=0, nan=0)

		rel = jnp.abs((Z_gravpop))/(jnp.abs(Z_gwpop) + 1e-16)

		fig = plt.figure(layout="constrained")
		gs = GridSpec(1, 1, figure=fig)
		ax3 = fig.add_subplot(gs[0, 0])
		ax3.hist(rel.flatten(), bins=100);
		ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
		plt.suptitle("Probability comparison for redshift magnitude models")
		fig.savefig("./test/images/data_redshift_magnitude_model_comparison.png")
		assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs

	def test_compare_spin_orientation(self):
		cols = ['alpha','beta','mmax','mmin','lam','mpp','sigpp','delta_m','lamb',
		'mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin', 'lamb', 'amax','alpha_chi', 'beta_chi']
		param = PP_hyperposterior_samples[cols].iloc[0,:].to_dict()

		Z_gwpop = jnp.nan_to_num(self.evaluate_gwpop_spin_orientation_model_at(param), posinf = 0, neginf = 0, nan=0)
		Z_gravpop = jnp.nan_to_num(self.evaluate_gravpop_spin_orientation_model_at(param), posinf = 0, neginf = 0, nan=0)

		#rel = jnp.abs((Z_gwpop - Z_gravpop))/(jnp.abs(Z_gwpop) + 1e-16)

		rel = jnp.abs((Z_gravpop))/(jnp.abs(Z_gwpop) + 1e-16)

		fig = plt.figure(layout="constrained")
		gs = GridSpec(1, 1, figure=fig)
		ax3 = fig.add_subplot(gs[0, 0])
		ax3.hist(rel.flatten(), bins=100);
		ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
		plt.suptitle("Probability comparison for spin orientation models")
		fig.savefig("./test/images/data_spin_orientation_model_comparison.png")
		#assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs
		#assert rel.max() < 1e-2

	def test_compare_spin_magnitude(self):
		cols = ['alpha','beta','mmax','mmin','lam','mpp','sigpp','delta_m','lamb',
		'mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin', 'lamb', 'amax','alpha_chi', 'beta_chi']
		param = PP_hyperposterior_samples[cols].iloc[0,:].to_dict()

		Z_gwpop = jnp.nan_to_num(self.evaluate_gwpop_spin_magnitude_model_at(param), posinf = 0, neginf = 0, nan=0)
		Z_gravpop = jnp.nan_to_num(self.evaluate_gravpop_spin_magnitude_model_at(param), posinf = 0, neginf = 0, nan=0)

		#rel = jnp.abs((Z_gwpop - Z_gravpop))/(jnp.abs(Z_gwpop) + 1e-16)

		rel = jnp.abs((Z_gravpop))/(jnp.abs(Z_gwpop) + 1e-16)

		fig = plt.figure(layout="constrained")
		gs = GridSpec(1, 1, figure=fig)
		ax3 = fig.add_subplot(gs[0, 0])
		ax3.hist(rel.flatten(), bins=100);
		ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
		plt.suptitle("Probability comparison for spin magnitude models")
		fig.savefig("./test/images/data_spin_magnitude_model_comparison.png")
		#assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs
		#assert rel.max() < 1e-2

	def test_event_N_effective(self):
		cols = ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m',
				'mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin', 'lamb', 'amax','alpha_chi', 'beta_chi']

		param = self.HP.posterior[cols].iloc[0,:].to_dict()

		self.gwpop.likelihood.parameters.update(param)

		col_new = 'min_event_n_effective_gwpop'
		new_data = {col_new : np.zeros_like(self.HP.posterior['min_event_n_effective'].values)}
		for i in trange(len(self.HP.posterior)):
			param = self.HP.posterior[cols].iloc[i,:].to_dict()
			self.gwpop.likelihood.parameters.update(param)
			new_data[col_new][i] = jnp.min(self.gwpop.likelihood.per_event_bayes_factors_and_n_effective_and_variances()[1])
			
			
		self.HP.posterior.loc[:, col_new] = new_data[col_new];

		delta_N_eff = self.HP.posterior['min_event_n_effective_gwpop'] - self.HP.posterior['min_event_n_effective']

		fig = plt.figure(layout="constrained")
		gs = GridSpec(1, 1, figure=fig)
		ax3 = fig.add_subplot(gs[0, 0])
		ax3.hist(delta_N_eff.values, bins=10);
		ax3.set_title(r"Histogram of $N^{gravpop}_{eff}/N^{gwpop}_{eff}$")
		plt.suptitle(r"$N_{eff}$ comparison")
		fig.savefig("./test/images/data_event_N_eff_magnitude_model_comparison.png")

		#assert (().abs() < 5).all()

	def test_selection_N_effective(self):
		cols = ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m',
				'mu_chi', 'sigma_chi', 'xi_spin', 'sigma_spin', 'lamb', 'amax','alpha_chi', 'beta_chi']

		param = self.HP.posterior[cols].iloc[0,:].to_dict()

		self.gwpop.likelihood.parameters.update(param)

		col_selection_new = 'pdet_n_effective_gwpop'
		new_data = {col_selection_new : np.zeros_like(self.HP.posterior['pdet_n_effective'].values)}
		for i in trange(len(self.HP.posterior)):
			param = self.HP.posterior[cols].iloc[i,:].to_dict()
			self.gwpop.likelihood.parameters.update(param)
			mu, var = self.gwpop.selection.detection_efficiency(param)
			N_eff_sel = mu**2 / var
			new_data[col_selection_new][i] = N_eff_sel

		self.HP.posterior.loc[:, col_selection_new] = new_data[col_selection_new];
		
		delta_N_eff = self.HP.posterior['pdet_n_effective_gwpop'] - self.HP.posterior['pdet_n_effective']

		#assert ((self.HP.posterior['min_event_n_effective_gwpop'] - self.HP.posterior['min_event_n_effective']).abs() < 5).all()
		#assert ((self.HP.posterior['pdet_n_effective_gwpop'] - self.HP.posterior['pdet_n_effective']).abs() < 5).all()
		
		fig = plt.figure(layout="constrained")
		gs = GridSpec(1, 1, figure=fig)
		ax3 = fig.add_subplot(gs[0, 0])
		ax3.hist(delta_N_eff.values, bins=10);
		ax3.set_title(r"Histogram of Selection $N^{gwpop}_{eff} - N^{gravpop}_{eff}$")
		plt.suptitle(r"Selection $N_{eff}$ comparison")
		fig.savefig("./test/images/data_selection_N_eff_magnitude_model_comparison.png")

	def teardown_method(self, method):
		pass

	def test_mass_model_matches_gwpop(self):
		pass
