from dataclasses import dataclass
import pandas as pd
from typing import Optional, List, Union, Dict, Any
import jax
import jax.numpy as jnp
from ..plots import *
from ..models import *
from ..hyper import *
from ..utils import *

BETA_CONVERTER = BetaDistributionConverter()

MASS_MODELS = [SmoothedTwoComponentPrimaryMassRatio]

@dataclass
class HyperPosterior:
	posterior : Optional[Union[pd.DataFrame, Dict[str, jax.Array]]]
	likelihood : Any
	mass_model : Optional[AbstractPopulationModel] = None
	redshift_model : Optional[AbstractPopulationModel] = None
	rate : bool = False

	def __post_init__(self):
		#print("started making posterior")
		if self.mass_model is None:
			possible_models = [model for model in self.likelihood.models if any([isinstance(model, test_model) for test_model in MASS_MODELS])]
			if len(possible_models) != 0:
				self.mass_model = possible_models[0]

		if self.redshift_model is None:
			possible_models = [model for model in self.likelihood.models if isinstance(model, Redshift)]
			if len(possible_models) != 0:
				self.redshift_model = possible_models[0]

		if isinstance(self.posterior, pd.DataFrame):
			self.posterior_dict = {col : jnp.array(self.posterior[col].values) for col in self.posterior.columns}
		elif isinstance(self.posterior, Dict):
			self.posterior_dict = self.posterior
			self.posterior = pd.DataFrame(self.posterior)
		else:
			raise ValueError("posterior should be either a pandas dataframe or a dictonary of arrays")

		#print("making plots")
		self.mass_plot = MassPlot(self.posterior_dict, model=self.mass_model)
		self.redshift_plot = RedshiftPlot(self.posterior_dict, model=self.redshift_model)
		#print("finished making posterior")

		self.posterior_dict = BETA_CONVERTER.convert_parameters(self.posterior_dict, remove=False)
		self.posterior = BETA_CONVERTER.convert_parameters(self.posterior, remove=False)
		

	@classmethod
	def from_file(cls, posterior_sample_file, event_data_file, selection_file, models, rate=False, LikelihoodClass=PopulationLikelihood, SelectionClass=SelectionFunction, **kwargs):
		posterior_file_extension = posterior_sample_file.split(".")[-1]
		#print("reading...")
		if posterior_file_extension == "csv":
			## Assume it can be read by pandas
			posterior = pd.read_csv(posterior_sample_file)

		#print("creating likelihood...")
		likelihood = LikelihoodClass.from_file(
		            event_data_filename = event_data_file,
		            selection_data_filename = selection_file,
		            models = models,
		            **kwargs
		            )

		return cls(posterior, likelihood, rate=rate)

	def calculate_selection_N_eff(self, chunk=100):
		N_eff = self.likelihood.selection_data.calculate_N_eff_chunked(self.likelihood, self.posterior_dict, chunk=chunk)
		self.posterior_dict['pdet_n_effective'] = N_eff
		self.posterior.loc[:, 'pdet_n_effective'] = N_eff

	def calculate_event_N_eff_chunked(self, params, chunk=100):
		N_eff_func = lambda params : self.likelihood.compute_event_N_eff_only(params)
		axes_dict_shape = {key:0 for key in params.keys()}
		N_eff_chunked_func = chunked_vmap(N_eff_func, in_axes=(axes_dict_shape,), chunk=chunk, progress_note="Calculating Event N_effective ...")

		N_effs = N_eff_chunked_func(params)

		return N_effs

	def calculate_N_eff_event(self, chunk=100):
		N_eff = self.calculate_event_N_eff_chunked(self.posterior_dict, chunk=chunk)
		self.posterior_dict['min_event_n_effective'] = N_eff
		self.posterior.loc[:, 'min_event_n_effective'] = N_eff

	def calculate_N_eff_selection(self, chunk=100):
		N_eff = self.likelihood.selection_data.calculate_N_eff_chunked(self.likelihood, self.posterior_dict, chunk=chunk)
		self.posterior_dict['pdet_n_effective'] = N_eff
		self.posterior.loc[:, 'pdet_n_effective'] = N_eff

	def calculate_rates(self, chunk=100):
		rate = self.likelihood.selection_data.calculate_rate_for_hyperparameters_chunked(self.likelihood, self.posterior_dict, chunk=chunk)
		self.posterior_dict['rate'] = rate
		self.posterior.loc[:, 'rate'] = rate

	def N_effective_cuts(self, chunk=100, selection=True, events=True):
		any_column = next(iter(self.posterior_dict.keys()))
		safe = jnp.ones(shape=self.posterior_dict[any_column].shape, dtype='bool')
		if events:
			if 'min_event_n_effective' not in self.posterior.columns:
				self.calculate_N_eff_event(chunk=chunk)
			safe = safe | (self.posterior_dict['min_event_n_effective'] > self.likelihood.N_events)

		if selection:
			if 'pdet_n_effective' not in self.posterior.columns:
				self.calculate_N_eff_selection(chunk=chunk)
			safe = safe | (self.posterior_dict['pdet_n_effective'] > 4*self.likelihood.N_events)

		self.posterior_dict_with_cuts = {key:value[safe] for key,value in self.posterior_dict.items()}
		self.posterior_with_cuts = pd.DataFrame(self.posterior_dict_with_cuts)
		return self.posterior_with_cuts

	def N_effective_cuts_events(self, chunk=100):
		if 'min_event_n_effective' not in self.posterior.columns:
			self.calculate_event_N_eff(chunk=chunk)
		safe = (self.posterior_dict['min_event_n_effective'] > self.likelihood.N_events)
		self.posterior_dict_with_cuts = {key:value[safe] for key,value in self.posterior_dict.items()}
		self.posterior_with_cuts = pd.DataFrame(self.posterior_dict_with_cuts)
		return self.posterior_with_cuts

