from dataclasses import dataclass
import pandas as pd
from typing import Optional, List, Union, Dict, Any
import jax
import jax.numpy as jnp
from ..plots import *
from ..models import *
from ..hyper import *

MASS_MODELS = [SmoothedTwoComponentPrimaryMassRatio]

@dataclass
class HyperPosterior:
	posterior : Optional[Union[pd.DataFrame, Dict[str, jax.Array]]]
	likelihood : Any
	mass_model : Optional[AbstractPopulationModel] = None
	redshift_model : Optional[AbstractPopulationModel] = None
	rate : bool = False

	def __post_init__(self):
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

		self.mass_plot = MassPlot(self.posterior_dict, self.mass_model)
		self.redshift_plot = RedshiftPlot(self.posterior_dict, self.redshift_model)
		

	@classmethod
	def from_file(cls, posterior_sample_file, event_data_file, selection_file, models, rate=False, LikelihoodClass=PopulationLikelihood, SelectionClass=SelectionFunction):
		posterior_file_extension = posterior_sample_file.split(".")[-1]
		if posterior_file_extension == "csv":
			## Assume it can be read by pandas
			posterior = pd.read_csv(posterior_sample_file)

		likelihood = LikelihoodClass.from_file(
		            event_data_filename = event_data_file,
		            selection_data_filename = selection_file,
		            models = models
		            )

		return cls(posterior, likelihood, rate=rate)

	def calculate_N_eff(self, chunk=100):
		N_eff = self.likelihood.selection_data.calculate_N_eff_chunked(self.likelihood, self.posterior_dict, chunk=chunk)
		self.posterior_dict['N_eff'] = N_eff
		self.posterior.loc[:, 'N_eff'] = N_eff

	def calculate_rates(self, chunk=100):
		rate = self.likelihood.selection_data.calculate_rate_for_hyperparameters_chunked(self.likelihood, self.posterior_dict, chunk=chunk)
		self.posterior_dict['rate'] = rate
		self.posterior.loc[:, 'rate'] = rate

	def N_effective_cuts(self, chunk=100):
		if 'N_eff' not in self.posterior.columns:
			self.calculate_N_eff(chunk=chunk)
		safe = self.posterior_dict['N_eff'] > 4*self.likelihood.N_events
		self.posterior_dict_with_cuts = {key:value[safe] for key,value in self.posterior_dict.items()}
		self.posterior_with_cuts = pd.DataFrame(self.posterior_dict_with_cuts)
		return self.posterior_with_cuts


