from dataclasses import dataclass, field
from ..models import Redshift, AbstractPopulationModel
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Union, Optional
import math
import numpy as np
import scipy
from ..utils import chunked_vmap
from tqdm import trange

@dataclass
class SelectionFunction:
	selection_data : Dict[str, jax.Array]
	analysis_time : float = 1
	total_generated : Optional[int] = None
	redshift_model : Optional[AbstractPopulationModel] = None
	total_detected :Optional[int] = None

	def __post_init__(self):
		self.total_detected = self.total_detected or math.prod(self.selection_data['prior'].shape)
		self.total_generated = self.total_generated or self.total_detected
		self.detection_ratio = self.total_detected/self.total_generated

	def surveyed_hypervolume(self, params):
		return self.redshift_model.normalization(params) / 1e9 * self.analysis_time

	def rate(self, likelihood, params, N_events=None):
		hyper_vol = self.surveyed_hypervolume(params)
		efficiency = jnp.exp(likelihood.total_event_bayes_factors(self.selection_data, params, N=self.total_generated))
		N_events = N_events or (likelihood.event_data['prior'].shape[0])
		return scipy.stats.gamma.rvs(a=N_events) / (efficiency * hyper_vol)

	def calculate_rate_for_hyperparameters_chunked(self, likelihood, params, N_events=None, chunk=100):
		N_events = N_events or (likelihood.N_events)
		arbitrary_variable_in_params = next(iter(params.keys()))
		param_shape = params[arbitrary_variable_in_params].shape
		in_axes_dict_shape = {variable : 0  for variable in params.keys()}
		prog_note_1 = "Calculating Rates : Selection Function ..."
		prog_note_2 = "Calculating Rates : Surveyed Hypervolume ..."
		efficiency_func = chunked_vmap(lambda params : jnp.exp(likelihood.total_event_bayes_factors(self.selection_data, params, N=self.total_generated)), in_axes=(in_axes_dict_shape,), chunk=chunk, progress_note=prog_note_1)
		hyper_vol_func = chunked_vmap(lambda params : self.surveyed_hypervolume(params), in_axes=(in_axes_dict_shape,), chunk=chunk, progress_note=prog_note_2)

		efficiency = efficiency_func(params)
		hyper_vol = hyper_vol_func(params)

		return scipy.stats.gamma.rvs(a=N_events, size=param_shape) / (efficiency * hyper_vol)

	def calculate_rate_for_hyperparameters(self, likelihood, params, N_events=None):
		arbitrary_variable_in_params = next(iter(params.keys()))
		N = params[arbitrary_variable_in_params].size

		list_of_params = [{key: params[key][i] for key in params.keys()} for i in range(N)]
		result = []
		for i in trange(N):
			result.append(self.rate(likelihood, list_of_params[i], N_events=N_events))

		return jnp.array(result)

	def calculate_N_eff_chunked(self, likelihood, params, chunk=100):
		N_eff_func = lambda params : likelihood.compute_selection_N_eff_only(params, N=self.total_generated)
		axes_dict_shape = {key:0 for key in params.keys()}
		N_eff_chunked_func = chunked_vmap(N_eff_func, in_axes=(axes_dict_shape,), chunk=100, progress_note="Calculating Selection Function N_effective ...")

		N_effs = N_eff_chunked_func(params)

		return N_effs




