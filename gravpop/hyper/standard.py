import jax
import jax.numpy as jnp
from jax import jit
from ..models.utils import *
from ..models.redshift import Redshift
from ..utils import *
from .selection import SelectionFunction
import numpy as np


from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from functools import partial


@dataclass
class PopulationLikelihood:
    models:   List  # List of population models evaluated using monte-carlo
    event_data:       Dict[str, jax.Array] = field(repr=False)    # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    selection_data:   Optional[Union[Dict[str, jax.Array], SelectionFunction]] = field(default=None, repr=False)   # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    analysis_time: float = 1 # Analysis duration, default to one year
    total_generated: Optional[int] = None
    enforce_convergence: bool = False
    event_names : Optional[List[str]] = None
    
    def __post_init__(self):
        #print("redoing prior")
        if "prior" not in self.event_data:
            print("Warning: Prior not specified... setting prior to 1")
            keyvalues = list(self.event_data.items())
            largest_shape = keyvalues[np.argmax([len(value.shape) for key,value in keyvalues])][1].shape # most probably a sampled variable
            self.event_data["prior"] = jnp.ones(shape=largest_shape)
        self.N_events = self.event_data['prior'].shape[0]
        self.log_epsilon = 1e-30
        self.redshift_model = [model for model in self.models if isinstance(model, Redshift)]
        if len(self.redshift_model) == 0:
            self.redshift_model = None
        else:
            self.redshift_model = self.redshift_model[0]

        if isinstance(self.selection_data, Dict):
            self.selection_data = SelectionFunction(self.selection_data, self.analysis_time, self.total_generated, self.redshift_model)
        elif isinstance(self.selection_data, SelectionFunction):
            self.analysis_time = self.selection_data.analysis_time
            self.total_generated = self.selection_data.total_generated

        self.N_posterior_samples = self.event_data["prior"].shape[-1]
        if len(self.models) == 0:
            raise ValueError("We've got no models to work with :(")
        #print("finished making likleihood")
    
    @staticmethod
    def log(x):
        return jnp.log(x + 1e-37)

    def sampled_compute_log_weights(self, data, params):
        return sum(self.log(model(data, params)) for model in self.models) - self.log(data["prior"]) # E x N 
    
    def sampled_event_bayes_factors(self, data, params, N=None):
        logweights = self.sampled_compute_log_weights(data, params)
        N = N or logweights.shape[-1]
        loglikes = jax.scipy.special.logsumexp(logweights, axis=-1) - jnp.log(N)  # E
        return loglikes

    def total_event_bayes_factors(self, data, params, N=None, selection=False):
        if selection and self.enforce_convergence:
            logweights = self.sampled_compute_log_weights(data, params)
            log_mu, N_eff = self._compute_selection_N_eff(logweights)
            return jnp.where(N_eff > 4*self.N_events, log_mu, jnp.nan_to_num(-jnp.inf))
        elif (not selection) and self.enforce_convergence:
            logweights = self.sampled_compute_log_weights(data, params)
            log_mu, N_eff = self._compute_event_min_N_eff(logweights, N=N)
            return jnp.where(N_eff > self.N_events, log_mu, jnp.nan_to_num(-jnp.inf))
        else:
            return self.sampled_event_bayes_factors(data, params, N=N)

    def _compute_selection_N_eff(self, logweights, N=None):
        N = N or self.selection_data.total_generated
        log_mu = jax.scipy.special.logsumexp(logweights, axis=-1) - jnp.log(N)
        mu = jnp.exp(log_mu)
        log_var_1 = jax.scipy.special.logsumexp(2*logweights , axis=-1) - 2*jnp.log(N)
        log_var_2 = 2*log_mu - jnp.log(N)
        var = jnp.exp(log_var_1) - jnp.exp(log_var_2)
        N_eff = mu**2 / var
        return log_mu, N_eff

    def _compute_event_min_N_eff(self, logweights, N=None):
        N = N or logweights.shape[-1]
        log_mu = jax.scipy.special.logsumexp(logweights, axis=-1)
        log_N_eff = 2*log_mu - jax.scipy.special.logsumexp(2*logweights , axis=-1)
        N_eff = jnp.min(jnp.exp(log_N_eff), axis=-1)
        return log_mu- jnp.log(N), N_eff

    def compute_selection_N_eff(self, params, N=None):
        logweights = self.sampled_compute_log_weights(self.selection_data.selection_data, params)
        return self._compute_selection_N_eff(logweights, N=N)

    def compute_event_N_eff(self, params, N=None):
        N = N or self.N_posterior_samples
        logweights = self.sampled_compute_log_weights(self.event_data, params)
        return self._compute_event_min_N_eff(logweights, N=N)

    def compute_selection_N_eff_only(self, params, N=None):
        logweights = self.sampled_compute_log_weights(self.selection_data.selection_data, params)
        return self._compute_selection_N_eff(logweights, N=N)[1]

    def compute_event_N_eff_only(self, params, N=None):
        N = N or self.N_posterior_samples
        logweights = self.sampled_compute_log_weights(self.event_data, params)
        return self._compute_event_min_N_eff(logweights, N=N)[1]
    
    def logpdf(self, params):
        # Event Likelihoods
        loglikes_event =self.total_event_bayes_factors(self.event_data, params, N=self.N_posterior_samples)
        
        # Selection Likelihood
        loglikes_selection = 0.0
        if self.selection_data:
            loglikes_selection += self.total_event_bayes_factors(self.selection_data.selection_data, params, 
                                                                 N=self.selection_data.total_generated,
                                                                 selection=True)
        
        return jnp.nan_to_num( loglikes_event.sum() - self.N_events * loglikes_selection )

    @classmethod
    def from_file(cls, event_data_filename, selection_data_filename, models, SelectionClass=SelectionFunction, enforce_convergence=False, samples_per_event=1000):
        event_file_ext = event_data_filename.split('.')[-1]
        if event_file_ext in ('hdf5', 'h5'):
            # Extract the entire hdf5 as a nested dictionary of jax arrays
            event_data, event_names = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename))
        elif event_file_ext == 'pkl':
            # Extract a posteriors.pkl file usually an intermediate data product of gwpopulation_pipe
            # Must be a pickle of a list of pandas dataframes
            import numpy as np
            import pandas as pd
            df_list = np.load(event_data_filename, allow_pickle=True)
            minimum_length = min([min(len(df),samples_per_event) for df in df_list])
            event_data = {col : jnp.stack([df_list[i][col][0:minimum_length].values for i in range(len(df_list))]) for col in df_list[0].columns}
            event_data['mass_1_source'] = event_data['mass_1']
            event_data['chi_1'] = event_data['a_1']
            event_data['chi_2'] = event_data['a_2']
            event_names = list(range(len(event_data)))
        print(list(event_data.keys()))
        print("(N_events, N_samples_per_event) = ", event_data['prior'].shape)
        selection_data = load_hdf5_to_jax_dict(selection_data_filename)
        selection_attributes = load_hdf5_attributes(selection_data_filename)

        #print("reordering data...")
        if "selection" in selection_data.keys():
            selection_data = selection_data["selection"]

        if "selection" in selection_attributes.keys():
            selection_attributes = selection_attributes["selection"]

        redshift_model = [model for model in models if isinstance(model, Redshift)]
        if len(redshift_model) == 0:
            redshift_model = None
        else:
            redshift_model = redshift_model[0]

        #print("creating selection")
        selection = SelectionClass(selection_data, 
                                   selection_attributes['analysis_time'],
                                   selection_attributes['total_generated'],
                                   redshift_model)
        return cls(models, event_data, selection, enforce_convergence=enforce_convergence, event_names=event_names)
