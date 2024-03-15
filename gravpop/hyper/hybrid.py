import jax
import jax.numpy as jnp
from ..models.utils import *
from .selection import SelectionFunction
from ..models.redshift import Redshift
from ..utils import *


from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from functools import partial

@dataclass
class HybridPopulationLikelihood:
    sampled_models:   List  # List of population models evaluated using monte-carlo
    analytic_models:  List  # List of population models evaluated analytically
    event_data:       Dict[str, jax.Array] = field(repr=False)    # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    selection_data:   Optional[Union[Dict[str, jax.Array], SelectionFunction]] = field(default=None, repr=False)   # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    
    def __post_init__(self):
        if not ('weights' in self.event_data):
            raise ValueError("Expected 'weights' key in the event_data dictionary")
        self.N_events = self.event_data['weights'].shape[0]

        if isinstance(self.selection_data, Dict):
            redshift_model = [model for model in self.sampled_models if isinstance(model, Redshift)]
            redshift_model += [model for model in self.analytic_models if isinstance(model, Redshift)]
            if len(redshift_model) == 0:
                redshift_model = None
            else:
                redshift_model = redshift_model[0]
            self.selection_data = SelectionFunction(self.selection_data, self.analysis_time, self.total_generated, self.redshift_model)
        elif isinstance(self.selection_data, SelectionFunction):
            self.analysis_time = self.selection_data.analysis_time
            self.total_generated = self.selection_data.total_generated
            self.total_detected = self.selection_data.total_detected
            self.detection_efficiency = self.detection_efficiency
        else:
            print("No selection function provided")
            self.selection_data = None
    
    @staticmethod
    def log(x):
        return jnp.log(x + 1e-30)

    def total_event_bayes_factors(self, data, params, N=None, detection_efficiency=1):
        return self.analytic_event_bayes_factors(self.event_data, params, detection_efficiency) \
                + self.sampled_event_bayes_factors(self.event_data, params, N)
    
    @staticmethod
    def aggregate_kernels(data, loglikes):
        weights = data['weights'] # E x K
        return jax.scipy.special.logsumexp( loglikes + jnp.log(weights), axis=-1) # E

    def analytic_event_bayes_factors(self, data, params, detection_efficiency=1):
        if len(self.analytic_models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.analytic_models) # E x K
        return self.aggregate_kernels(data, loglikes) + jnp.log(detection_efficiency)
    
    def sampled_event_bayes_factors(self, data, params, N=None):
        if len(self.sampled_models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.sampled_models) # E x K x N
        N = N or loglikes.shape[-1]
        if "prior" not in data.keys():
            log_priors = 0
        else:
            log_priors = self.log(data["prior"])
        loglikes = jax.scipy.special.logsumexp( loglikes - log_priors, axis=-1) - jnp.log(N)
        return self.aggregate_kernels(data, loglikes)
    
    def logpdf(self, params):
        # Event Likelihoods
        loglikes_event = self.total_event_bayes_factors(self.event_data, params)
        
        # Selection Likelihood
        loglikes_selection = 0.0
        if self.selection_data:
            loglikes_selection += self.total_event_bayes_factors(self.selection_data, params, N=self.selection, detection_efficiency=self.detection_efficiency)
        
        return loglikes_event.sum() - self.N_events * loglikes_selection

    @classmethod
    def from_file(cls, event_data_filename, selection_data_filename, sampled_models, analytic_models, SelectionClass=SelectionFunction):
        event_data = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename))
        selection_data = load_hdf5_to_jax_dict(selection_data_filename)
        selection_attributes = load_hdf5_attributes(selection_data_filename)
        if "selection" in selection_data.keys():
            selection_data = selection_data["selection"]

        if "selection" in selection_attributes.keys():
            selection_attributes = selection_attributes["selection"]

        redshift_model = [model for model in sampled_models if isinstance(model, Redshift)]
        redshift_model += [model for model in analytic_models if isinstance(model, Redshift)]
        if len(redshift_model) == 0:
            redshift_model = None
        else:
            redshift_model = redshift_model[0]
        selection = SelectionClass(selection_data, 
                                   selection_attributes['analysis_time'],
                                   selection_attributes['total_generated'],
                                   redshift_model)
        return cls(sampled_models, analytic_models, event_data, selection)
