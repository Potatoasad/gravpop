import jax
import jax.numpy as jnp
from jax import jit
from ..models.utils import *
from ..models.redshift import Redshift
from ..utils import *
from .selection import SelectionFunction


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
    
    def __post_init__(self):
        self.N_events = self.event_data['prior'].shape[0]
        self.log_epsilon = 1e-30
        redshift_model = [model for model in self.models if isinstance(model, Redshift)]
        if len(redshift_model) == 0:
            redshift_model = None
        else:
            redshift_model = redshift_model[0]

        if isinstance(self.selection_data, Dict):
            self.selection_data = SelectionFunction(self.selection_data, self.analysis_time, self.total_generated, self.redshift_model)
        elif isinstance(self.selection_data, SelectionFunction):
            self.analysis_time = self.selection_data.analysis_time
            self.total_generated = self.selection_data.total_generated
    
    @staticmethod
    def log(x):
        return jnp.log(x + 1e-30)
    
    def sampled_event_bayes_factors(self, data, params, N=None):
        if len(self.models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.models) # E x N
        log_priors = self.log(data["prior"])
        if N is None:
            N = loglikes.shape[-1]
        loglikes = jax.scipy.special.logsumexp( loglikes - log_priors, axis=-1) - jnp.log(N)
        return loglikes

    def total_event_bayes_factors(self, data, params, N=None):
        return self.sampled_event_bayes_factors(data, params, N=N)
    
    def logpdf(self, params):
        # Event Likelihoods
        loglikes_event =self.total_event_bayes_factors(self.event_data, params)
        
        # Selection Likelihood
        loglikes_selection = 0.0
        if self.selection_data:
            loglikes_selection += self.total_event_bayes_factors(self.selection_data.selection_data, params, N=self.selection_data.total_generated)
        
        return jnp.nan_to_num( loglikes_event.sum() - self.N_events * loglikes_selection )

    @classmethod
    def from_file(cls, event_data_filename, selection_data_filename, models, SelectionClass=SelectionFunction):
        event_data = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename))
        selection_data = load_hdf5_to_jax_dict(selection_data_filename)
        selection_attributes = load_hdf5_attributes(selection_data_filename)
        if "selection" in selection_data.keys():
            selection_data = selection_data["selection"]

        if "selection" in selection_attributes.keys():
            selection_attributes = selection_attributes["selection"]

        redshift_model = [model for model in models if isinstance(model, Redshift)]
        if len(redshift_model) == 0:
            redshift_model = None
        else:
            redshift_model = redshift_model[0]
        selection = SelectionClass(selection_data, 
                                   selection_attributes['analysis_time'],
                                   selection_attributes['total_generated'],
                                   redshift_model)
        return cls(models, event_data, selection)
