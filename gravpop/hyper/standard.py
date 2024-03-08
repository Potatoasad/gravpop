import jax
import jax.numpy as jnp
from ..models.utils import *


from dataclasses import dataclass, field
from typing import List, Dict
from functools import partial

@dataclass
class PopulationLikelihood:
    models:   List  # List of population models evaluated using monte-carlo
    event_data:       Dict[str, jax.Array] = field(repr=False)    # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    selection_data:   Dict[str, jax.Array] = field(repr=False)   # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    
    def __post_init__(self):
        self.N_events = self.event_data['prior'].shape[0]
    
    @staticmethod
    def log(x):
        return jnp.log(x)
    
    def sampled_event_bayes_factors(self, data, params):
        if len(self.models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.models) # E x N
        N = loglikes.shape[-1]
        log_priors = self.log(data["prior"])
        loglikes = jax.scipy.special.logsumexp( loglikes - log_priors, axis=-1) - jnp.log(N)
        return loglikes
    
    def logpdf(self, params):
        # Event Likelihoods
        loglikes_event =self.sampled_event_bayes_factors(self.event_data, params)
        
        # Selection Likelihood
        loglikes_selection = 0.0
        if self.selection_data:
            loglikes_selection += self.sampled_event_bayes_factors(self.selection_data, params)
        
        return jnp.nan_to_num( loglikes_event.sum() - self.N_events * loglikes_selection )