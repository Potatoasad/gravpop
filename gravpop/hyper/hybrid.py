import jax
import jax.numpy as jnp
from ..models.utils import *


from dataclasses import dataclass, field
from typing import List, Dict
from functools import partial

@dataclass
class HybridPopulationLikelihood:
    sampled_models:   List  # List of population models evaluated using monte-carlo
    analytic_models:  List  # List of population models evaluated analytically
    event_data:       Dict[str, jax.Array] = field(repr=False)    # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    selection_data:   Dict[str, jax.Array] = field(repr=False)   # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    
    def __post_init__(self):
        if not ('weights' in self.event_data):
            raise ValueError("Expected 'weights' key in the event_data dictionary")
        self.N_events = self.event_data['weights'].shape[0]
    
    @staticmethod
    def log(x):
        return jnp.log(x + 1e-10)
    
    @staticmethod
    def aggregate_kernels(data, loglikes):
        weights = data['weights'] # E x K
        return jax.scipy.special.logsumexp( loglikes + jnp.log(weights), axis=-1) # E

    def analytic_event_bayes_factors(self, data, params):
        if len(self.analytic_models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.analytic_models) # E x K
        return self.aggregate_kernels(data, loglikes)
    
    def sampled_event_bayes_factors(self, data, params):
        if len(self.sampled_models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.sampled_models) # E x K x N
        N = loglikes.shape[-1]
        if "prior" not in data.keys():
            log_priors = 0
        else:
            log_priors = self.log(data["prior"])
        loglikes = jax.scipy.special.logsumexp( loglikes - log_priors, axis=-1) - jnp.log(N)
        return self.aggregate_kernels(data, loglikes)
    
    def logpdf(self, params):
        # Event Likelihoods
        loglikes_event = self.analytic_event_bayes_factors(self.event_data, params) \
                        + self.sampled_event_bayes_factors(self.event_data, params)
        
        # Selection Likelihood
        loglikes_selection = 0.0
        if self.selection_data:
            loglikes_selection += self.analytic_event_bayes_factors(self.selection_data, params) \
                                + self.sampled_event_bayes_factors(self.selection_data, params)
        
        return loglikes_event.sum() - self.N_events * loglikes_selection