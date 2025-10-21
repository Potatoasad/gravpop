import jax
import jax.numpy as jnp
from ..models.generic.abstract import *
from ..models.utils import *
from ..utils import *
import numpy as np


from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from functools import partial

@dataclass
class MarginalizedHybridLikelihood:
    analytic_models : List[AbstractPopulationModel]
    sampled_models : List[AbstractPopulationModel]
    data : Dict[str,jax.Array] = None
    selection_data : Dict = None
    sampled_models_selection : List[AbstractPopulationModel] = None
    analytic_models_selection : List[AbstractPopulationModel] = None
    compute_variance = False

    def __post_init__(self):
        self.E, self.N, self.K = self.data['responsibilities'].shape
        if 'prior' not in self.data:
            self.data['prior'] = jnp.ones((self.E, self.N))

        if self.analytic_models_selection is None:
            self.analytic_models_selection =  self.analytic_models

        if self.sampled_models_selection is None:
            self.sampled_models_selection =  self.sampled_models

    def compute_analytic_and_sampled_lpdf(self, data, param, analytic_models, sampled_models, N, K, E=None):
        if len(analytic_models) != 0:
            analytic_lpdf = analytic_models[0](data, param)
            for model in analytic_models[1:]:
                analytic_lpdf *= model(data, param)
        else:
            if E is None:
                analytic_lpdf = jnp.ones(K)
            else:
                analytic_lpdf = jnp.ones((E, K))
        
        if len(sampled_models) != 0:
            sampled_lpdf = sampled_models[0](data, param)
            for model in sampled_models[1:]:
                sampled_lpdf *= model(data, param)
        else:
            sampled_lpdf = jnp.ones(N)
            if E is None:
                sampled_lpdf = jnp.ones(N)
            else:
                sampled_lpdf = jnp.ones((E, N))

        return analytic_lpdf, sampled_lpdf

    def logpdf(self, param):
        E, N,K = self.E, self.N, self.K
        analytic_lpdf, sampled_lpdf = self.compute_analytic_and_sampled_lpdf(self.data, param,
                                                                             self.analytic_models, 
                                                                             self.sampled_models, 
                                                                             N, K, E)

        reweighted_N = (sampled_lpdf / self.data['prior']) # ExN
        Z = self.data['responsibilities'] # E x N x K

        per_sample_weights = jnp.einsum("enk,ek->en", Z, analytic_lpdf);
        per_sample_weights *= reweighted_N;

        if self.compute_variance == True:
            expectation_events = jnp.mean(per_sample_weights, axis=-1)
            squared_expectation_events = jnp.mean(per_sample_weights**2, axis=-1)
            variance_events = ( (squared_expectation_events - expectation_events)**2 )/(N * expectation_events**2)
        else:
            expectation_events = jnp.mean(per_sample_weights, axis=-1)
        
        final_lpdf = jnp.log(1e-30 + jnp.einsum("en,en->e", reweighted_N, per_sample_weights))
        #final_lpdf = jnp.log(1e-30 + jnp.einsum("en,enk,ek->e", reweighted_N, Z, analytic_lpdf))

        if self.selection_data is not None:
            analytic_lpdf_sel, sampled_lpdf_sel = self.compute_analytic_and_sampled_lpdf(self.selection_data, param,
                                                                                 self.analytic_models_selection, 
                                                                                 self.sampled_models_selection, 
                                                                                 N, K, E=None)
            reweighted_N_sel = (sampled_lpdf_sel / self.selection_data['prior']) # ExN
            Z_sel = self.selection_data['responsibilities'] # E x N x K

            per_sample_weights_sel = jnp.einsum("enk,ek->en", Z_sel, analytic_lpdf_sel);
            per_sample_weights_sel *= reweighted_N_sel;

            if self.compute_variance == True:
                expectation_sel = jnp.mean(per_sample_weights_sel, axis=-1)
                squared_expectation_sel = jnp.mean(per_sample_weights_sel**2, axis=-1)
                variance_sel = ( (squared_expectation_sel - expectation_sel**2) )/(N * expectation_sel**2)
            else:
                expectation_sel = jnp.mean(per_sample_weights_sel, axis=-1)



            
            #final_lpdf_sel = jnp.log(1e-30 + jnp.einsum("n,nk,k", reweighted_N_sel, Z_sel, analytic_lpdf_sel))
            lpdf = jnp.sum(jnp.log(1e-30 + expectation_events),axis=-1) - self.E * jnp.log(1e-30 + expectation_sel)
            if self.compute_variance ==  True:
                return lpdf, jnp.sum(variance_events, axis=-1) + self.E * jnp.sum(expectation_sel)
            else:
                return lpdf

        if self.compute_variance == True:
            return jnp.sum(jnp.log(1e-30 + expectation_events),axis=-1), jnp.sum(variance_events, axis=-1)
        else:
            return jnp.sum(jnp.log(1e-30 + expectation_events),axis=-1)