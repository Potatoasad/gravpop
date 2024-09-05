import jax
import jax.numpy as jnp
from ..models.utils import *
from .selection import SelectionFunction
from ..models.redshift import Redshift
from ..utils import *
from .expectation import *
import numpy as np


from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from functools import partial


def fix_kernels(mus, sigmas, a, b, lower_lim=-4, upper_lim=4, n=10):
    alphas, betas = (mus - a)/sigmas, (mus - b)/sigmas
    too_far_left = (alphas < lower_lim) & (betas < 0)
    too_far_right = (betas > upper_lim) & (alphas > 0)
    sigmas_fixed = jnp.where(too_far_left | too_far_right, sigmas/jnp.sqrt(n), sigmas) 
    mus_fixed = jnp.where(too_far_left, a - (a-mus)/n, jnp.where(too_far_right, b + (mus-b)/n, mus))
    return mus_fixed, sigmas_fixed

@dataclass
class HybridPopulationLikelihood:
    sampled_models:   List  # List of population models evaluated using monte-carlo
    analytic_models:  List  # List of population models evaluated analytically
    event_data:       Dict[str, jax.Array] = field(repr=False)    # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    selection_data:   Optional[Union[Dict[str, jax.Array], SelectionFunction]] = field(default=None, repr=False)   # Dictionary of data e.g. {'mass_1' : jnp.array([21.2, 23.0, ...], ....)}
    event_expectation: Optional[AbstractExpectation] = field(default=HybridEventExpectation())
    selection_expectation: Optional[AbstractExpectation] = field(default=HybridSelectionExpectation())
    enforce_convergence : Optional[bool] = False
    event_names : Optional[List[str]] = None
    selection_sampled_models: Optional[str] = None
    selection_analytic_models: Optional[str] = None
    kernel_fix_iterations : int = 10
    
    def __post_init__(self):
        if not ('weights' in self.event_data):
            raise ValueError("Expected 'weights' key in the event_data dictionary")
        self.N_events = self.event_data['weights'].shape[0]

        if self.selection_sampled_models is None:
            self.selection_sampled_models = self.sampled_models

        if self.selection_analytic_models is None:
            self.selection_analytic_models = self.analytic_models

        if "prior" not in self.event_data:
            keyvalues = list(self.event_data.items())
            largest_shape = keyvalues[np.argmax([len(value.shape) for key,value in keyvalues])][1].shape # most probably a sampled variable
            #largest_shape = self.event_data[key_with_largest_shape].shape
            self.event_data["prior"] = jnp.ones(shape=largest_shape)

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
            self.detection_ratio = self.selection_data.detection_ratio
        else:
            print("No selection function provided")
            self.selection_data = None

        self._models = self.sampled_models + self.analytic_models

        self.analytic_limits = {}
        for model in self.analytic_models:
            self.analytic_limits.update(model.limits)

        if self.event_names is None:
            self.event_names = []

        for var in self.analytic_limits.keys():
            if self.selection_data is not None:
                sel = self.selection_data.selection_data
                mus, sigmas = fix_kernels(sel[var + '_mu_kernel'], sel[var  + '_sigma_kernel'], self.analytic_limits[var][0], self.analytic_limits[var][1], n=self.kernel_fix_iterations)
                self.selection_data.selection_data[var + '_mu_kernel'], self.selection_data.selection_data[var + '_sigma_kernel'] = mus, sigmas

            eventd = self.event_data
            mus, sigmas = fix_kernels(eventd[var + '_mu_kernel'], eventd[var  + '_sigma_kernel'], self.analytic_limits[var][0], self.analytic_limits[var][1], n=self.kernel_fix_iterations)
            self.event_data[var + '_mu_kernel'], self.event_data[var + '_sigma_kernel'] = mus, sigmas

        #print("Initialized Likelihood with variables:")
        #print(self.event_data.keys())
        #print("With events:")
        #print(self.event_names)

    @property
    def models(self):
        return self._models
    
    @staticmethod
    def log(x):
        return jnp.log(x + 1e-30)

    def total_event_bayes_factors(self, data, params, N=None, detection_ratio=1):
        return self.analytic_event_bayes_factors(self.event_data, params, detection_ratio) \
                + self.sampled_event_bayes_factors(self.event_data, params, N)

    def sampled_compute_log_weights(self, data, params):
        return sum(self.log(model(data, params)) for model in self.sampled_models) - self.log(data["prior"])

    def _compute_sampled_selection_N_eff(self, logweights, weights, N=None):
        log_mu = jax.scipy.special.logsumexp(logweights, axis=-1) - jnp.log(N) 
        mu = jnp.exp(log_mu)  # E x K
        log_var_1 = jax.scipy.special.logsumexp(2*logweights , axis=-1) - 2*jnp.log(N) # E x K
        log_var_2 = 2*log_mu - jnp.log(N)  # E x K
        var = jnp.exp(log_var_1) - jnp.exp(log_var_2) # E x K
        var = jnp.sum(var * weights ,axis=-1) # E
        mu  = jnp.sum(mu  * weights ,axis=-1) # E
        N_eff = mu**2 / var # E
        return N_eff

    def _compute_sampled_event_N_eff(self, logweights, weights, N=None):
        N = N or logweights.shape[-1]
        log_mu = jax.scipy.special.logsumexp(logweights, axis=-1) # E x K
        mu = jnp.exp(log_mu) # E x K
        mu_squared = jnp.exp(jax.scipy.special.logsumexp(2*logweights , axis=-1)) # E x K

        mu = jnp.sum( mu * weights ,axis=-1)
        mu_squared = jnp.sum( mu_squared * weights, axis=-1)
        N_eff = mu**2 / mu_squared # E x K
        N_eff = jnp.min(jnp.exp(log_N_eff), axis=-1)
        return log_mu- jnp.log(N), N_eff
    
    @staticmethod
    def aggregate_kernels(data, loglikes):
        weights = data['weights'] # E x K
        return jax.scipy.special.logsumexp( loglikes + jnp.log(weights), axis=-1) # E

    def analytic_event_bayes_factors(self, data, params, detection_ratio=1):
        if len(self.analytic_models) == 0:
            return 0.0
        loglikes = sum(self.log(model(data, params)) for model in self.analytic_models) # E x K
        return self.aggregate_kernels(data, loglikes) + jnp.log(detection_ratio)
    
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

    def selection_cut(self, loglikes, N_eff):
        #print(N_eff)
        return jnp.where(N_eff > 4*self.N_events, loglikes, -jnp.nan_to_num(jnp.inf) * jnp.ones_like(loglikes))

    def event_cut(self, loglikes, N_eff):
        #print(N_eff, self.N_events, jnp.min(N_eff))
        return jnp.where(jnp.min(N_eff) > self.N_events, loglikes, -jnp.nan_to_num(jnp.inf) * jnp.ones_like(loglikes))

    def log_bayes_factors_event(self, params):
        if len(self.sampled_models) != 0:
            sampled_logweights_event = sum(self.log(model(self.event_data, params)) for model in self.sampled_models) - self.log(self.event_data["prior"])
        else:
            sampled_logweights_event = None

        ## Event analytic:
        if len(self.analytic_models) != 0:
            analytic_logweights_event = sum(self.log(model(self.event_data, params)) for model in self.analytic_models)
        else:
            analytic_logweights_event = None
        loglikes_event = self.event_expectation.log_bayes_factors(sampled_logweights=sampled_logweights_event, 
                                                                  analytic_logweights=analytic_logweights_event, 
                                                                  weights=self.event_data["weights"])
        if self.enforce_convergence:
            N_eff_events = self.event_expectation.N_eff(sampled_logweights=sampled_logweights_event, 
                                                        analytic_logweights=analytic_logweights_event, 
                                                        weights=self.event_data["weights"],
                                                        mu = jnp.exp(loglikes_event))

            return self.event_cut(loglikes_event, N_eff_events)


        return loglikes_event

    def log_bayes_factors_selection(self, params):
        data = self.selection_data.selection_data
        if len(self.sampled_models) != 0:
            sampled_logweights_selection = sum(self.log(model(data, params)) for model in self.selection_sampled_models) - self.log(data["prior"])
        else:
            sampled_logweights_selection = None

        if len(self.analytic_models) != 0:
            analytic_logweights_selection = sum(self.log(model(data, params)) for model in self.selection_analytic_models)
        else:
            analytic_logweights_selection = None

        loglikes_selection = self.selection_expectation.log_bayes_factors(sampled_logweights=sampled_logweights_selection, 
                                                                          analytic_logweights=analytic_logweights_selection, 
                                                                          weights=data["weights"])

        if self.enforce_convergence:
            N_eff_selection = self.selection_expectation.N_eff(sampled_logweights=sampled_logweights_selection, 
                                                                analytic_logweights=analytic_logweights_selection, 
                                                                weights=data["weights"],
                                                                mu = jnp.exp(loglikes_selection),
                                                                total_generated=self.selection_data.total_generated)
            return self.selection_cut(loglikes_selection, N_eff_selection)


        return loglikes_selection
    
    def logpdf(self, params):
        # Event Likelihoods
        loglikes_event = self.log_bayes_factors_event(params)
                
        # Selection Likelihood
        loglikes_selection = 0.0
        if self.selection_data:
            loglikes_selection += self.log_bayes_factors_selection(params)

        return loglikes_event.sum() - self.N_events * loglikes_selection

    @classmethod
    def from_file(cls, event_data_filename, selection_data_filename, sampled_models, analytic_models, 
                       SelectionClass=SelectionFunction, enforce_convergence=False, ignore_events=[], 
                       downsample=None, inflate_selection_spins=False, inflate_selection_spins_factor=4, downsample_selection=False,
                       selection_sampled_models=None, selection_analytic_models=None, kernel_fix_iterations=10):
        event_data, event_names = stack_nested_jax_arrays(load_hdf5_to_jax_dict(event_data_filename, ignore_events=ignore_events))
        #print(event_data.keys())

        if (len(sampled_models) != 0) and (downsample is not None):
            varname = sampled_models[0].var_names[0]
            #print(varname, len(event_data[varname].shape))
            E,K,N = event_data[varname].shape
            inds = np.random.randint(0,N, size=downsample)
            for col in event_data.keys():
                the_var_to_change = event_data[col]
                its_an_array = hasattr(the_var_to_change, 'shape')
                if its_an_array:
                    if len(the_var_to_change.shape) == 3:
                        event_data[col] = event_data[col][:,:,inds]

        redshift_model = [model for model in sampled_models if isinstance(model, Redshift)]
        redshift_model += [model for model in analytic_models if isinstance(model, Redshift)]
        if len(redshift_model) == 0:
            redshift_model = None
        else:
            redshift_model = redshift_model[0]

        if selection_data_filename is not None:
            selection_data = load_hdf5_to_jax_dict(selection_data_filename)
            selection_attributes = load_hdf5_attributes(selection_data_filename)
            if "selection" in selection_data.keys():
                selection_data = selection_data["selection"]

            if "selection" in selection_attributes.keys():
                selection_attributes = selection_attributes["selection"]

            if (len(sampled_models) != 0) and (downsample is not None) and (downsample_selection):
                varname = sampled_models[0].var_names[0]
                K,N = selection_data[varname].shape
                inds = np.random.randint(0,N, size=downsample)
                for col in selection_data.keys():
                    the_var_to_change = selection_data[col]
                    its_an_array = hasattr(the_var_to_change, 'shape')
                    if its_an_array:
                        if len(selection_data[col].shape) == 2:
                            selection_data[col] = selection_data[col][:,inds]


            if inflate_selection_spins:
                selection_data['chi_1_sigma_kernel'] *= inflate_selection_spins_factor
                selection_data['chi_2_sigma_kernel'] *= inflate_selection_spins_factor

            selection = SelectionClass(selection_data, 
                                       selection_attributes['analysis_time'],
                                       selection_attributes['total_generated'],
                                       redshift_model)
        else:
            selection = None

        return cls(sampled_models=sampled_models, analytic_models=analytic_models, selection_sampled_models=selection_sampled_models, 
                  selection_analytic_models=selection_analytic_models, event_data=event_data, 
                  selection_data=selection, enforce_convergence=enforce_convergence, event_names=event_names, kernel_fix_iterations=kernel_fix_iterations)
