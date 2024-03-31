from typing import Union, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from .grid import Grid1D, Grid
from ..utils.vmap import chunked_vmap
from ..models import AbstractPopulationModel
from ..hyper import PopulationLikelihood
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_GRID =  Grid1D(name='redshift', minimum=0, maximum=1.9, N=100, latex_name=r"$z$")

@dataclass
class RedshiftPlot:
    hyper_posterior_samples : Dict[str, jax.Array] = field(repr=False)
    model : Optional[AbstractPopulationModel] = None
    redshift_grid : Optional[Dict[str, Union[Grid, Grid1D]]] = None
    confidence_interval : float = 0.95
    rate : float = False
    chunk : int = 100
    n_samples : int = 1000
    
    def __post_init__(self):
        acceptable_sample_names = self.model.hyper_var_names + ['rate']
        total_indices = self.hyper_posterior_samples[next(iter(self.hyper_posterior_samples.keys()))].size
        self.index_sampling = np.random.randint(total_indices, size=self.n_samples)
        self.hyper_posterior_samples = {col:value[self.index_sampling] for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        #self.hyper_posterior_samples = {col:value for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        self._shapes = {key:0 for key in self.hyper_posterior_samples.keys()}
        self.redshift_grid = self.redshift_grid or DEFAULT_GRID
        data = self.redshift_grid.data
        compute_rate = lambda data,x : (1+data['redshift'])**(x[self.model.hyper_var_names[0]])
        self._vmapped_func = chunked_vmap( lambda x: self.model.evaluate(data, x), in_axes=(self._shapes,), chunk=self.chunk)
        progress_title = "Computing Redshift Model on the Grid"
        self._vmapped_func_rate = chunked_vmap( lambda x: compute_rate(data, x), in_axes=(self._shapes,), chunk=self.chunk, progress_note=progress_title)
        self.result = None
        self.conf = ((1-self.confidence_interval)/2, self.confidence_interval + (1-self.confidence_interval)/2)

    def compute(self):
        self.result = self._vmapped_func(self.hyper_posterior_samples)
        if self.rate:
            self.result = self.result * self.hyper_posterior_samples["rate"][..., None, None]

    def compute_if_not_computed(self):
        if self.result is None:
            self.compute()
            
    def plot_model(self, ax=None, aspect=0.5, log_lower=-15, color=None, label=None, alpha=0.3):
        self.compute_if_not_computed()
        redshift_x = list(self.redshift_grid.data.values())[0]
        redshift_y_median = jnp.quantile(self.result, 0.5, axis=0)
        redshift_y_lower = jnp.quantile(self.result, self.conf[0], axis=0)
        redshift_y_upper = jnp.quantile(self.result, self.conf[1], axis=0)
        
        highest, lowest = jnp.log10(redshift_y_upper.max()),  max(jnp.log10(redshift_y_lower.min()), log_lower)
        high_x, low_x = redshift_x.max(), redshift_x.min()
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
        else:
            fig = ax.get_figure()
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.plot(redshift_x, redshift_y_median, color=color, label=label)
        ax.fill_between(redshift_x, redshift_y_lower, redshift_y_upper, alpha=alpha, color=color)
        ax.set_yscale("log")
        ax.set_xlabel(self.redshift_grid.latex_name)
        ax.set_ylabel(r"$p(z | \Lambda)$")
        second_lowest = np.partition(redshift_x, 1)[1]  
        ax.set_xlim(second_lowest, redshift_x.max())
        ax.set_ylim(bottom=10**(log_lower))
        #ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        new_aspect = aspect*(high_x - low_x)/(highest-lowest)
        if (new_aspect > 0) and not (np.isinf(new_aspect)):
            ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        
        return fig
    
    def plot_1pz(self, ax=None, aspect=0.5, log_lower=-15, color=None, label=None, alpha=0.3):
        self.compute_if_not_computed()
        self.result_1pz = self._vmapped_func_rate(self.hyper_posterior_samples)
        redshift_x = list(self.redshift_grid.data.values())[0]
        redshift_y_median = jnp.quantile(self.result_1pz, 0.5, axis=0)
        redshift_y_lower = jnp.quantile(self.result_1pz, self.conf[0], axis=0)
        redshift_y_upper = jnp.quantile(self.result_1pz, self.conf[1], axis=0)
        
        highest, lowest = jnp.log10(redshift_y_upper.max()),  max(jnp.log10(redshift_y_lower.min()), log_lower)
        high_x, low_x = redshift_x.max(), redshift_x.min()
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
        else:
            fig = ax.get_figure()
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.plot(redshift_x, redshift_y_median, color=color, label=label)
        ax.fill_between(redshift_x, redshift_y_lower, redshift_y_upper, alpha=alpha, color=color)
        ax.set_yscale("log")
        ax.set_xlabel(self.redshift_grid.latex_name)
        ax.set_ylabel(r"$p(z | \Lambda)$")
        second_lowest = np.partition(redshift_x, 1)[1]  
        ax.set_xlim(second_lowest, redshift_x.max())
        ax.set_ylim(bottom=10**(log_lower))
        #ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        new_aspect = aspect*(high_x - low_x)/(highest-lowest)
        if (new_aspect > 0) and not (np.isinf(new_aspect)):
            ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        
        return fig