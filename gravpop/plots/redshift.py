from typing import Union, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from .grid import Grid1D, Grid
from ..utils.vmap import chunked_vmap
from ..models import AbstractPopulationModel
from ..hyper import PopulationLikelihood
import jax
import jax.numpy as jnp


DEFAULT_GRID =  Grid1D(name='redshift', minimum=0, maximum=1.9, N=100, latex_name=r"$z$")

@dataclass
class RedshiftPlot:
    hyper_posterior_samples : Dict[str, jax.Array] = field(repr=False)
    model : Optional[AbstractPopulationModel] = None
    redshift_grid : Optional[Dict[str, Union[Grid, Grid1D]]] = None
    confidence_interval : float = 0.95
    rate : float = False
    chunk : int = 1000
    
    def __post_init__(self):
        self._shapes = {key:0 for key in self.hyper_posterior_samples.keys()}
        self.redshift_grid = self.redshift_grid or DEFAULT_GRID
        data = self.redshift_grid.data
        compute_rate = lambda data,x : (1+data['redshift'])**(x['lamb'])
        self._vmapped_func = chunked_vmap( lambda x: self.model(data, x), in_axes=(self._shapes,), chunk=self.chunk)
        self._vmapped_func_rate = chunked_vmap( lambda x: compute_rate(data, x), in_axes=(self._shapes,), chunk=self.chunk)
            
        self.result = self._vmapped_func(self.hyper_posterior_samples)
        if self.rate:
            self.result = self.result * self.hyper_posterior_samples["rate"][..., None, None]
        
        self.conf = ((1-self.confidence_interval)/2, self.confidence_interval + (1-self.confidence_interval)/2)
            
    def plot_model(self, ax=None, aspect=0.5, log_lower=-15):
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
        ax.grid(True, which='major', linestyle='dotted', linewidth=1, color='black', alpha=0.3)
        ax.plot(redshift_x, redshift_y_median)
        ax.fill_between(redshift_x, redshift_y_lower, redshift_y_upper, alpha=0.3)
        ax.set_yscale("log")
        ax.set_xlabel(self.redshift_grid.latex_name)
        ax.set_ylabel(r"$p(z | \Lambda)$")
        second_lowest = np.partition(redshift_x, 1)[1]  
        ax.set_xlim(second_lowest, redshift_x.max())
        ax.set_ylim(bottom=10**(log_lower))
        ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        
        return fig
    
    def plot_1pz(self, ax=None, aspect=0.5, log_lower=-15):
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
        ax.grid(True, which='major', linestyle='dotted', linewidth=1, color='black', alpha=0.3)
        ax.plot(redshift_x, redshift_y_median)
        ax.fill_between(redshift_x, redshift_y_lower, redshift_y_upper, alpha=0.3)
        ax.set_yscale("log")
        ax.set_xlabel(self.redshift_grid.latex_name)
        ax.set_ylabel(r"$p(z | \Lambda)$")
        second_lowest = np.partition(redshift_x, 1)[1]  
        ax.set_xlim(second_lowest, redshift_x.max())
        ax.set_ylim(bottom=10**(log_lower))
        ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        
        return fig