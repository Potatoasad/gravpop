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

DEFAULT_GRID =  Grid([Grid1D(name='mass_1_source', minimum=2, maximum=100, N=200, latex_name=r"$m_1$"), 
                      Grid1D(name='mass_ratio', minimum=0, maximum=1  , N=200, latex_name=r"$q$")])


@dataclass
class MassPlot:
    hyper_posterior_samples : Dict[str, jax.Array] = field(repr=False)
    model : Optional[AbstractPopulationModel] = None
    mass_grid : Optional[Dict[str, Union[Grid, Grid1D]]] = None
    confidence_interval : float = 0.95
    rate : bool = False
    chunk : int = 20
    n_samples : int = 1000
    
    def __post_init__(self):
        acceptable_sample_names = self.model.hyper_var_names + ['rate']
        total_indices = self.hyper_posterior_samples[next(iter(self.hyper_posterior_samples.keys()))].size
        self.index_sampling = np.random.randint(total_indices, size=self.n_samples)
        self.hyper_posterior_samples = {col:value[self.index_sampling] for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        self._shapes = {key:0 for key in self.hyper_posterior_samples.keys()}
        self.mass_grid = self.mass_grid or DEFAULT_GRID 
        data = self.mass_grid.data
        self.result = None
        progress_title = "Computing Mass Model on the Grid"
        self._vmapped_func = chunked_vmap( lambda x: self.model.evaluate(data, x), in_axes=(self._shapes,), chunk=self.chunk, progress_note=progress_title)
        self.conf = ((1-self.confidence_interval)/2, self.confidence_interval + (1-self.confidence_interval)/2)

    def compute(self):
        self.result = self._vmapped_func(self.hyper_posterior_samples)
        if self.rate:
            self.result = self.result * self.hyper_posterior_samples["rate"][..., None, None]

    def compute_if_not_computed(self):
        if self.result is None:
            self.compute()
            
    def mass_marginal_plot(self, ax=None, aspect=0.5, log_lower=-15, color=None, label=None, alpha=0.3):
        self.compute_if_not_computed()
        mass_name = [m.name for m in self.mass_grid.grid_list if "mass_1" in m.name][0]
        mass_ratio_name = [m.name for m in self.mass_grid.grid_list if (("mass" in m.name) and ("ratio" in m.name))][0]
        
        mass_1_grid_obj = [m for m in self.mass_grid.grid_list if "mass_1" in m.name][0]
        mass_ratio_grid_obj = [m for m in self.mass_grid.grid_list if (("mass" in m.name) and ("ratio" in m.name))][0]
        
        mass_1_grid = self.mass_grid.grid_1ds[mass_name]
        mass_ratio_grid = self.mass_grid.grid_1ds[mass_ratio_name]
        
        mass_x = mass_1_grid
        mass_y_all = jax.scipy.integrate.trapezoid(self.result, mass_ratio_grid, axis=1)
        mass_y_median = jnp.quantile(mass_y_all, 0.5, axis=0)
        mass_y_lower = jnp.quantile(mass_y_all, self.conf[0], axis=0)
        mass_y_upper = jnp.quantile(mass_y_all, self.conf[1], axis=0)
        
        highest, lowest = jnp.log10(mass_y_upper.max()),  max(jnp.log10(mass_y_lower.min()), log_lower)
        high_x, low_x = mass_x.max(), mass_x.min()
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
        ax.plot(mass_x, mass_y_median, color=color, label=label)
        ax.fill_between(mass_x, mass_y_lower, mass_y_upper, color=color, alpha=alpha)
        ax.set_yscale("log")
        ax.set_xlabel(mass_1_grid_obj.latex_name)
        if self.rate:
            ax.set_ylabel(r"$\frac{dN}{dm_1}$")
        else:
            ax.set_ylabel(r"$p(m_1 | \Lambda)$")
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.set_xlim(mass_x.min(), mass_x.max())
        ax.set_ylim(bottom=10**(lowest))
        new_aspect = aspect*(high_x - low_x)/(highest-lowest)
        if (new_aspect > 0) and not (np.isinf(new_aspect)):
            ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))

        return ax
    
    def mass_ratio_marginal_plot(self, ax=None, aspect=0.5, log_lower=-15, color=None, label=None, alpha=0.3):
        self.compute_if_not_computed()
        mass_name = [m.name for m in self.mass_grid.grid_list if "mass_1" in m.name][0]
        mass_ratio_name = [m.name for m in self.mass_grid.grid_list if (("mass" in m.name) and ("ratio" in m.name))][0]
        
        mass_1_grid_obj = [m for m in self.mass_grid.grid_list if "mass_1" in m.name][0]
        mass_ratio_grid_obj = [m for m in self.mass_grid.grid_list if (("mass" in m.name) and ("ratio" in m.name))][0]
        
        mass_1_grid = self.mass_grid.grid_1ds[mass_name]
        mass_ratio_grid = self.mass_grid.grid_1ds[mass_ratio_name]
        
        mass_x = mass_ratio_grid
        mass_ratio_y_all = jax.scipy.integrate.trapezoid(self.result, mass_1_grid, axis=2)
        mass_ratio_y_median = jnp.quantile(mass_ratio_y_all, 0.5, axis=0)
        mass_ratio_y_lower = jnp.quantile(mass_ratio_y_all, self.conf[0], axis=0)
        mass_ratio_y_upper = jnp.quantile(mass_ratio_y_all, self.conf[1], axis=0)
        
        highest, lowest = jnp.log10(mass_ratio_y_upper.max()),  max(jnp.log10(mass_ratio_y_lower.min()), log_lower)
        high_x, low_x = mass_x.max(), mass_x.min()
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
        ax.plot(mass_x, mass_ratio_y_median, color=color, label=label)
        ax.fill_between(mass_x, mass_ratio_y_lower, mass_ratio_y_upper, color=color, alpha=alpha)
        ax.set_xlabel(mass_ratio_grid_obj.latex_name)
        if self.rate:
            ax.set_ylabel(r"$\frac{dN}{dq}$")
        else:
            ax.set_ylabel(r"$p(q | \Lambda)$")
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.set_xlim(mass_x.min(), mass_x.max())
        ax.set_yscale("log")
        ax.set_ylim(bottom=10**(log_lower))
        new_aspect = aspect*(high_x - low_x)/(highest-lowest)
        if (new_aspect > 0) and not (np.isinf(new_aspect)):
            ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        #ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        #ax.set_aspect(aspect*(1)/(highest-lowest))
        
        return ax