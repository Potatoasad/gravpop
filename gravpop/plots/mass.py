from typing import Union, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from .grid import Grid1D, Grid
from ..utils.vmap import chunked_vmap
from ..models import AbstractPopulationModel
from ..hyper import PopulationLikelihood
import jax
import jax.numpy as jnp

@dataclass
class MassPlot:
    hyper_posterior_samples : Dict[str, jax.Array] = field(repr=False)
    mass_grid : Dict[str, Union[Grid, Grid1D]] = field(repr=False)
    model : Optional[AbstractPopulationModel] = None
    confidence_interval : float = 0.95
    rate : bool = False
    chunk : int = 100
    
    def __post_init__(self):
        self._shapes = {key:0 for key in self.hyper_posterior_samples.keys()}
        data = self.mass_grid.data
        self._vmapped_func = chunked_vmap( lambda x: self.model(data, x), in_axes=(self._shapes,), chunk=self.chunk)
        
        self.result = self._vmapped_func(self.hyper_posterior_samples)
        if self.rate:
            self.result = self.result * self.hyper_posterior_samples["rate"][..., None, None]
        self.conf = ((1-self.confidence_interval)/2, self.confidence_interval + (1-self.confidence_interval)/2)
            
    def mass_marginal_plot(self, ax=None, aspect=0.5, log_lower=-15):
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
        
        #print(highest, lowest, high_x, low_x)
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
        ax.plot(mass_x, mass_y_median)
        ax.fill_between(mass_x, mass_y_lower, mass_y_upper, alpha=0.3)
        ax.set_yscale("log")
        ax.set_xlabel(mass_1_grid_obj.latex_name)
        if self.rate:
            ax.set_ylabel(r"$\frac{dN}{dm_1}$")
        else:
            ax.set_ylabel(r"$p(m_1 | \Lambda)$")
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.set_xlim(mass_x.min(), mass_x.max())
        ax.set_ylim(bottom=10**(lowest))
        ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        #ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))

        return ax
    
    def mass_ratio_marginal_plot(self, ax=None, aspect=0.5, log_lower=-15):
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
        ax.plot(mass_x, mass_ratio_y_median)
        ax.fill_between(mass_x, mass_ratio_y_lower, mass_ratio_y_upper, alpha=0.3)
        ax.set_xlabel(mass_ratio_grid_obj.latex_name)
        if self.rate:
            ax.set_ylabel(r"$\frac{dN}{dq}$")
        else:
            ax.set_ylabel(r"$p(q | \Lambda)$")
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.set_xlim(mass_x.min(), mass_x.max())
        ax.set_yscale("log")
        ax.set_ylim(bottom=10**(log_lower))
        ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        #ax.set_aspect(aspect*(1)/(highest-lowest))
        
        return ax