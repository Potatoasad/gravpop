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

DEFAULT_GRID =  Grid([Grid1D(name='cos_tilt_1', minimum=-1, maximum=1, N=100, latex_name=r"$\cos(\theta_1)$"), 
                      Grid1D(name='cos_tilt_2', minimum=-1, maximum=1, N=100, latex_name=r"$\cos(\theta_2)$")])


@dataclass
class SpinOrientationPlot:
    hyper_posterior_samples : Dict[str, jax.Array] = field(repr=False)
    model : Optional[AbstractPopulationModel] = None
    spin_grid : Optional[Dict[str, Union[Grid, Grid1D]]] = None
    confidence_interval : float = 0.95
    rate : bool = False
    chunk : int = 20
    n_samples : int = 1000
    
    def __post_init__(self):
        acceptable_sample_names = self.model.hyper_var_names + ['rate']
        total_indices = self.hyper_posterior_samples[next(iter(self.hyper_posterior_samples.keys()))].size
        self.index_sampling = np.random.randint(total_indices, size=self.n_samples)
        self.hyper_posterior_samples = {col:value[self.index_sampling] for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        #self.hyper_posterior_samples = {col:value for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        self._shapes = {key:0 for key in self.hyper_posterior_samples.keys()}
        self.spin_grid = self.spin_grid or DEFAULT_GRID 
        data = self.spin_grid.data
        self.result = None
        progress_title = "Computing Spin Orientation Model on the Grid"
        self._vmapped_func = chunked_vmap(lambda x: self.model.evaluate(data, x), in_axes=(self._shapes,), chunk=self.chunk, progress_note=progress_title)
        self.conf = ((1-self.confidence_interval)/2, self.confidence_interval + (1-self.confidence_interval)/2)

        spin_1_name = [m.name for m in self.spin_grid.grid_list if "_1" in m.name][0]
        spin_2_name = [m.name for m in self.spin_grid.grid_list if "_2" in m.name][0]

        self.spin_1_grid_obj = [m for m in self.spin_grid.grid_list if "_1" in m.name][0]
        self.spin_2_grid_obj = [m for m in self.spin_grid.grid_list if "_2" in m.name][0]
        
        self.spin_1_grid = self.spin_grid.grid_1ds[spin_1_name]
        self.spin_2_grid = self.spin_grid.grid_1ds[spin_2_name]

    def compute(self):
        self.result = self._vmapped_func(self.hyper_posterior_samples)
        if self.rate:
            self.result = self.result * self.hyper_posterior_samples["rate"][..., None, None]

    def compute_if_not_computed(self):
        if self.result is None:
            self.compute()

    def spin_marginal_plot(self, spin_1_or_2=1, generic=False, ax=None, aspect=0.5, lower=0, color=None, label=None, alpha=0.3):
        self.compute_if_not_computed()
        
        if spin_1_or_2 == 1:
            spin_a = self.spin_1_grid
            spin_b = self.spin_2_grid
            latex_name = self.spin_1_grid_obj.latex_name
            if generic:
                latex_name = latex_name.replace("_1", "")
            axis_to_integrate = 1
            axis_to_aggregate = 0
        else:
            spin_a = self.spin_2_grid
            spin_b = self.spin_1_grid
            latex_name = self.spin_2_grid_obj.latex_name
            if generic:
                latex_name = latex_name.replace("_2", "")
            axis_to_integrate = 2
            axis_to_aggregate = 0
        spin_all = jax.scipy.integrate.trapezoid(self.result, spin_b, axis=axis_to_integrate)
        spin_median = jnp.quantile(spin_all, 0.5, axis=axis_to_aggregate)
        spin_lower = jnp.quantile(spin_all, self.conf[0], axis=axis_to_aggregate)
        spin_upper = jnp.quantile(spin_all, self.conf[1], axis=axis_to_aggregate)
        
        highest, lowest = spin_upper.max(),  max(spin_lower.min(), lower)
        high_x, low_x = spin_a.max(), spin_a.min()
        
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
        ax.plot(spin_a, spin_median, color=color, label=label)
        ax.fill_between(spin_a, spin_lower, spin_upper, alpha=alpha, color=color)
        #ax.set_yscale("log")
        ax.set_xlabel(latex_name)
        if self.rate:
            if generic:
                ax.set_ylabel(r"$\frac{dN}{d\cos(\theta)}$")
            else:
                ax.set_ylabel(r"$\frac{dN}{d\cos(\theta_" + str(spin_1_or_2) + r")}$")
        else:
            if generic:
                ax.set_ylabel(r"$p(\cos(\theta)| \Lambda)$")
            else:
                ax.set_ylabel(r"$p(\cos(\theta_" + str(spin_1_or_2) + r" )| \Lambda)$")
        ax.grid(True, which='major', linestyle='dotted', linewidth='0.5', color='gray', alpha=0.5)
        ax.set_xlim(spin_a.min(), spin_a.max())
        #ax.set_ylim(bottom=10**(lowest))
        #ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        new_aspect = aspect*(high_x - low_x)/(highest-lowest)
        if (new_aspect > 0) and not (np.isinf(new_aspect)):
            ax.set_aspect(aspect*(high_x - low_x)/(highest-lowest))
        #ax.set_aspect(aspect)

        return ax

    def spin_1_marginal_plot(self, *args, **kwargs):
        return self.spin_marginal_plot(*args, **kwargs, spin_1_or_2=1, generic=False)

    def spin_2_marginal_plot(self, *args, **kwargs):
        return self.spin_marginal_plot(*args, **kwargs, spin_1_or_2=2, generic=False)

    def spin_generic_marginal_plot(self, *args, **kwargs):
        return self.spin_marginal_plot(*args, **kwargs, spin_1_or_2=1, generic=True)