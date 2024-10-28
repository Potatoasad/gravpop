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

DEFAULT_GRID =  Grid([Grid1D(name='chi_1', minimum=0, maximum=1, N=100, latex_name=r"$\chi_1$"), 
                      Grid1D(name='chi_2', minimum=0, maximum=1, N=100, latex_name=r"$\chi_2$")])


CHI_DEFAULT_GRID =  Grid([Grid1D(name='chi_1', minimum=0, maximum=1, N=100, latex_name=r"$\chi_1$"), 
                      Grid1D(name='chi_2', minimum=0, maximum=1, N=100, latex_name=r"$\chi_2$")])

A_DEFAULT_GRID =  Grid([Grid1D(name='a_1', minimum=0, maximum=1, N=100, latex_name=r"$a_1$"), 
                      Grid1D(name='a_2', minimum=0, maximum=1, N=100, latex_name=r"$a_2$")])


latex_labels = {'chi_1' : r'$\chi_1$', 'chi_2' : r'$\chi_2$', 'a_1' : r'$a_1$', 'a_2' : r'$a_2$'}


def has_spin_named_chi(posterior):
    if isinstance(posterior, pd.DataFrame):
        cols = list(posterior.columns)
    else:
        cols = list(posterior.keys())
    return any(('chi' in col) for col in cols)

def get_spin_mag_grid(posterior):
    if has_spin_named_chi(posterior):
        return CHI_DEFAULT_GRID
    else:
        return A_DEFAULT_GRID 


@dataclass
class SpinMagintudePlot:
    hyper_posterior_samples : Dict[str, jax.Array] = field(repr=False)
    model : Optional[AbstractPopulationModel] = None
    spin_grid : Optional[Dict[str, Union[Grid, Grid1D]]] = None
    confidence_interval : float = 0.95
    rate : bool = False
    chunk : int = 20
    n_samples : Optional[int] = 1000
    
    def __post_init__(self):
        acceptable_sample_names = self.model.hyper_var_names + ['rate']
        total_indices = self.hyper_posterior_samples[next(iter(self.hyper_posterior_samples.keys()))].size
        print(total_indices)
        if self.n_samples is not None:
            self.index_sampling = np.random.randint(total_indices, size=self.n_samples)
        else:
            self.index_sampling = np.arange(total_indices)
        self.hyper_posterior_samples = {col:value[self.index_sampling] for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        #self.hyper_posterior_samples = {col:value for col,value in self.hyper_posterior_samples.items() if col in acceptable_sample_names}
        self._shapes = {key:0 for key in self.hyper_posterior_samples.keys()}
        self.spin_grid = self.spin_grid or get_spin_mag_grid(self.hyper_posterior_samples)
        data = self.spin_grid.data
        self.result = None
        progress_title = "Computing Spin Magnitude Model on the Grid"
        self._vmapped_func = chunked_vmap(lambda x: self.model.evaluate(data, x), in_axes=(self._shapes,), chunk=self.chunk, progress_note=progress_title)
        self.conf = ((1-self.confidence_interval)/2, self.confidence_interval + (1-self.confidence_interval)/2)

        spin_1_name = [m.name for m in self.spin_grid.grid_list if "_1" in m.name][0]
        spin_2_name = [m.name for m in self.spin_grid.grid_list if "_2" in m.name][0]
        self.spin_1_name = spin_1_name
        self.spin_2_name = spin_2_name

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

    def spin_2D_plot(self, quantiles=[0.9, 0.5], dpi=200):
        from truncnormkde import BoundedKDEPlot
        quantiles.sort(reverse=True)
        hyper_posterior_samples = pd.DataFrame(self.hyper_posterior_samples)
        oversample = 10000 // len(hyper_posterior_samples)
        dfsamps = self.model.sample(hyper_posterior_samples, oversample=oversample)

        return BoundedKDEPlot(dfsamps, latex_labels=latex_labels).plot(quantiles=quantiles, dpi=dpi);


    def spin_2D_plot_old(self, quantiles=[0.05, 0.5, 0.95], dpi=200):
        self.compute_if_not_computed()
        spin_all = self.result
        spin_1s = self.spin_grid.data[self.spin_1_name]
        spin_2s = self.spin_grid.data[self.spin_2_name]
        #spin_median = jnp.quantile(spin_all, 0.5, axis=0)
        #spin_lower = jnp.quantile(spin_all, quantiles[0], axis=0)
        #spin_upper = jnp.quantile(spin_all, quantiles[1], axis=0)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(ncols=len(quantiles), figsize=(5*(len(quantiles)),5), dpi=dpi)
        for i in range(len(quantiles)):
            contour_plot = axes[i].contourf(spin_1s, spin_2s, jnp.quantile(spin_all, quantiles[i], axis=0))
            axes[i].set_title(f"{np.round(quantiles[i]*100, 2)} % Posterior Predictive")
            plt.colorbar(contour_plot, ax=axes[i])
            axes[i].set_ylabel(r"$\chi_2$")
            axes[i].set_xlabel(r"$\chi_1$")

        return fig

    def spin_2D_plot_old_mean(self, levels=None, log_scale=False, dpi=200, func_to_use=jnp.mean):
        self.compute_if_not_computed()
        spin_all = self.result
        spin_1s = self.spin_grid.data[self.spin_1_name]
        spin_2s = self.spin_grid.data[self.spin_2_name]
        #spin_median = jnp.quantile(spin_all, 0.5, axis=0)
        #spin_lower = jnp.quantile(spin_all, quantiles[0], axis=0)
        #spin_upper = jnp.quantile(spin_all, quantiles[1], axis=0)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, figsize=(5,5), dpi=dpi)
        if log_scale:
            contour_plot = axes.contourf(spin_1s, spin_2s, jnp.log(func_to_use(spin_all, axis=0)), levels=levels)
        else:
            contour_plot = axes.contourf(spin_1s, spin_2s, func_to_use(spin_all, axis=0), levels=levels)
        axes.set_title(f"Posterior Predictive")
        plt.colorbar(contour_plot, ax=axes)
        axes.set_ylabel(r"$\chi_2$")
        axes.set_xlabel(r"$\chi_1$")

        return fig


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
                ax.set_ylabel(r"$\frac{dN}{d\chi}$")
            else:
                ax.set_ylabel(r"$\frac{dN}{d\chi" + str(spin_1_or_2) + r"}$")
        else:
            if generic:
                ax.set_ylabel(r"$p(\chi | \Lambda)$")
            else:
                ax.set_ylabel(r"$p(\chi_" + str(spin_1_or_2) + r" | \Lambda)$")
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