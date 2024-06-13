import jax
import jax.numpy as jnp
from .abstract import *
from ..utils import *
import pandas as pd
import numpy as np

class Uniform2DAnalytic(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=[]):
        self.a = a
        self.b = b
        self.norm = np.prod([(self.b[i] - self.a[i]) for i in range(len(a))])
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.var_name_1 = var_names[0]
        self.var_name_2 = var_names[1]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}
    
    def get_data(self, data, params, component):
        var_name = self.var_name_1 if (component == 1) else self.var_name_2
        X_locations = data[var_name + '_mu_kernel'];
        X_scales    = data[var_name + '_sigma_kernel'];
        return X_locations, X_scales
    
    def evaluate(self, data, params):
        x_1 = data[self.var_name_1]
        return jnp.ones(len(x_1)) / self.norm
        
    def __call__(self, data, params):
        X_locations, X_scales = self.get_data(data, params, component=1);
        return jnp.ones_like(X_locations) / self.norm
    
    def _sample(self, df_hyper_samples):
        samps = {}
        for var in self.var_names:
            a,b = self.limits[var]
            samps[var] = a + np.random.rand(len(df_hyper_samples))*(b-a)
        return pd.DataFrame(samps)

    def sample(self, df_hyper_samples, oversample=1):
        return pd.concat([self._sample(df_hyper_samples) for _ in range(oversample)])
