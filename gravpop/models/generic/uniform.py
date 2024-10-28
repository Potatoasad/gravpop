import jax
import jax.numpy as jnp
from .abstract import *
from ..utils import *
from .truncatedgaussian1D import logC
import pandas as pd
import numpy as np

class Uniform1DAnalytic(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['chi_1'], hyper_var_names=[]):
        self.a = a
        self.b = b
        self.norm = (self.b - self.a)
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}
    
    def get_data(self, data, params, component):
        var_name = self.var_name_1 if (component == 1) else self.var_name_2
        X_locations = data[var_name + '_mu_kernel'];
        X_scales    = data[var_name + '_sigma_kernel'];
        return X_locations, X_scales
    
    def evaluate(self, data, params):
        x_1 = data[self.var_names[0]]
        return jnp.ones(len(x_1)) / self.norm
        
    def __call__(self, data, params):
        X_locations = data[self.var_names[0] + '_mu_kernel'];
        return jnp.ones_like(X_locations) / self.norm
    
    def _sample(self, df_hyper_samples):
        samps = {}
        for var in self.var_names:
            a,b = self.limits[var]
            samps[var] = a + np.random.rand(len(df_hyper_samples))*(b-a)
        return pd.DataFrame(samps)

    def sample(self, df_hyper_samples, oversample=1):
        return pd.concat([self._sample(df_hyper_samples) for _ in range(oversample)])


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



class Uniform1DAnalyticVariedLimits(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['x_min', 'x_max']):
        self.a = a
        self.b = b
        self.var_name = var_names[0]
        self.x_min = hyper_var_names[0];
        self.x_max = hyper_var_names[1];
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}
        
    def get_data(self, data, params):
        X_locations = data[self.var_name + '_mu_kernel'];
        X_scales    = data[self.var_name + '_sigma_kernel'];
        x_min       = params.get(self.x_min, self.a);
        x_max       = params.get(self.x_max, self.b); 
        return X_locations, X_scales, x_min, x_max

    def evaluate(self, data, params):
        """
        Evaluates the value of the *integrand* and not the integral
        """
        x_min       = params.get(self.x_min, self.a);
        x_max       = params.get(self.x_max, self.b);
        #loglikes = jax.scipy.special.logsumexp( jnp.log(truncnorm(Xs, mu, sigma, low=a, high=b)) , axis=-1) - jnp.log(Xs.shape[-1])
        prob = box(data[self.var_name], x_min,x_max)/(x_max - x_min)
        return prob
        
    def __call__(self, data, params):
        X_locations, X_scales, x_min, x_max = self.get_data(data, params);
        X_stddevs_outside = jnp.where(X_locations <= x_max, 
                                        jnp.where(X_locations <= x_min, 
                                            (x_min - X_locations)/X_scales, jnp.zeros_like(X_locations)),
                                                    (X_locations - x_max)/X_scales)
        loglikes = logC(X_locations, X_scales, x_min, x_max) - logC(X_locations, X_scales, self.a, self.b) - jnp.log(x_max - x_min)
        likes = jnp.where(X_stddevs_outside < 4, jnp.exp(loglikes), jnp.zeros_like(loglikes))
        return likes

    def sample(self, df_hyper_samples, oversample=1):
        N = len(df_hyper_samples)
        u = np.random.rand(N);
        x_max = df_hyper_samples[self.x_max];
        x_min = df_hyper_samples[self.x_min];
        
        return pd.DataFrame({self.var_name : u*(x_max - x_min) + x_min})


class Uniform1DIndependentAnalyticVariedLimits(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x', 'y'], hyper_var_names=['x_min', 'x_max', 'y_min', 'y_max']):
        self.a, self.b = a,b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b}
        self.models = [Uniform1DAnalyticVariedLimits(var_names=[var_names[i]], 
                                                   hyper_var_names=[hyper_var_names[2*i], hyper_var_names[2*i + 1]], 
                                                   **kwargs) for i in range(len(self.var_names))]
    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def evaluate(self, data, params):
        """
        Evaluates the value of the *integrand* and not the integral
        """
        result = self.models[0].evaluate(data, params)
        for i in range(1,len(self.models)):
            result *= self.models[i].evaluate(data, params)
        return result
        
    def __call__(self, data, params):
        result = self.models[0](data, params)
        for i in range(1,len(self.models)):
            result *= self.models[i](data, params)
        return result

    def sample(self, df_hyper_samples, oversample=1):
        return pd.concat([model.sample(df_hyper_samples, oversample=oversample) for model in self.models])


class Uniform2DAnalyticVariedLimits(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x', 'y'], hyper_var_names=['x_min', 'x_max', 'y_min', 'y_max']):
        self.a = a
        self.b = b
        self.x_var_name = var_names[0]
        self.y_var_name = var_names[1]
        self.x_min = hyper_var_names[0];
        self.x_max = hyper_var_names[1];
        self.y_min = hyper_var_names[2];
        self.y_max = hyper_var_names[3];
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}
        
    def get_data(self, data, params):
        X_locations = data[self.x_var_name + '_mu_kernel'];
        X_scales    = data[self.x_var_name + '_sigma_kernel'];
        Y_locations = data[self.y_var_name + '_mu_kernel'];
        Y_scales    = data[self.y_var_name + '_sigma_kernel'];
        XY_rho      = data.get(self.x_var_name + '_rho_kernel', 1e-6);
        x_min       = params.get(self.x_min, self.a[0])
        x_max       = params.get(self.x_max, self.b[0])
        y_min       = params.get(self.y_min, self.a[1])
        y_max       = params.get(self.y_max, self.b[1])
        return X_locations, X_scales, x_min, x_max, Y_locations, Y_scales, y_min, y_max, XY_rho

    def evaluate(self, data, params):
        """
        Evaluates the value of the *integrand* and not the integral
        """
        x_min       = params.get(self.x_min, self.a[0])
        x_max       = params.get(self.x_max, self.b[0])
        y_min       = params.get(self.y_min, self.a[1])
        y_max       = params.get(self.y_max, self.b[1])
        #loglikes = jax.scipy.special.logsumexp( jnp.log(truncnorm(Xs, mu, sigma, low=a, high=b)) , axis=-1) - jnp.log(Xs.shape[-1])
        prob = box(data[self.x_var_name], x_min,x_max) * box(data[self.y_var_name], y_min,y_max) /((x_max - x_min)*(y_max - y_min))
        return prob
        
    def __call__(self, data, params):
        X_locations, X_scales, x_min, x_max, Y_locations, Y_scales, y_min, y_max, XY_rho = self.get_data(data, params);
        X_stddevs_outside = jnp.where(X_locations <= x_max, 
                                        jnp.where(X_locations <= x_min, 
                                            (x_min - X_locations)/X_scales, jnp.zeros_like(X_locations)),
                                                    (X_locations - x_max)/X_scales)
        Y_stddevs_outside = jnp.where(Y_locations <= y_max, 
                                        jnp.where(Y_locations <= y_min, 
                                            (y_min - Y_locations)/Y_scales, jnp.zeros_like(Y_locations)),
                                                    (Y_locations - y_max)/Y_scales)
        overlap = mvnorm2d(X_locations, Y_locations, X_scales, Y_scales, x_min, y_min, x_max, y_max, XY_rho)
        overlap /= (x_max - x_min)*(y_max - y_min)
        overlap /= mvnorm2d(X_locations, Y_locations, X_scales, Y_scales, self.a[0], self.a[1], self.b[0], self.b[1], XY_rho)
        new_overlap = jnp.where(X_stddevs_outside < 4, jnp.where(Y_stddevs_outside < 4, overlap, jnp.zeros_like(overlap)),jnp.zeros_like(overlap))
        return new_overlap

    def sample(self, df_hyper_samples, oversample=1):
        N = len(df_hyper_samples)
        u1 = np.random.rand(N);
        u2 = np.random.rand(N);
        x_min       = params.get(self.x_min, self.a[0])
        x_max       = params.get(self.x_max, self.b[0])
        y_min       = params.get(self.y_min, self.a[1])
        y_max       = params.get(self.y_max, self.b[1])
        return pd.DataFrame({self.x_var_name : u1*(x_max - x_min) + x_min, self.y_var_name : u2*(y_max - y_min) + y_min})


