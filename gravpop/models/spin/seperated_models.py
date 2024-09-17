import jax
import jax.numpy as jnp
from ..generic import *
from .analytic_models import TruncatedGaussian1DIndependentAnalytic
import pandas as pd
import numpy as np

class MixtureTruncatedGaussian1DFixedZero(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1'], hyper_var_names=['mu_chi_a', 'sigma_chi_a', 'sigma_chi_b', 'eta']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.eta_variable = self.hyper_var_names[-1]
        kwargs = {'a' : a, 'b' : b}
        self.models = [TruncatedGaussian1DAnalytic(var_names=[var_names[0]], 
                                                   hyper_var_names=[hyper_var_names[0], hyper_var_names[1]], 
                                                   **kwargs),
                       TruncatedGaussian1DAnalytic(var_names=[var_names[0]], 
                                                   hyper_var_names=['mu_zero_spin_1d_fixed', hyper_var_names[2]], 
                                                   **kwargs),]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.eta_variable]  * self.models[0](data, params)
        result += (1 - params[self.eta_variable]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.eta_variable]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.eta_variable]) * self.models[1].evaluate(data, params)
        return result



class TruncatedGaussian1DMixtureChi1StandardChi2(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1_a', 'sigma_chi_1_a', 'sigma_chi_1_b', 'eta_1', 'mu_chi_2', 'sigma_chi_2']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b}
        chi_1_var_names = [var_names[0]]
        chi_2_var_names = [var_names[1]]

        chi_1_hyper_var_names = self.hyper_var_names[0:4]
        chi_2_hyper_var_names = self.hyper_var_names[4:6]

        self.models = [MixtureTruncatedGaussian1DFixedZero(var_names=chi_1_var_names, hyper_var_names=chi_1_hyper_var_names, a=a[0], b=b[0]),
                       TruncatedGaussian1DAnalytic(var_names=chi_2_var_names, hyper_var_names=chi_2_hyper_var_names, a=a[1], b=b[1])]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        result = self.models[0](data, params)
        for i in range(1,len(self.models)):
            result *= self.models[i](data, params)
        return result

    def evaluate(self, data, params):
        result = self.models[0].evaluate(data, params)
        for i in range(1,len(self.models)):
            result *= self.models[i].evaluate(data, params)
        return result



class TruncatedGaussian1DMixtureZeroAndFloating(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['sigma_chi_at_0', 'mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'mu_zero_spin_1d_fixed', self.hyper_var_names[0]]
        comp2_hyper_var_names = self.hyper_var_names[1:5]
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        df_hyper_samples['mu_zero_spin_1d_fixed'] = 0
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)


class TruncatedGaussian1DMixtureZeroAndFloatingSpike(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', 'sigma_chi_at_0', 'mu_zero_spin_1d_fixed', 'sigma_chi_at_0']
        comp2_hyper_var_names = self.hyper_var_names[0:4]
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        params['sigma_chi_at_0'] = 1e-6
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        params['sigma_chi_at_0'] = 1e-6
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        df_hyper_samples['mu_zero_spin_1d_fixed'] = 0
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)

class TruncatedGaussian1DMixtureZeroAndUniform(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['sigma_chi_at_0','eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'mu_zero_spin_1d_fixed', self.hyper_var_names[0]]
        comp2_hyper_var_names = []
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       Uniform2DAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=[a,a], b=[b,b])]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        df_hyper_samples['mu_zero_spin_1d_fixed'] = 0
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)

class TruncatedGaussian1DMixtureZeroFloatingAndUniform(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2','eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = [self.hyper_var_names[0], self.hyper_var_names[1], self.hyper_var_names[2], self.hyper_var_names[3]]
        comp2_hyper_var_names = []
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       Uniform2DAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=[a,a], b=[b,b])]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)

class TruncatedGaussian2DMixtureZeroFloatingAndUniform(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2','rho_chi', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = [self.hyper_var_names[0], self.hyper_var_names[1], self.hyper_var_names[2], self.hyper_var_names[3], self.hyper_var_names[4]]
        comp2_hyper_var_names = []
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian2DAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       Uniform2DAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)


class TruncatedGaussian2DWithLabelSwitching(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2','rho_chi', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = [self.hyper_var_names[0], self.hyper_var_names[1], self.hyper_var_names[2], self.hyper_var_names[3], self.hyper_var_names[4]]
        comp2_hyper_var_names = []
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian2DAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian2DAnalytic(var_names=self.var_names[::-1], hyper_var_names=comp1_hyper_var_names, a=a[::-1], b=b[::-1])]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)


class TruncatedGaussian1DWithLabelSwitching(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = [self.hyper_var_names[0], self.hyper_var_names[1], self.hyper_var_names[2], self.hyper_var_names[3]]
        comp2_hyper_var_names = []
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names[::-1], hyper_var_names=comp1_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)


class TruncatedGaussian1DMixtureZeroAndFloatingIID(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['sigma_chi_at_0','mu_chi', 'sigma_chi', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'mu_zero_spin_1d_fixed', self.hyper_var_names[0]]
        comp2_hyper_var_names = self.hyper_var_names[1:3]*2
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian1DIndependentAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        df_hyper_samples['mu_zero_spin_1d_fixed'] = 0
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)

