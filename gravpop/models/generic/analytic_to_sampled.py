from .abstract import *
import numpy as np
import jax.numpy as jnp
import pandas as pd

def make_list_if_not(x):
    if not isinstance(x, list):
        return [x]

class ProductModel(AbstractPopulationModel):
    def __init__(self, models):
        self.var_names = []
        self.hyper_var_names = []
        self.a = []; self.b = [];
        for model in models:
            self.var_names = self.var_names + model.var_names
            self.hyper_var_names = self.hyper_var_names + model.hyper_var_names
            self.a = self.a + make_list_if_not(model.a);
            self.b = self.b + make_list_if_not(model.b);

        self.models = models

    @property
    def limits(self):
        lims = {}
        for model in self.models:
            lims = {**lims, **model.limits}
        return lims

    def __call__(self, data, params):
        result  = self.models[0](data, params);
        for i in range(1,len(self.models)):
            result *= self.models[i](data, params);
        return result

    def evaluate(self, data, params):
        result  = self.models[0].evaluate(data, params);
        for i in range(1,len(self.models)):
            result *= self.models[i].evaluate(data, params);
        return result

    def sample(self, df_hyper_samples, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        series_s = []
        for model in self.models:
            series_s.append(model.sample(df_hyper_samples, **kwargs).reset_index(drop=True))

        N = len(df_hyper_samples)
        return pd.concat(series_s, axis=1)

class Mixture2D(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, model1, model2, mixture_hyper_var_name="eta_spin", var_names=None, hyper_var_names=None):
        if var_names is not None:
            self.var_names = model1.var_names
        else:
            self.var_names = model1.var_names
        self.hyper_var_names = model1.hyper_var_names + model2.hyper_var_names + [mixture_hyper_var_name]
        self.a = model1.a; self.b = model1.b;

        self.mixture_hyper_var_name = mixture_hyper_var_name

        self.models = [model1, model2]

    @property
    def limits(self):
        return self.models[0].limits

    def __call__(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs).reset_index(drop=True)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs).reset_index(drop=True)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] >= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)

class SampledFromAnalytic(SampledPopulationModel):
    def __init__(self, analytic_model):
        self.analytic_model = analytic_model
        self.__dict__.update(analytic_model.__dict__)
        
    def __call__(self, data, params):
        return self.analytic_model.evaluate(data, params)
        
    def evaluate(self, data, params):
        return self.analytic_model.evaluate(data, params)

class FixedParameters(AnalyticPopulationModel):
    def __init__(self, model, fixed_parameters : dict):
        self.model = model
        self.fixed_parameters = fixed_parameters
        for attr, value in model.__dict__.items():
            setattr(self, attr, value)

    @property
    def limits(self):
        return self.model.limits

    def evaluate(self, data, params):
        return self.model.evaluate(data, {**params, **self.fixed_parameters})

    def __call__(self, data, params):
        return self.model(data, {**params, **self.fixed_parameters})

    def sample(self, df_hyper_samples, **kwargs):
        df2 = df_hyper_samples.copy()
        for p,v in self.fixed_parameters.items():
            df2[p] = v
        return self.model.sample(df2, **kwargs)