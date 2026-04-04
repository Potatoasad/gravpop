from .abstract import *
import numpy as np
import jax.numpy as jnp
import pandas as pd

def make_list_if_not(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x

class ProductModel(InvertiblePopulationModel):
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

    @property
    def n_draws(self):
        return sum(m.n_draws for m in self.models)

    def ppf(self, u, params):
        """
        PPF for a product of independent models.

        Sub-models are independent, so their draws are non-overlapping slices
        of u.  Each sub-model receives exactly its own n_draws columns.

        Parameters
        ----------
        u : shape (N_samp, n_draws)  where n_draws = sum(m.n_draws for m in models)

        Returns
        -------
        dict  -- merged {var_name: array} from all sub-models
        """
        result = {}
        offset = 0
        for model in self.models:
            nd = model.n_draws
            u_slice = u[:, offset] if nd == 1 else u[:, offset:offset + nd]
            result.update(model.ppf(u_slice, params))
            offset += nd
        return result

class Mixture2D(AnalyticPopulationModel, SpinPopulationModel, InvertiblePopulationModel):
    # InvertiblePopulationModel already inherits AbstractPopulationModel;
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

    @property
    def n_draws(self):
        # 1 draw to select the component, then the inner draws are shared
        # between both branches (same u passed to both, jnp.where selects).
        return 1 + max(self.models[0].n_draws, self.models[1].n_draws)

    def ppf(self, u, params):
        """
        Generic PPF for a 2-component mixture.

        Draw layout:
            u[:, 0]          -- component selector: < mixture_weight → model[0]
            u[:, 1:]         -- inner draws, passed to BOTH model ppfs

        Both branches are evaluated for every sample (JAX traces both paths);
        jnp.where then picks the right value per sample.  The inner draws are
        reused across branches, which is valid because exactly one branch
        contributes to the final output for each sample.

        This correctly preserves the joint distribution across all var_names
        within each component: samples inside a component are drawn jointly
        from that component's ppf, not from independent marginals.

        Note: the gradient w.r.t. the mixture weight flows through
        jnp.where's two branches but not through the discrete switching
        boundary (zero-measure event).  Gradients w.r.t. all other
        hyperparameters are exact.

        Parameters
        ----------
        u : shape (N_samp, n_draws)

        Returns
        -------
        dict {var_name: array of shape (N_samp,)}
        """
        import jax.numpy as jnp
        xi   = params[self.mixture_hyper_var_name]
        comp = u[:, 0] < xi           # True → model[0], False → model[1]
        u_inner = u[:, 1:]            # shared inner draws

        n0, n1 = self.models[0].n_draws, self.models[1].n_draws
        u0 = u_inner[:, 0] if n0 == 1 else u_inner[:, :n0]
        u1 = u_inner[:, 0] if n1 == 1 else u_inner[:, :n1]

        out_0 = self.models[0].ppf(u0, params)
        out_1 = self.models[1].ppf(u1, params)

        return {
            var: jnp.where(comp, out_0[var], out_1[var])
            for var in self.var_names
        }

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