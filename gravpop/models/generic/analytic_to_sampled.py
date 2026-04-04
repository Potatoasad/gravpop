from .abstract import *
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd


# ── Differentiable mixture selection ────────────────────────────────────────
#
# Problem: jnp.where(comp, out_0, out_1) correctly selects samples per
# component, but d(where)/d(xi) = 0 because comp = u < xi is a Heaviside.
#
# True gradient by the distributional (density-ratio) argument:
#   d/dxi  E_{mixture}[f] = E_{comp0}[f] - E_{comp1}[f]
#
# Implemented via custom_vjp: the forward pass is correct (exact joint
# samples); the backward pass for xi uses the distributional gradient
#   g_xi = sum_i  g_i · (out_0_i - out_1_i)
# which is an unbiased Monte Carlo estimate of E_0[f] - E_1[f] when g_i is
# the upstream gradient d(loss)/d(theta_i).
#
# Gradients w.r.t. out_0 and out_1 (→ each component's hyperparameters)
# are routed through the correct branch via the comp mask.
#
@jax.custom_vjp
def _mixture_select(out_0, out_1, comp, xi):
    """
    Per-sample selection from two pre-computed component outputs.

    Parameters
    ----------
    out_0 : (N_samp, D) -- samples from component 0
    out_1 : (N_samp, D) -- samples from component 1
    comp  : (N_samp,) bool -- True  → pick out_0
    xi    : scalar -- mixture weight (passed so its gradient is tracked)

    Returns
    -------
    (N_samp, D) -- selected samples
    """
    return jnp.where(comp[:, None], out_0, out_1)

def _mixture_select_fwd(out_0, out_1, comp, xi):
    selected = _mixture_select(out_0, out_1, comp, xi)
    return selected, (out_0, out_1, comp)

def _mixture_select_bwd(res, g):
    out_0, out_1, comp = res
    c = comp[:, None].astype(g.dtype)          # (N, 1) float mask

    # Gradient w.r.t. out_0: only the comp=True rows contributed
    g_out_0 = g * c

    # Gradient w.r.t. out_1: only the comp=False rows contributed
    g_out_1 = g * (1.0 - c)

    # Gradient w.r.t. comp: discrete, not differentiable
    g_comp = jnp.zeros_like(comp, dtype=g.dtype)

    # Gradient w.r.t. xi: distributional gradient
    #   d/dxi E_mix[f] = E_0[f] - E_1[f]
    #   ≈ (1/N) sum_i  g_i · (out_0_i - out_1_i)   (Monte Carlo estimate)
    g_xi = jnp.sum(g * (out_0 - out_1))

    return g_out_0, g_out_1, g_comp, g_xi

_mixture_select.defvjp(_mixture_select_fwd, _mixture_select_bwd)

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
        Generic PPF for a 2-component mixture with exact xi gradient.

        Draw layout:
            u[:, 0]    component selector — u[:,0] < xi  →  model[0]
            u[:, 1:]   inner draws, passed to BOTH component ppfs

        Both component ppfs are evaluated for every sample (JAX traces both
        paths) and _mixture_select picks the correct one per sample.  The
        inner draws are reused across branches — valid because exactly one
        branch contributes per sample.

        Joint accuracy: samples within each component are drawn jointly from
        that component's ppf, preserving the within-component joint structure
        (no independent-marginal approximation).

        Gradient w.r.t. xi: provided by _mixture_select's custom_vjp using
        the distributional gradient  g_xi = sum_i g_i · (out_0_i - out_1_i),
        which is a Monte Carlo estimate of  d/dxi E[f] = E_0[f] - E_1[f].

        Gradient w.r.t. all other params: exact — routed through the active
        component via the comp mask in _mixture_select's custom_vjp.

        Parameters
        ----------
        u : shape (N_samp, n_draws)

        Returns
        -------
        dict {var_name: array of shape (N_samp,)}
        """
        xi      = params[self.mixture_hyper_var_name]
        comp    = u[:, 0] < xi        # True → model[0]
        u_inner = u[:, 1:]            # shared inner draws

        n0, n1 = self.models[0].n_draws, self.models[1].n_draws
        u0 = u_inner[:, 0] if n0 == 1 else u_inner[:, :n0]
        u1 = u_inner[:, 0] if n1 == 1 else u_inner[:, :n1]

        out_0 = self.models[0].ppf(u0, params)   # {var: (N,)}
        out_1 = self.models[1].ppf(u1, params)   # {var: (N,)}

        # Stack vars into (N, D) so _mixture_select operates on one array
        out_0_arr = jnp.stack([out_0[v] for v in self.var_names], axis=-1)
        out_1_arr = jnp.stack([out_1[v] for v in self.var_names], axis=-1)

        selected = _mixture_select(out_0_arr, out_1_arr, comp, xi)

        return {v: selected[:, i] for i, v in enumerate(self.var_names)}

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