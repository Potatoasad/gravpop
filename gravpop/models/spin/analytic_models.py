import jax
import jax.numpy as jnp
from ..utils import *
from ..generic import *
from ..sample import *


class GaussianIsotropicSpinOrientationsIIDAnalytic(AnalyticPopulationModel, SpinPopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','sigma_spin'], a=-1, b=1):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.models = [TruncatedGaussian1DAnalytic(a=a, b=b, var_names=[var_names[i]], hyper_var_names=['mu_spin', hyper_var_names[1]]) for i in range(len(var_names))]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def evaluate(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob
    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        new_params = {**params, 'mu_spin':1}
        prob  = self.models[0](data, new_params)           #truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= self.models[1](data, new_params)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob

class GaussianIsotropicSpinOrientationsFloatingAnalytic(AnalyticPopulationModel, SpinPopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','mu_spin_1', 'sigma_spin_1', 'mu_spin_2', 'sigma_spin_2'], a=-1, b=1):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.models = [TruncatedGaussian1DAnalytic(a=a, b=b, var_names=[var_names[i]], hyper_var_names=[hyper_var_names[2*i + 1], hyper_var_names[2*i + 2]]) for i in range(len(var_names))]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def evaluate(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        mu_spin_1, mu_spin_2 = params[self.hyper_var_names[1]], params[self.hyper_var_names[3]]
        sigma_spin_1, sigma_spin_2 = params[self.hyper_var_names[2]], params[self.hyper_var_names[4]]
        prob  = truncnorm(data[self.var_names[0]], mu=mu_spin_1, sigma=sigma_spin_1, high=self.b, low=self.a)
        prob *= truncnorm(data[self.var_names[1]], mu=mu_spin_2, sigma=sigma_spin_2, high=self.b, low=self.a)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob
    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        mu_spin_1, mu_spin_2 = params[self.hyper_var_names[1]], params[self.hyper_var_names[3]]
        sigma_spin_1, sigma_spin_2 = params[self.hyper_var_names[2]], params[self.hyper_var_names[4]]
        #new_params = {**params, 'mu_spin':1}
        prob  = self.models[0](data, new_params)           #truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= self.models[1](data, new_params)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob



class IIDTruncatedGaussian1DAnalytic(AnalyticPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['mu', 'sigma']):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b, 'hyper_var_names' : hyper_var_names}
        self.models = [TruncatedGaussian1DAnalytic(var_names=[var_name], **kwargs) for var_name in self.var_names]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

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


class MixtureTruncated1DAnalyticFixed0(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['sigma_0', 'mu', 'sigma', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.eta_variable = hyper_var_names[-1]
        kwargs = {'a' : a, 'b' : b, 'hyper_var_names' : hyper_var_names}

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', self.hyper_var_names[0]]
        comp2_hyper_var_names = [self.hyper_var_names[1], self.hyper_var_names[2]]

        self.models = [TruncatedGaussian1DAnalytic(var_names=var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian1DAnalytic(var_names=var_names, hyper_var_names=comp2_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        eta = params[self.eta_variable]
        result  = eta * self.models[0](data, params)
        result += (1-eta) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        eta = params[self.eta_variable]
        result  = eta * self.models[0].evaluate(data, params)
        result += (1-eta) * self.models[1].evaluate(data, params)
        return result

class IIDTruncatedGaussian1DAnalyticMixture(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['x'], hyper_var_names=['sigma_0', 'mu', 'sigma', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b, 'hyper_var_names' : hyper_var_names}
        self.models = [MixtureTruncated1DAnalyticFixed0(var_names=[var_name], **kwargs) for var_name in self.var_names]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

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


class TruncatedGaussian1DIndependentAnalytic(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2']):
        self.a, self.b = a,b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b}
        self.models = [TruncatedGaussian1DAnalytic(var_names=[var_names[i]], 
                                                   hyper_var_names=[hyper_var_names[2*i], hyper_var_names[2*i + 1]], 
                                                   **kwargs) for i in range(len(self.var_names))]
    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

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

    def sample(self, df_hyper_samples, oversample=1):
        return ppd_truncUncorrelatedanalytic(self, df_hyper_samples, oversample=oversample)

class TruncatedGaussian1DIndependentAnalyticVariedLimits(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['mu_chi_1', 'sigma_chi_1', 'chi_1_min', 'chi_1_max', 
                                                                            'mu_chi_2', 'sigma_chi_2', 'chi_2_min', 'chi_2_max']):
        self.a, self.b = a,b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        kwargs = {'a' : a, 'b' : b}
        self.models = [TruncatedGaussian1DAnalyticVariedLimits(var_names=[var_names[i]], 
                                                   hyper_var_names=[hyper_var_names[4*i], hyper_var_names[4*i + 1], hyper_var_names[4*i + 2], hyper_var_names[4*i + 3]], 
                                                   **kwargs) for i in range(len(self.var_names))]
    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

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

    def sample(self, df_hyper_samples, oversample=1):
        return ppd_truncUncorrelatedanalytic(self, df_hyper_samples, oversample=oversample)


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



class TruncatedGaussian2DMixtureZeroAndFloatingIID(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['sigma_chi_at_0','mu_chi', 'sigma_chi', 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'rho_zero_fixed']
        comp2_hyper_var_names = self.hyper_var_names[1:3]*2 + ['rho_zero_fixed']
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian2DAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=[a,a], b=[b,b]),
                       TruncatedGaussian2DAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=[a,a], b=[b,b])]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        params['rho_zero_fixed'] = 1e-6    # Small non-zero number to prevent nan gradients
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        params['rho_zero_fixed'] = 1e-6    # Small non-zero number to prevent nan gradients
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result

    def sample(self, df_hyper_samples_in, oversample=1, **kwargs):
        kwargs['oversample'] = oversample
        df_hyper_samples = df_hyper_samples_in.copy()
        df_hyper_samples['mu_zero_spin_1d_fixed'] = 0
        df_hyper_samples['rho_zero_fixed'] = 0
        series_1 = self.models[0].sample(df_hyper_samples, **kwargs)
        series_2 = self.models[1].sample(df_hyper_samples, **kwargs)
        N = len(df_hyper_samples)
        sampled_mixture = np.hstack([(df_hyper_samples[self.mixture_hyper_var_name] <= np.random.rand(N)) for _ in range(oversample)])
        d = {col : ((sampled_mixture * series_1[col]) + ((1 - sampled_mixture) * series_2[col])) for col in series_1.columns}
        return pd.DataFrame(d)


class GaussianIsotropicSpinOrientationsIIDAnalytic2D(AnalyticPopulationModel, SpinPopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','sigma_spin'], a=-1, b=1):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.model = TruncatedGaussian2DAnalytic(a=[a,a], b=[b,b], var_names=var_names, hyper_var_names=['mu_spin', hyper_var_names[1], 'mu_spin', hyper_var_names[1], 'rho_zero_fixed'])

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def evaluate(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob
        
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        new_params = {**params, 'mu_spin':1, 'rho_zero_fixed':1e-6}
        prob  = self.model(data, new_params)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob


class GaussianIsotropicSpinOrientationsIIDAnalytic2DWithMinimum(AnalyticPopulationModel, SpinPopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','sigma_spin', 'z_min'], a=-1, b=1):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.model = TruncatedGaussian2DAnalyticLimits(a=[a,a], b=[b,b], 
                                                var_names=var_names, 
                                                hyper_var_names=['mu_spin', hyper_var_names[1], 
                                                                 'mu_spin', hyper_var_names[1], 'rho_zero_fixed',
                                                                 'z_min', 'z_max_fixed_to_b', 'z_min', 'z_max_fixed_to_b'])

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def evaluate(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        z_min = params[self.hyper_var_names[2]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=z_min)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=z_min)
        prob *= xi_spin
        prob += (1-xi_spin)/((1-z_min)**2)
        return prob
        
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        z_min = params[self.hyper_var_names[2]]
        new_params = {**params, 'mu_spin':1, 'rho_zero_fixed':1e-6,  'z_max_fixed_to_b' : self.b}
        prob  = self.model(data, new_params)
        prob *= xi_spin
        prob += (1-xi_spin)/((1-z_min)**2)
        return prob

class GaussianIsotropicSpinOrientationsIIDAnalytic1DWithMinimum(AnalyticPopulationModel, SpinPopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','sigma_spin', 'z_min'], a=-1, b=1):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.models = [TruncatedGaussian1DAnalyticVariedLimits(a=a, b=b, var_names=[var_names[i]], hyper_var_names=['mu_spin', hyper_var_names[1], hyper_var_names[2], 'z_max_fixed_to_b']) for i in range(len(var_names))]

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def evaluate(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        z_min = params[self.hyper_var_names[2]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=z_min)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=z_min)
        prob *= xi_spin
        prob += (1-xi_spin)/((1-z_min)**2)
        return prob
    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        z_min = params[self.hyper_var_names[2]]
        new_params = {**params, 'mu_spin':1, 'z_max_fixed_to_b' : self.b}
        prob  = self.models[0](data, new_params)           #truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= self.models[1](data, new_params)
        prob *= xi_spin
        prob += (1-xi_spin)/((1-z_min)**2)
        return prob


class TruncatedGaussian2DMixtureZeroAndFloatingIndependent(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['sigma_chi_at_0','mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'rho_chi' 'eta_spin']):
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.a = a; self.b=b;
        kwargs = {'a' : a, 'b' : b}
        comp1_var_names = [var_names[0]]
        comp2_var_names = [var_names[1]]

        comp1_hyper_var_names = ['mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'mu_zero_spin_1d_fixed', self.hyper_var_names[0], 'rho_zero_fixed']
        comp2_hyper_var_names = self.hyper_var_names[1:6]
        self.mixture_hyper_var_name = self.hyper_var_names[-1]

        self.models = [TruncatedGaussian2DAnalytic(var_names=self.var_names, hyper_var_names=comp1_hyper_var_names, a=a, b=b),
                       TruncatedGaussian2DAnalytic(var_names=self.var_names, hyper_var_names=comp2_hyper_var_names, a=a, b=b)]

    @property
    def limits(self):
        return {var : [self.a[i], self.b[i]] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        params['rho_zero_fixed'] = 1e-6
        result  =      params[self.mixture_hyper_var_name]  * self.models[0](data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1](data, params)
        return result

    def evaluate(self, data, params):
        params['mu_zero_spin_1d_fixed'] = 0.0
        params['rho_zero_fixed'] = 1e-6
        result  =      params[self.mixture_hyper_var_name]  * self.models[0].evaluate(data, params)
        result += (1 - params[self.mixture_hyper_var_name]) * self.models[1].evaluate(data, params)
        return result


class TruncatedGaussian1DMixtureZeroAndFloatingIndependent(AnalyticPopulationModel, SpinPopulationModel):
    def __init__(self, a, b, var_names=['chi_1', 'chi_2'], hyper_var_names=['sigma_chi_at_0','mu_chi_1', 'sigma_chi_1', 'mu_chi_2', 'sigma_chi_2', 'eta_spin']):
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







