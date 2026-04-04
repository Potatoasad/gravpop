import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from ..utils import *
from ..generic import *



def alpha_beta_max_to_mu_var_max(alpha, beta, amax):
    mu = alpha / (alpha + beta) * amax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax ** 2
    return mu, var, amax


def mu_var_max_to_alpha_beta_max(mu, var, amax):
    mu /= amax
    var /= amax ** 2
    alpha = (mu ** 2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta, amax


class GaussianIsotropicSpinOrientationsIID(SampledPopulationModel, SpinPopulationModel, InvertiblePopulationModel):
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

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob

    @property
    def n_draws(self):
        # 3 draws: u[:,0] selects component, u[:,1] -> z1, u[:,2] -> z2
        return 3

    def ppf(self, u, params):
        """
        Hierarchical inverse CDF for the joint (cos_tilt_1, cos_tilt_2) mixture.

        The joint is a 2-component mixture:
            xi  * TruncNorm(z1|1,sigma) * TruncNorm(z2|1,sigma)   [aligned]
          + (1-xi)/4                                                [isotropic]

        Within each component z1 and z2 are independent, so we sample:
            u[:,0] < xi  => Gaussian component; else Uniform component
            u[:,1]       => z1 via component-conditional PPF
            u[:,2]       => z2 via component-conditional PPF

        Note: the gradient of the detection efficiency w.r.t. xi is approximate
        (zero contribution from samples that switch component as xi varies), but
        the gradient w.r.t. sigma is exact for the Gaussian-component samples.

        Parameters
        ----------
        u      : shape (N_samp, 3)
        params : dict

        Returns
        -------
        dict {cos_tilt_1: ..., cos_tilt_2: ...}  each shape (N_samp,)
        """
        xi    = params[self.hyper_var_names[0]]
        sigma = params[self.hyper_var_names[1]]

        gaussian_comp = u[:, 0] < xi   # bool (N_samp,)

        # Truncated-normal PPF: mu=1, sigma=sigma, support [a, b]
        alpha  = (self.a - 1.0) / sigma
        beta   = (self.b - 1.0) / sigma
        p_low  = jax.scipy.stats.norm.cdf(alpha)
        p_high = jax.scipy.stats.norm.cdf(beta)

        def tn_ppf(u_val):
            return 1.0 + sigma * jax.scipy.special.ndtri(p_low + u_val * (p_high - p_low))

        z1 = jnp.where(gaussian_comp, tn_ppf(u[:, 1]), self.a + (self.b - self.a) * u[:, 1])
        z2 = jnp.where(gaussian_comp, tn_ppf(u[:, 2]), self.a + (self.b - self.a) * u[:, 2])
        return {self.var_names[0]: z1, self.var_names[1]: z2}


class GaussianIsotropicSpinOrientationsIIDWithMinimum(SampledPopulationModel, SpinPopulationModel):
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

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        z_min = params[self.hyper_var_names[2]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=self.a)
        prob *= xi_spin
        prob += jnp.heaviside(data[self.var_names[0]] - z_min, 0.5)*jnp.heaviside(data[self.var_names[1]] - z_min, 0.5)*(1-xi_spin)/4
        return prob

class GaussianIsotropicSpinOrientationsIIDExtended(SampledPopulationModel, SpinPopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['cos_tilt_1', 'cos_tilt_2'], hyper_var_names=['xi_spin','sigma_spin_1', 'sigma_spin_2', 'z_min_1', 'z_min_2'], a=-1, b=1):
        self.a = a
        self.b = b
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin_1 = params[self.hyper_var_names[1]]
        sigma_spin_2 = params[self.hyper_var_names[2]]
        z_min_1 = params[self.hyper_var_names[3]]
        z_min_2 = params[self.hyper_var_names[4]]
        prob  = truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin_1, high=self.b, low=z_min_1)
        prob *= truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin_2, high=self.b, low=z_min_2)
        prob *= xi_spin
        prob += ((1-xi_spin)) * jnp.heaviside(data[self.var_names[0]] - z_min_1, 0.5) * jnp.heaviside(data[self.var_names[1]] - z_min_2, 0.5) / ((1-z_min_1)*(1-z_min_2))
        return prob


class GaussianIsotropicSpinOrientationsIIDFullExtended(SampledPopulationModel, SpinPopulationModel):
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

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        sigma_spin = params[self.hyper_var_names[1]]
        z_min = params[self.hyper_var_names[2]]
        prob1  = xi_spin*(1/(1-z_min))*jnp.where(data[self.var_names[0]] < z_min, 0, 1) + (1-xi_spin)*truncnorm(data[self.var_names[0]], mu=1, sigma=sigma_spin, high=self.b, low=z_min)
        prob2  = xi_spin*(1/(1-z_min))*jnp.where(data[self.var_names[1]] < z_min, 0, 1) + (1-xi_spin)*truncnorm(data[self.var_names[1]], mu=1, sigma=sigma_spin, high=self.b, low=z_min)
        return prob1*prob2


class GaussianIsotropicSpinOrientationsFloating(SampledPopulationModel, SpinPopulationModel):
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

    @property
    def limits(self):
        return {var : [self.a, self.b] for i,var in enumerate(self.var_names)}

    
    def __call__(self, data, params):
        xi_spin = params[self.hyper_var_names[0]]
        mu_spin_1, mu_spin_2 = params[self.hyper_var_names[1]], params[self.hyper_var_names[3]]
        sigma_spin_1, sigma_spin_2 = params[self.hyper_var_names[2]], params[self.hyper_var_names[4]]
        prob  = truncnorm(data[self.var_names[0]], mu=mu_spin_1, sigma=sigma_spin_1, high=self.b, low=self.a)
        prob *= truncnorm(data[self.var_names[1]], mu=mu_spin_2, sigma=sigma_spin_2, high=self.b, low=self.a)
        prob *= xi_spin
        prob += (1-xi_spin)/4
        return prob



def sample_beta(alpha, beta, amax, size=1):
    """
    Generate samples from a scaled beta distribution.

    Parameters:
    alpha (np.array): Alpha parameter of the Beta distribution.
    beta (np.array): Beta parameter of the Beta distribution.
    amax (np.array): Maximum value to scale the Beta distribution to.
    size (int): Number of samples to generate (default is 1).

    Returns:
    np.array: Scaled samples from the Beta distribution.
    """
    # Ensure alpha, beta, and amax are numpy arrays
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    amax = np.asarray(amax)

    # Generate beta samples (values between 0 and 1)
    samples = np.random.beta(alpha, beta, size=size)
    
    # Scale samples by amax
    scaled_samples = samples * amax

    return scaled_samples


class BetaSpinMagnitudeIID(SampledPopulationModel, SpinPopulationModel, InvertiblePopulationModel):
    r"""
    Mixture of gaussian and isotropic distribution over spin orientations.
    Performs a monte carlo estimate of the population likelihood. 

    Event Level Parameters:         :math:`z_1=cos(\theta_1), z_2=cos(\theta_2)`
    Population Level Parameters:    :math:`\xi, \sigma` 

    .. math::
    
        P(z_1, z_2| \xi, \sigma) = \frac{1-\xi}{4} + \xi \left(  \mathcal{N}_{[-1,1]}(z_1 | \xi, \sigma) \mathcal{N}_{[-1,1]}(z_2 | \xi, \sigma) \right) 
    """
    def __init__(self, var_names=['a_1', 'a_2'], hyper_var_names=['mu_chi','sigma_chi','amax'],parameterization="mu_sigma", constraints=None):
        self.parameterization = parameterization
        self.converter = lambda x,y,z: (x,y,z)
        self.var_names = var_names
        self.constraints = constraints
        if self.constraints is not None:
            self.alpha_min_constraint = constraints['alpha'][0]
            self.alpha_max_constraint = constraints['alpha'][1]
            self.beta_min_constraint = constraints['beta'][0]
            self.beta_max_constraint = constraints['beta'][1]
        self.hyper_var_names = hyper_var_names
        if (len(self.hyper_var_names) == 2) and 'amax' not in self.hyper_var_names:
            self.hyper_var_names = self.hyper_var_names + ['amax']

        ### SIGMA_CHI IS ACTUALLY SIGMA_CHI**2 THIS IS UNFORTUNATE

        hyper_var_names_are_mu_sigma = (("mu" in self.hyper_var_names[0]) or (("sigma" in self.hyper_var_names[1])))

        if (parameterization in ("mu_sigma")) and hyper_var_names_are_mu_sigma:
            self.converter = lambda mu,sigma,amax=1 : mu_var_max_to_alpha_beta_max(mu, sigma, amax)

    @property
    def limits(self):
        return {var : [0, 1] for i,var in enumerate(self.var_names)}
    
    def __call__(self, data, params):
        amax = params.get(self.hyper_var_names[2], 1)
        alpha_chi, beta_chi, amax = self.converter(params[self.hyper_var_names[0]], params[self.hyper_var_names[1]], amax)
        if self.constraints is not None:
            is_alpha_within_constraint = (alpha_chi > self.alpha_min_constraint) & (alpha_chi < self.alpha_max_constraint)
            is_beta_within_constraint = (beta_chi > self.beta_min_constraint) & (beta_chi < self.beta_max_constraint)
            are_both_within_constraint = is_alpha_within_constraint & is_beta_within_constraint
        prob  = beta(data[self.var_names[0]], alpha=alpha_chi, beta=beta_chi, scale=amax)
        prob *= beta(data[self.var_names[1]], alpha=alpha_chi, beta=beta_chi, scale=amax)
        if self.constraints is not None:
            prob = jnp.where(are_both_within_constraint, prob, jnp.nan_to_num(-jnp.inf)*jnp.ones_like(prob))
        return prob

    def sample(self, df_hyper_samples, oversample=1):
        amax = df_hyper_samples.get(self.hyper_var_names[2], 1)
        alpha_chi, beta_chi, amax = self.converter(df_hyper_samples[self.hyper_var_names[0]], df_hyper_samples[self.hyper_var_names[1]], amax)
        chi_1 = sample_beta(alpha_chi, beta_chi, amax, size=oversample);
        chi_2 = sample_beta(alpha_chi, beta_chi, amax, size=oversample);
        return pd.DataFrame({self.var_names[0] : chi_1, self.var_names[1] : chi_2})

    def ppf(self, u, params, n_grid=500):
        """
        Grid-based inverse CDF for a scaled Beta distribution.

        For Chi ~ Beta(alpha, beta) on [0, amax] the PDF is evaluated on a
        dense grid, the CDF is built via cumulative trapezoid, and inverted
        with jnp.interp.  chi_1 and chi_2 are i.i.d., so u[:,0] and u[:,1]
        are used against the same CDF independently.

        Differentiable w.r.t. mu_chi / sigma_chi through the CDF grid values.

        Parameters
        ----------
        u      : shape (N_samp, 2)  -- u[:,0] for chi_1, u[:,1] for chi_2
        params : dict
        n_grid : int  -- number of grid points (default 500)

        Returns
        -------
        dict {a_1: ..., a_2: ...}  each shape (N_samp,)
        """
        amax = params.get(self.hyper_var_names[2], 1)
        alpha_chi, beta_chi, amax = self.converter(
            params[self.hyper_var_names[0]], params[self.hyper_var_names[1]], amax)

        # Avoid evaluating at the exact boundaries 0 and amax: when alpha < 1
        # or beta < 1 the Beta PDF diverges there, turning p_chi[0] into inf
        # and corrupting the cumulative sum.  Interior midpoints are safe.
        chi_grid = jnp.linspace(amax / (2 * n_grid), amax * (1 - 1 / (2 * n_grid)), n_grid)
        p_chi = beta(chi_grid, alpha_chi, beta_chi, scale=amax)

        dchi = jnp.diff(chi_grid)
        cdf = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum((p_chi[:-1] + p_chi[1:]) * 0.5 * dchi)
        ])
        cdf = cdf / jnp.clip(cdf[-1], 1e-30)

        chi_1 = jnp.interp(u[:, 0], cdf, chi_grid)
        chi_2 = jnp.interp(u[:, 1], cdf, chi_grid)
        return {self.var_names[0]: chi_1, self.var_names[1]: chi_2}

