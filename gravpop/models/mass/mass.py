from ..utils import *
from ..generic import AbstractPopulationModel
import jax.numpy as jnp
import jax

def _smoothing(masses, mmin, mmax, delta_m):
    shifted_mass = jnp.nan_to_num((masses - mmin) / delta_m, nan=0)
    shifted_mass = jnp.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    window = jax.scipy.special.expit(-exponent)
    return window*box(masses, mmin, mmax)

def smoothing(masses, mmin, mmax, delta_m):
    return jnp.where(delta_m > 0.0,  _smoothing(masses, mmin, mmax, delta_m), jnp.ones_like(masses))

def two_component_single(mass, alpha, lam, mmin, mmax, mpp, sigpp, gaussian_mass_maximum=100):
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob

class SmoothedTwoComponentPrimaryMassRatio(AbstractPopulationModel):
    def __init__(self, gaussian_mass_maximum=100, mmin_fixed=2, mmax_fixed=100, primary_mass_name="mass_1", mass_ratio_name="mass_ratio", normalization_shape=(1000,500)):
        self.gaussian_mass_maximum = gaussian_mass_maximum
        self.mmin_fixed = 2
        self.mmax_fixed = 100
        self.primary_mass_name = primary_mass_name
        self.mass_ratio_name = mass_ratio_name
        self.m1s = jnp.linspace(mmin_fixed, mmax_fixed, normalization_shape[0])
        self.qs = jnp.linspace(0, 1, normalization_shape[1])
        self.m1s_grid, self.qs_grid = jnp.meshgrid(self.m1s, self.qs)
        #print("""Note: SmoothedTwoComponentPrimaryMassRatio is an unnormalized distribution. 
        #    Be wary when using these for infering merger rates. 
        #    In addition, this model might have a different primary mass marginal due to this lack of normalization in q""")

    def norm_p_q(self, data, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )
        norms = jnp.where(
            jnp.array(delta_m) > 0,
            jax.scipy.integrate.trapezoid(jnp.nan_to_num(p_q), self.qs, axis=0),
            jnp.ones(self.m1s.shape),
        )

        return jnp.interp(data[self.primary_mass_name], self.m1s, norms)

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = jnp.where(jnp.array(delta_m) > 0, jax.scipy.integrate.trapezoid(p_m, self.m1s), 1)
        return norm

    def __call__(self, data, params):
        # Get params
        alpha = params['alpha']
        lam = params['lam']
        mmin = params.get('mmin', self.mmin_fixed)
        mmax = params.get('mmax', self.mmax_fixed)
        beta = params['beta']
        mpp = params['mpp']
        sigpp = params['sigpp']
        delta_m = params.get("delta_m", 0)

        # Compute primary mass unnormalized distribution with smoothing
        p_m1 = two_component_single(data[self.primary_mass_name],  alpha, lam, mmin, mmax, mpp, sigpp, gaussian_mass_maximum=self.gaussian_mass_maximum)
        p_m1 *= smoothing(data[self.primary_mass_name], mmin, mmax, delta_m)

        p_m1_grid = two_component_single(self.m1s, alpha, lam, mmin, mmax, mpp, sigpp, gaussian_mass_maximum=self.gaussian_mass_maximum)
        p_m1_grid *= smoothing(self.m1s, mmin, mmax, delta_m)
        norm = jnp.where(delta_m > 0, jax.scipy.integrate.trapezoid(jnp.nan_to_num(p_m1_grid), self.m1s), 1)
        p_m1 /= norm

        # Compute mass_ratio unnormalized distribution with smoothing
        p_q = powerlaw(data[self.mass_ratio_name], beta, 1, mmin / data[self.primary_mass_name])
        p_q *= smoothing(
            data[self.primary_mass_name] * data[self.mass_ratio_name],
            mmin=mmin,
            mmax=data[self.primary_mass_name],
            delta_m=delta_m,
        )
        p_q /= self.norm_p_q(data=data, beta=beta, mmin=mmin, delta_m=delta_m)

        return jnp.nan_to_num(p_m1 * p_q, nan=0)
