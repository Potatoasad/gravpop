from ..utils import *
from ..generic import AbstractPopulationModel
import jax.numpy as jnp
import jax

def smoothing(masses, mmin, mmax, delta_m):
    shifted_mass = jnp.nan_to_num((masses - mmin) / delta_m, nan=0)
    shifted_mass = jnp.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    window = jax.scipy.special.expit(-exponent)
    return window*box(masses, mmin, mmax)

def two_component_single(mass, alpha, lam, mmin, mmax, mpp, sigpp, gaussian_mass_maximum=100):
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob

class SmoothedTwoComponentPrimaryMassRatio(AbstractPopulationModel):
    def __init__(self, gaussian_mass_maximum=100, mmin_fixed=2, mmax_fixed=100):
        self.gaussian_mass_maximum = gaussian_mass_maximum
        self.mmin_fixed = 2
        self.mmax_fixed = 100
        print("""Note: SmoothedTwoComponentPrimaryMassRatio is an unnormalized distribution. 
            Be wary when using these for infering merger rates. 
            In addition, this model might have a different primary mass marginal due to this lack of normalization in q""")

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
        p_m1 = two_component_single(data["mass_1"],  alpha, lam, mmin, mmax, mpp, sigpp, gaussian_mass_maximum=self.gaussian_mass_maximum)
        p_m1 *= smoothing(data['mass_1'], mmin, mmax, delta_m)

        # Compute mass_ratio unnormalized distribution with smoothing
        p_q = powerlaw(data["mass_ratio"], beta, 1, mmin / data["mass_1"])
        p_q *= smoothing(
            data["mass_1"] * data["mass_ratio"],
            mmin=mmin,
            mmax=data["mass_1"],
            delta_m=delta_m,
        )

        return jnp.nan_to_num(p_m1 * p_q)
