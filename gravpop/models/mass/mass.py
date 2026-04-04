from ..utils import *
from ..generic import *
import jax.numpy as jnp
import jax


def _cumtrapz(y, x):
    """Cumulative trapezoidal integration; returns array same length as x, starting at 0."""
    dx = jnp.diff(x)
    increments = (y[:-1] + y[1:]) * 0.5 * dx
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(increments)])

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

class SmoothedTwoComponentPrimaryMassRatio(SampledPopulationModel, MassPopulationModel, InvertiblePopulationModel):
    def __init__(self, gaussian_mass_maximum=100, mmin_fixed=2, mmax_fixed=100, 
        primary_mass_name=None, mass_ratio_name=None, 
        normalization_shape=(1000,500), var_names=['mass_1', 'mass_ratio'],
        hyper_var_names=['alpha', 'beta', 'lam', 'mpp', 'sigpp', 'delta_m', 'mmin', 'mmax']):
        var_names[0] = primary_mass_name or var_names[0]
        var_names[1] = mass_ratio_name or var_names[1]
        self.hyper_var_names = hyper_var_names
        self.var_names = var_names
        self.gaussian_mass_maximum = gaussian_mass_maximum
        self.mmin_fixed = 2
        self.mmax_fixed = 100
        self.primary_mass_name = self.var_names[0]
        self.mass_ratio_name = self.var_names[1]
        self.m1s = jnp.linspace(mmin_fixed, mmax_fixed, normalization_shape[0])
        self.qs = jnp.linspace(mmin_fixed/mmax_fixed, 1, normalization_shape[1])
        self.m1s_grid, self.qs_grid = jnp.meshgrid(self.m1s, self.qs)
        self.mass_ratio_norm_clip_threshold = 1e-16
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

        result = jnp.interp(data[self.primary_mass_name], self.m1s, norms)
        return jnp.clip(result, self.mass_ratio_norm_clip_threshold)  # For preventing 1/0 at masses that are below mmin

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = jnp.where(jnp.array(delta_m) > 0, jax.scipy.integrate.trapezoid(p_m, self.m1s), 1)
        return norm

    def __call__(self, data, params):
        # Get params
        alpha = params[self.hyper_var_names[0]]
        lam = params[self.hyper_var_names[2]]
        mmin = params.get(self.hyper_var_names[6], self.mmin_fixed)
        mmax = params.get(self.hyper_var_names[7], self.mmax_fixed)
        beta = params[self.hyper_var_names[1]]
        mpp = params[self.hyper_var_names[3]]
        sigpp = params[self.hyper_var_names[4]]
        delta_m = params.get(self.hyper_var_names[5], 0)

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

    def ppf(self, u, params):
        """
        Hierarchical inverse CDF for the joint (m1, q) distribution.

        Because p(m1, q | Lambda) = p(m1 | Lambda) * p(q | m1, Lambda) we sample
        in two steps using two independent uniform draws u[:,0] and u[:,1].

        Step 1 — m1 marginal (no analytic inverse, mixture + smoothing):
            Build CDF on self.m1s grid via cumulative trapezoid, invert with interp.

        Step 2 — conditional q | m1 (power-law * smoothing, no analytic inverse):
            For all N_samp sampled m1 values simultaneously, evaluate p(q | m1_i)
            on self.qs using broadcasting: shape (N_samp, N_q).
            Build per-sample CDFs, invert each with interp via vmap.

        Both steps are differentiable w.r.t. hyperparams because jnp.interp is
        differentiable w.r.t. its fp (y-value) argument, through which the CDF
        values — functions of params — propagate gradients.

        Parameters
        ----------
        u      : shape (N_samp, 2)   -- u[:,0] for m1, u[:,1] for q
        params : dict with keys matching hyper_var_names

        Returns
        -------
        dict {primary_mass_name: m1_samples, mass_ratio_name: q_samples}
             both arrays of shape (N_samp,)
        """
        alpha   = params[self.hyper_var_names[0]]
        beta    = params[self.hyper_var_names[1]]
        lam     = params[self.hyper_var_names[2]]
        mpp     = params[self.hyper_var_names[3]]
        sigpp   = params[self.hyper_var_names[4]]
        delta_m = params.get(self.hyper_var_names[5], 0)
        mmin    = params.get(self.hyper_var_names[6], self.mmin_fixed)
        mmax    = params.get(self.hyper_var_names[7], self.mmax_fixed)

        # ── Step 1: PPF for p(m1 | Lambda) ──────────────────────────────────
        # p(m1) = [(1-lam)*PowerLaw + lam*TruncNorm] * Smoothing, on 1-D grid
        p_m1 = two_component_single(
            self.m1s, alpha, lam, mmin, mmax, mpp, sigpp,
            gaussian_mass_maximum=self.gaussian_mass_maximum)
        p_m1 = p_m1 * smoothing(self.m1s, mmin, mmax, delta_m)

        cdf_m1 = _cumtrapz(p_m1, self.m1s)
        cdf_m1 = cdf_m1 / jnp.clip(cdf_m1[-1], 1e-30)   # normalize to [0, 1]

        m1_samples = jnp.interp(u[:, 0], cdf_m1, self.m1s)

        # ── Step 2: PPF for p(q | m1, Lambda) for each sampled m1 ───────────
        # Broadcast: m1_samples (N_samp,1) × self.qs (1,N_q) → (N_samp, N_q)
        m1_bc = m1_samples[:, None]   # (N_samp, 1)
        q_bc  = self.qs[None, :]      # (1, N_q)

        p_q = powerlaw(q_bc, beta, 1, mmin / m1_bc)
        p_q = p_q * smoothing(m1_bc * q_bc, mmin, m1_bc, delta_m)

        # Per-sample CDF along q axis; shape (N_samp, N_q)
        dq = jnp.diff(self.qs)
        cdf_q = jnp.concatenate([
            jnp.zeros((u.shape[0], 1)),
            jnp.cumsum((p_q[:, :-1] + p_q[:, 1:]) * 0.5 * dq[None, :], axis=-1)
        ], axis=-1)
        cdf_q = cdf_q / jnp.clip(cdf_q[:, -1:], 1e-30)  # normalize each row

        # Invert independently for each sample
        q_samples = jax.vmap(
            lambda u_q, cdf: jnp.interp(u_q, cdf, self.qs)
        )(u[:, 1], cdf_q)

        return {self.primary_mass_name: m1_samples, self.mass_ratio_name: q_samples}
