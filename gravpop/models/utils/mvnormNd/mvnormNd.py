# JAX-native truncated MVN moments (no NumPy/SciPy). JIT/VMAP/grad-friendly.
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.stats import norm
from jax.scipy.special import ndtri as normal_ppf


# ------------------------- utilities -------------------------
def _complement_indices(d: int, idx: Sequence[int]) -> jnp.ndarray:
    idx = jnp.asarray(idx, dtype=jnp.int32)
    mask = jnp.ones((d,), dtype=bool).at[idx].set(False)
    return jnp.nonzero(mask, size=d - idx.size, fill_value=0)[0]

@partial(jax.jit, static_argnames=[])
def _safe_logdiffexp(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    # log(exp(a) - exp(b)) with stability; swaps if needed.
    swap = a < b
    a1 = jnp.where(swap, b, a)
    b1 = jnp.where(swap, a, b)
    out = a1 + jnp.log1p(-jnp.exp(b1 - a1))
    return jnp.where(swap, -out, out)

# MVN logpdf (no scipy.multivariate_normal in jax.scipy)
@partial(jax.jit, static_argnames=[])
def _mvn_logpdf(x: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    d = cov.shape[0]
    L = jnp.linalg.cholesky(cov)             # cov = L L^T
    y = jax.scipy.linalg.solve_triangular(L, x - mean, lower=True)
    quad = jnp.dot(y, y)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (quad + logdet + d * jnp.log(2.0 * jnp.pi))

# ------------------------- JAX MVN box probability (Genz) -------------------------
from functools import partial
@partial(jax.jit, static_argnames=['n_samples'])
def mvn_box_prob(mu: jnp.ndarray,
                 Sigma: jnp.ndarray,
                 a: jnp.ndarray,
                 b: jnp.ndarray,
                 *,
                 key: jax.Array,
                 n_samples: int = 2048) -> jnp.ndarray:
    """
    P(a < X < b) for X ~ N(mu, Sigma), estimated by Genz's recursive transform.
    Fully JAX-native; differentiable; works under jit/vmap/pmap.
    """
    mu = jnp.asarray(mu); Sigma = jnp.asarray(Sigma)
    a = jnp.asarray(a);   b = jnp.asarray(b)
    d = Sigma.shape[0]
    if d == 0:
        return jnp.array(1.0, Sigma.dtype)

    L = jnp.linalg.cholesky(Sigma)
    U = jax.random.uniform(key, (n_samples, d), minval=1e-12, maxval=1.0 - 1e-12)
    idx = jnp.arange(d)

    def one_sample(u):
        def body(carry, i):
            y, logw = carry
            Li_full = L[i]                     # length-d row (static)
            Lii = L[i, i]
            mu_i = mu[i]
            contrib = jnp.sum(jnp.where(idx < i, Li_full * y, 0.0))
            li = (a[i] - (mu_i + contrib)) / Lii
            ui = (b[i] - (mu_i + contrib)) / Lii
            Phi_li = norm.cdf(li)
            Phi_ui = norm.cdf(ui)
            pi = jnp.clip(Phi_ui - Phi_li, a_min=0.0)
            yi = normal_ppf(Phi_li + pi * u[i])
            y = y.at[i].set(yi)
            logw = logw + jnp.log(pi + 1e-300)
            return (y, logw), None

        y0 = jnp.zeros((d,), dtype=Sigma.dtype)
        (_, logw), _ = lax.scan(body, (y0, 0.0), jnp.arange(d))
        return jnp.exp(logw)

    w = jax.vmap(one_sample)(U)
    return jnp.mean(w)