import jax.numpy as jnp
import jax
from jax import jit

#def box(x,a,b):
#    return jnp.heaviside(x-a, 0.5)*jnp.heaviside(b-x, 0.5)

@jit
def box(x,a,b):
    return jnp.where(x > a, jnp.where(x < b, jnp.ones_like(x), jnp.zeros_like(x)), jnp.zeros_like(x))

@jit
def powerlaw(x, alpha, high, low):
    norm = jnp.where(
        jnp.array(alpha) == -1,
        1 / jnp.log(high / low),
        (1 + alpha) / jnp.array(high ** (1 + alpha) - low ** (1 + alpha)),
    )
    prob = jnp.power(x, alpha)*norm/box(x, low, high)
    return prob

@jit
def truncnorm(x, mu, sigma, high, low):
    norm = 2**0.5 / jnp.pi**0.5 / sigma
    norm /= jax.scipy.special.erf((high - mu) / 2**0.5 / sigma) + jax.scipy.special.erf(
        (mu - low) / 2**0.5 / sigma
    )
    prob = jnp.exp(-jnp.power(x - mu, 2) / (2 * sigma**2))
    prob *= norm*box(x, low, high)
    return prob