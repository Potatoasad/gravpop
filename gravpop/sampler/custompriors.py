import numpy as np
from jax import random
import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes

class LKJCorrelationPrior(Distribution):
    support = constraints.interval(-1, 1)  # Define the support of the distribution
    has_enumerate_support = False
    low = -1
    high = 1

    def __init__(self):
        super().__init__(batch_shape=(), event_shape=())
        
    @staticmethod
    def real_value(u):
        # Define the complex number
        z = (1 - 1j * np.sqrt(3)) / (2 * (-1 + 2*u + 2j * np.sqrt(u - u**2))**(1/3)) + \
            0.5 * (1 + 1j * np.sqrt(3)) * (-1 + 2*u + 2j * np.sqrt(u - u**2))**(1/3)

        # Compute the real part of the complex number
        real_part = np.real(z)
        return real_part

    def sample(self, key, sample_shape=()):
        # Inverse transform sampling method
        u = random.uniform(key, shape=sample_shape)
        return self.real_value(u)  # Sampling from the inverse CDF of the distribution

    def log_prob(self, value):
        # Ensure the value is within the support
        value = promote_shapes(value)[0]
        valid = (value >= -1) & (value <= 1)
        log_prob = jnp.where(
            valid,
            jnp.log(3 / 4) + jnp.log(1 - value ** 2),
            -jnp.inf
        )
        return log_prob

    @property
    def mean(self):
        return 0.0

    @property
    def variance(self):
        return 4 / 45  # Variance of the distribution


import jax
import jax.numpy as jnp
from jax import random
from numpyro.distributions import Distribution, constraints

class DiracDelta(Distribution):
    support = constraints.real
    has_enumerate_support = False

    def __init__(self, value):
        self.value = value
        batch_shape = ()
        event_shape = ()
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        return jnp.full(sample_shape, self.value)

    def log_prob(self, value):
        log_prob = jnp.where(value == self.value, 0.0, -jnp.inf)
        return log_prob

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0.0

import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro.distributions import constraints
from numpyro.handlers import seed, trace
import matplotlib.pyplot as plt

class Triangular(Distribution):
    support = constraints.interval(0, 1)
    
    def __init__(self, a=0.0, b=1.0, mode=0.5, validate_args=None):
        self.a, self.b, self.mode = a, b, mode
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)
    
    def sample(self, key, sample_shape=()):
        u = numpyro.distributions.Uniform(0, 1).sample(key, sample_shape=sample_shape)
        condition = (u < (self.mode - self.a) / (self.b - self.a))
        x = jnp.where(condition,
                      self.a + jnp.sqrt(u * (self.b - self.a) * (self.mode - self.a)),
                      self.b - jnp.sqrt((1 - u) * (self.b - self.a) * (self.b - self.mode)))
        return x

    def log_prob(self, value):
        a, b, mode = self.a, self.b, self.mode
        norm = 2 / (b - a)
        condition1 = (value >= a) & (value < mode)
        condition2 = (value >= mode) & (value <= b)
        result = jnp.where(condition1, norm * (value - a) / (mode - a), 0)
        result = jnp.where(condition2, norm * (b - value) / (b - mode), result)
        return jnp.log(result)





