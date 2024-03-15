### Test Data
from dataclasses import dataclass
import numpyro.distributions as dist
from numpyro.distributions import Distribution
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
from gravpop.utils.vmap import chunked_vmap
import numpy as np

from typing import Dict, List

def jax_to_numpy(obj):
    if isinstance(obj, jnp.ndarray):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {key: jax_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(jax_to_numpy(item) for item in obj)
    else:
        return obj

def model_gradient(model, data, param, canonical_parameter_order=None):
	canonical_parameter_order = canonical_parameter_order or list(param.keys())

	def make_vector(d):
		return jnp.array([d[param] for param in canonical_parameter_order])

	def make_dictionary(x):
		return {parameter : x[i] for i,parameter in enumerate(canonical_parameter_order)}

	#dYdx = jacrev(lambda x: model(data, make_dictionary(x)))(make_vector(param))
	dYdx = jacfwd(lambda x: model(data, make_dictionary(x)))(make_vector(param))
	if len(canonical_parameter_order) == 1:
		return {parameter : dYdx.flatten() for i,parameter in enumerate(canonical_parameter_order)}

	return {parameter : dYdx[..., i] for i,parameter in enumerate(canonical_parameter_order)}

@dataclass
class PriorDictionary:
	priors : Dict[str, Distribution]
	seed : int = 0

	def sample(seed=None):
		seed = seed or self.seed
		key = jax.random.key(seed)
		point = {}
		for variable, prior_distribution in self.priors.items():
			point[variable] = prior_distribution.sample(key)

		return point


class Dirac(dist.Distribution):
    arg_constraints = {}
    support = dist.constraints.real
    has_enumerate_support = False

    def __init__(self, value):
        super(Dirac, self).__init__(event_shape=(1,))
        self.value = value

    def sample(self, key, sample_shape=()):
        return jnp.full(sample_shape + self.event_shape, self.value)

    def log_prob(self, value):
        return jnp.where(value == self.value, 0., -jnp.inf)

@dataclass
class Grid1D:
	name : str = 'x'
	minimum : float = 0.0
	maximum : float = 0.0
	N : int = 100
	latex_name : str = r'$x$'

	def __post_init__(self):
		if self.name != 'x':
			self.latex_name = fr'${self.name}$'

		self.grid = jnp.linspace(self.minimum, self.maximum, self.N)
		self._data = {self.name : self.grid}

	@property
	def data(self):
		return self._data
	

@dataclass
class Grid:
	grid_list : List[Grid1D]

	def __post_init__(self):
		self.grid_1ds = {g.name : g.grid for g in self.grid_list}
		grid_combos = jnp.meshgrid(*[g.grid for g in self.grid_list])
		data = {}
		for i,g in enumerate(self.grid_list):
			data[g.name] = grid_combos[i]
		self._data = data

	@property
	def data(self):
		return self._data


GLOBAL_DICTIONARY_OF_LIMITS = {
    'alpha' : [-4,12],
    'beta' : [-2,7],
    'mmin' : [2,10],
    'mmax' : [30,100],
    'delta_m' : [0,10],
    'lam' : [0,1],
    'mpp' : [20,50],
    'sigpp' : [1,10],
    'lamb' : [-6,6],
    'mu_1' : [0,1],
    'sigma_1' : [0,5],
    'mu_2' : [0,1],
    'sigma_2' : [0,5]
}



GLOBAL_DICTIONARY_OF_NICE_VALUES = {
    'alpha' : 3.5,
    'beta' : 1.1,
    'mmin' : 5,
    'mmax' : 95,
    'delta_m': 0.1,
    'lam' : 0.4,
    'mpp' : 35,
    'sigpp' : 4,
    'lamb' : 2.9,
    'mu_1' : 0.1,
    'sigma_1' : 0.1,
    'mu_2' : 0.1,
    'sigma_2' : 0.1
}


GLOBAL_DICTIONARY_OF_EVIL_VALUES = {
	 'alpha': 3.5,
	 'beta': 1.1,
	 'delta_m': 3.0,
	 'lam': 0.4,
	 'mmin': 3.0,
	 'mmax': 95.0,
	 'mpp': 35.0,
	 'sigpp': 4.0,
	 'lamb': 2.9
}


PRIOR_DICT = PriorDictionary({var : dist.Uniform(*limits) for var, limits in GLOBAL_DICTIONARY_OF_LIMITS.items()})




#{'alpha':3.5, 'beta':1.1, 'delta_m':0.03, 'mpp': 30.0, 'sigpp': 3.0, 'lam':0.04, 'mmin':2.0, 'mmax': 90.0}
		


