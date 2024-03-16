from dataclasses import dataclass
from typing import List, Union, Dict, Optional
import jax.numpy as jnp
import jax
import numpy


@dataclass
class Grid1D:
	name : str = 'x'
	minimum : float = 0.0
	maximum : float = 0.0
	N : int = 100
	latex_name : Optional[str] = None

	def __post_init__(self):
		if self.latex_name is None:
			new_name = self.name.replace('_','') 
			self.latex_name = fr'${new_name}$'

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
