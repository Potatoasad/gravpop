class AbstractPopulationModel:
	pass


class SampledPopulationModel(AbstractPopulationModel):
	def evaluate(self, data, params):
		return self.__call__(data, params)

class AnalyticPopulationModel(AbstractPopulationModel):
	pass


class InvertiblePopulationModel(AbstractPopulationModel):
	"""
	Mixin for population models that expose an inverse CDF (percent-point function).

	Subclasses implement ``ppf(u, params) -> dict`` which maps uniform draws
	``u ~ Uniform(0,1)^{n_draws}`` to a dict of physical parameter samples
	``{var_name: array}``.  This enables differentiable detection-efficiency
	evaluation via dynamic injection drawing (Algorithm 1 of arXiv:2404.xxxxx):

	    xi(Lambda) = mean_i  P_hat(det | Theta_i),   Theta_i = ppf(c_i, Lambda)

	where ``c_i`` are fixed low-discrepancy draws stored once before inference.

	Properties
	----------
	n_draws : int
	    Number of independent Uniform(0,1) draws required per sample.
	    Equals ``len(var_names)`` by default but may be larger for joint
	    mixture models that need an extra component-selector draw.
	"""

	@property
	def n_draws(self):
		return len(self.var_names)

	def ppf(self, u, params):
		"""
		Inverse CDF: map uniform draws to physical parameter samples.

		Parameters
		----------
		u : jax.Array, shape (N_samp,) for 1-D models or (N_samp, n_draws)
		params : dict of hyperparameter scalars

		Returns
		-------
		dict  {var_name: jax.Array of shape (N_samp,)}
		"""
		raise NotImplementedError


class SpinPopulationModel(AbstractPopulationModel):
	pass

class SpinMagnitudePopulationModel(SpinPopulationModel):
	pass

class SpinOrientationPopulationModel(SpinPopulationModel):
	pass

class RedshiftPopulationModel(AbstractPopulationModel):
	pass

class MassPopulationModel(AbstractPopulationModel):
	pass