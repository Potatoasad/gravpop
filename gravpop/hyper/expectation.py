import jax
import jax.numpy as jnp

def convert_log_weights_to_weights(logweight):
	if logweight is not None:
		return jnp.exp(logweight)
	else:
		return  None


class AbstractExpectation:
	"""
	Abstract class to compute the expectation of the likelihood integral
	in different cases
	"""
	def __init__(self):
		pass

	def num_samples(self, weights, N_samples=None):
		return N_samples or weights.shape[-1]

####################################################
### Sampled Expectation
###################################################

class SampledExpectation(AbstractExpectation): # takes in an array with shape E x ... x N
	def log_mu(self, logweights, N_samples=None):
		if N_samples is None:
			N_samples = self.num_samples(logweights, N_samples)
		return jax.scipy.special.logsumexp(logweights, axis=-1) - jnp.log(N_samples)  # E x ...

	def mu(self, logweights, N_samples):
		return jnp.exp(logweights).sum(axis=-1) / N_samples  

	def log_bayes_factors(self, logweights, N_samples=None):
		return self.log_mu(logweights, N_samples=N_samples)


class SampledEventExpectation(SampledExpectation):
	def var(self, logweights, N_samples=None):
		weights = jnp.exp(logweights)
		N = self.num_samples(weights, N_samples=N_samples)
		var = (weights**2).sum(axis=-1)
		var /= N**2
		return var # E

	def N_eff(self, logweights, mu=None, N_samples=None):
		weights = jnp.exp(logweights)
		N_samples = self.num_samples(weights, N_samples=N_samples)
		if mu is None:
			mu = self.mu(weights, N_samples=N_samples)
		var = self.var(weights, N_samples=N_samples)
		return mu**2 / var

class SampledSelectionExpectation(SampledExpectation):
	def var(self, logweights, mu=None, N_samples=None):
		weights = jnp.exp(logweights)
		if mu is None:
			mu = self.mu(weights, N_samples=None)
		N = self.num_samples(weights, N_samples=N_samples)
		var = (weights**2).sum(axis=-1)
		var /= N**2
		var -= (mu**2 / N)
		return var

	def N_eff(self, logweights, mu=None, N_samples=None):
		weights = jnp.exp(logweights)
		N_samples = self.num_samples(weights, N_samples=N_samples)
		if mu is None:
			mu = (weights.sum(axis=-1) / N_samples)
		var = self.var(weights, mu=mu, N_samples=N_samples)
		return mu**2 / var

####################################################
### Analytic Expectation
###################################################

class AnalyticExpectation(AbstractExpectation): # E x ... x K
	def num_components(self, logweights, N_components=None):
		return N_components or logweights.shape[-1]

	def log_mu(self, logweights):
		return logweights					    # E x ... x K

	def mu(self, logweights):
		return jnp.exp(log_weights)

	def aggregate_components(self, loglikes, weights=1):
		return jax.scipy.special.logsumexp( loglikes + jnp.log(weights), axis=-1) # E x ...

	def log_bayes_factors(self, logweights, weights=1):
		return self.aggregate_components(self.log_mu(logweights), weights) # E x ...

####################################################
### Hybrid Expectation
###################################################

class HybridExpectation(SampledExpectation, AnalyticExpectation): # E x ... x K x N
	def num_components(self, sampled_logweights=None, analytic_logweights=None, weights=None, N_components=None):
		if N_components is not None:
			return N_components
		if weights is not None:
			return weights.shape[-1]
		if analytic_logweights is not None:
			return analytic_logweights.shape[-1]
		if sampled_logweights is not None:
			return sampled_logweights.shape[-2]
		return N_components

	def log_mu(self, sampled_logweights=None, analytic_logweights=None, N_samples_per_component=None):
		log_bayes_factor_per_component = 0.0
		if sampled_logweights is not None:
			N_samples_per_component = self.num_samples(sampled_logweights, N_samples=N_samples_per_component)
			#print(SampledExpectation.log_mu(self, sampled_logweights, N_samples=N_samples_per_component).shape)
			log_bayes_factor_per_component += SampledExpectation.log_mu(self, sampled_logweights, N_samples=N_samples_per_component)
		if analytic_logweights is not None:
			#print(analytic_logweights.shape)
			#print(AnalyticExpectation.log_mu(self, analytic_logweights).shape)
			log_bayes_factor_per_component += AnalyticExpectation.log_mu(self, analytic_logweights)
		return log_bayes_factor_per_component

	def mu(self, sampled_logweights=None, analytic_logweights=None, N_samples_per_component=None):
		return jnp.exp(self.log_mu(sampled_logweights, analytic_logweights, N_samples_per_component))

	def log_bayes_factors(self, sampled_logweights=None, analytic_logweights=None, weights=1, N_samples_per_component=None):
		log_bayes_factor_per_component = self.log_mu(sampled_logweights, analytic_logweights, N_samples_per_component)
		log_bayes_factors = AnalyticExpectation.aggregate_components(self, log_bayes_factor_per_component, weights=weights)
		return log_bayes_factors


class HybridEventExpectation(HybridExpectation):
	def var(self, sampled_logweights=None, analytic_logweights=None, N_samples_per_component=None, weights=1):
		sampled_weights = jnp.exp(sampled_logweights) if sampled_logweights is not None else None
		analytic_weights = jnp.exp(analytic_logweights) if analytic_logweights is not None else None
		N_k = self.num_samples(sampled_weights, N_samples=N_samples_per_component)
		K = self.num_components(sampled_logweights, analytic_logweights, weights)
		var = (sampled_weights**2).sum(axis=-1)  # E x ... x K x N -> E x ... x K
		var *= (analytic_weights * weights)**2   # E x ... x K
		var = var.sum(axis=-1) / N_k**2 # E x ... 
		return var

	def N_eff(self, sampled_logweights=None, analytic_logweights=None, mu=None, N_samples_per_component=None, weights=1):
		N_samples_per_component = self.num_samples(sampled_logweights, N_samples=N_samples_per_component)
		if mu is None:
			mu = jnp.exp(self.log_bayes_factors(sampled_logweights=sampled_logweights, 
										   analytic_logweights=analytic_logweights, 
										   N_samples_per_component=N_samples_per_component,
										   weights=weights))
		var = self.var(sampled_logweights=sampled_logweights, 
					   analytic_logweights=analytic_logweights, 
					   N_samples_per_component=N_samples_per_component, 
					   weights=weights)
		return mu**2 / var


class HybridSelectionExpectation(HybridExpectation):
	def log_mu(self, sampled_logweights=None, analytic_logweights=None, N_samples_per_component=None, total_generated=None):
		if total_generated is not None:
			K = self.num_components(sampled_logweights, analytic_logweights)
			N_k = self.num_samples(sampled_logweights, N_samples=N_samples_per_component)
			detection_ratio = N_k * K / total_generated
		else:
			detection_ratio = 1
		return HybridExpectation.log_mu(self, sampled_logweights=sampled_logweights, analytic_logweights=analytic_logweights, 
								N_samples_per_component=N_samples_per_component) + jnp.log(detection_ratio)

	def log_bayes_factors(self, sampled_logweights=None, analytic_logweights=None, weights=1, N_samples_per_component=None, total_generated=None):
		log_bayes_factor_per_component = self.log_mu(sampled_logweights, analytic_logweights, N_samples_per_component)
		log_bayes_factors = AnalyticExpectation.aggregate_components(self, log_bayes_factor_per_component, weights=weights)
		return log_bayes_factors

	def mu(self, sampled_logweights=None, analytic_logweights=None, N_samples_per_component=None, total_generated=None):
		return jnp.exp(self.log_mu(sampled_logweights=sampled_logweights, analytic_logweights=analytic_logweights, 
								   N_samples_per_component=N_samples_per_component, total_generated=total_generated))

	def var(self, mu=None, sampled_logweights=None, analytic_logweights=None, N_samples_per_component=None, weights=1, total_generated=None):
		if mu is None:
			mu = self.log_bayes_factors(sampled_logweights=sampled_logweights, 
									   analytic_logweights=analytic_logweights, 
									   N_samples_per_component=N_samples_per_component,
									   weights=weights, total_generated=total_generated)
		sampled_weights = jnp.exp(sampled_logweights) if sampled_logweights is not None else None
		analytic_weights = jnp.exp(analytic_logweights) if analytic_logweights is not None else None
		K = self.num_components(sampled_logweights, analytic_logweights, weights)
		N_k = self.num_samples(sampled_weights, N_samples=N_samples_per_component)
		detection_ratio = N_k * K / total_generated
		var = (sampled_weights**2).sum(axis=-1)  # E x ... x K x N -> E x ... x K
		var *= (analytic_weights * weights)**2   # E x ... x K
		var = detection_ratio**2 * var.sum(axis=-1) / N_k # E x ... 
		var -= (mu**2 / total_generated)  # E x ... 
		return var

	def N_eff(self, sampled_logweights=None, analytic_logweights=None, mu=None, N_samples_per_component=None, weights=1, total_generated=None):
		N_samples_per_component = self.num_samples(sampled_logweights, N_samples=N_samples_per_component)
		if mu is None:
			mu = jnp.exp(self.log_bayes_factors(sampled_logweights=sampled_logweights, 
										   analytic_logweights=analytic_logweights, 
										   N_samples_per_component=N_samples_per_component,
										   weights=weights, total_generated=total_generated))
		var = self.var(sampled_logweights=sampled_logweights, 
					   analytic_logweights=analytic_logweights, 
					   mu=mu, N_samples_per_component=N_samples_per_component, 
					   weights=weights, total_generated=total_generated)
		return mu**2 / var


