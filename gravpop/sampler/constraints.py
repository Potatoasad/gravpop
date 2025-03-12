from dataclasses import dataclass, field
from typing import List, Union, Dict, Optional, Any
from ..models.spin import *
from ..models.utils import box
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

def alpha_beta_max_to_mu_var_max(alpha, beta, amax):
    mu = alpha / (alpha + beta) * amax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax ** 2
    return mu, var, amax


def mu_var_max_to_alpha_beta_max(mu, var, amax):
    mu /= amax
    var /= amax ** 2
    alpha = (mu ** 2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta, amax

def truncated_normal_stats(mu, var, a=0, b=1):
    sigma = jnp.sqrt(var)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    
    Z = norm.cdf(beta) - norm.cdf(alpha)
    phi_alpha = norm.pdf(alpha)
    phi_beta = norm.pdf(beta)
    
    truncated_mean = mu + sigma * (phi_alpha - phi_beta) / Z
    truncated_var = var * (1 + (alpha * phi_alpha - beta * phi_beta) / Z - ((phi_alpha - phi_beta) / Z) ** 2)
    
    return jnp.stack([truncated_mean, truncated_var], axis=0)

def det_of_map_var(mu, sigma):
        return jnp.linalg.det(
            jax.jacrev(lambda mu_sig : truncated_normal_stats(mu_sig[0,...], mu_sig[1,...]**2, 0, 1))(jnp.stack([mu, sigma], axis=0))
        )


def det_of_map_std(mu, sigma):
        return jnp.linalg.det(
            jax.jacrev(lambda mu_sig : truncated_normal_stats(mu_sig[0,...], mu_sig[1,...], 0, 1))(jnp.stack([mu, sigma], axis=0))
        )


@dataclass
class AlphaBetaConstraint:
	hyper_var_names : List[str] = field(default_factory=lambda : ['mu_chi', 'sigma_chi'])
	alpha : List[int] = field(default_factory=lambda : [1,1e5])
	beta : List[int] = field(default_factory=lambda : [1,1e5])

	def logpdf(self, x):
		mu = x[self.hyper_var_names[0]]
		var = x[self.hyper_var_names[1]]
		amax = x.get('amax', 1)

		alpha, beta, _ = mu_var_max_to_alpha_beta_max(mu, var, amax)

		is_alpha_within_constraint = (alpha > self.alpha[0]) & (alpha < self.alpha[1])
		is_beta_within_constraint = (beta > self.beta[0]) & (beta < self.beta[1])
		both_within_constraint = is_alpha_within_constraint & is_beta_within_constraint

		return jnp.log(box(alpha, self.alpha[0], self.alpha[1]))  + jnp.log(box(beta, self.beta[0], self.beta[1]))

@dataclass
class FlatInMeanVarExact:
    hyper_var_names : List[str] = field(default_factory=lambda : ['mu_chi', 'sigma_chi'])
    custom_additional_logprob_func : Optional[Any] = lambda E,V,Z: 0.0

    @staticmethod
    def log_prior_mean_var_trunc_norm(mu, sigma):
        return jnp.log(det_of_map_var(mu,sigma))

    def logpdf(self, x):
        mu = x[self.hyper_var_names[0]]
        sigma = x[self.hyper_var_names[1]]
        return self.log_prior_mean_var_trunc_norm(mu, sigma)

@dataclass
class FlatInMeanStdExact:
    hyper_var_names : List[str] = field(default_factory=lambda : ['mu_chi', 'sigma_chi'])
    custom_additional_logprob_func : Optional[Any] = lambda E,V,Z: 0.0

    @staticmethod
    def log_prior_mean_var_trunc_norm(mu, sigma):
        return jnp.log(det_of_map_std(mu,sigma))

    def logpdf(self, x):
        mu = x[self.hyper_var_names[0]]
        sigma = x[self.hyper_var_names[1]]
        return self.log_prior_mean_var_trunc_norm(mu, sigma)


@dataclass
class FlatInMeanVar:
    hyper_var_names : List[str] = field(default_factory=lambda : ['mu_chi', 'sigma_chi'])
    custom_additional_logprob_func : Optional[Any] = lambda E,V,Z: 0.0
    
    def mean_var_Z_trunc_norm(self, mu, sigma):
        alpha,  beta = (-mu/sigma), ((1-mu)/sigma)
        Z = jax.scipy.stats.norm.cdf(beta) -  jax.scipy.stats.norm.cdf(alpha)
        kappa = (jax.scipy.stats.norm.pdf(beta) - jax.scipy.stats.norm.pdf(alpha)) / Z
        nu = (beta*jax.scipy.stats.norm.pdf(beta) - alpha*jax.scipy.stats.norm.pdf(alpha)) / Z
        E = mu * (1 + kappa/alpha)
        V = (sigma**2) * (1 - nu - kappa**2)
        return E, V, Z

    def log_prior_mean_var_trunc_norm(self, mu, sigma):
        E, V, Z = self.mean_var_Z_trunc_norm(mu, sigma)
        log_prob = 5*jnp.log(Z)
        log_prob += -0.5*((jnp.log(V)-jnp.log(0.015))/(1.5))**2
        log_prob +=  -0.1*(0.5*((V-0.04)/(0.01))**2 + 0.5*((E-0.5)/(0.3))**2)
        log_prob += self.custom_additional_logprob_func(E,V,Z)
        return log_prob

    def logpdf(self, x):
        mu = x[self.hyper_var_names[0]]
        sigma = x[self.hyper_var_names[1]]
        return self.log_prior_mean_var_trunc_norm(mu, sigma)

@dataclass
class FlatInMeanStd:
    hyper_var_names : List[str] = field(default_factory=lambda : ['mu_chi', 'sigma_chi'])
    
    def mean_var_Z_trunc_norm(self, mu, sigma):
        alpha,  beta = (-mu/sigma), ((1-mu)/sigma)
        Z = jax.scipy.stats.norm.cdf(beta) -  jax.scipy.stats.norm.cdf(alpha)
        kappa = (jax.scipy.stats.norm.pdf(beta) - jax.scipy.stats.norm.pdf(alpha)) / Z
        nu = (beta*jax.scipy.stats.norm.pdf(beta) - alpha*jax.scipy.stats.norm.pdf(alpha)) / Z
        E = mu * (1 + kappa/alpha)
        V = (sigma**2) * (1 - nu - kappa**2)
        return E, V, Z

    def log_prior_mean_std_trunc_norm(self, mu, sigma):
        E, V, Z = self.mean_var_Z_trunc_norm(mu, sigma)
        log_prob = 5*jnp.log(Z)
        log_prob += -0.5*((jnp.log(V)-jnp.log(0.015))/(1.5))**2
        log_prob +=  -0.1*(0.5*((V-0.04)/(0.01))**2 + 0.5*((E-0.5)/(0.3))**2)
        return log_prob - 0.5*jnp.log(V)

    def logpdf(self, x):
        mu = x[self.hyper_var_names[0]]
        sigma = x[self.hyper_var_names[1]]
        return self.log_prior_mean_std_trunc_norm(mu, sigma)


@dataclass
class VarianceToStd:
    hyper_var_names : List[str] = field(default_factory=lambda : ['sigma_chi'])
    
    def logpdf(self, x):
        sigma = x[self.hyper_var_names[0]]
        return -jnp.log(sigma)


