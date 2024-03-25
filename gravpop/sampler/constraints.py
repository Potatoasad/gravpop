from dataclasses import dataclass, field
from typing import List, Union, Dict, Optional
from ..models.spin import *
from ..models.utils import box

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

		#return jnp.where(both_within_constraint, 0, -jnp.inf)
		return jnp.log(box(alpha, self.alpha[0], self.alpha[1]))  + jnp.log(box(beta, self.beta[0], self.beta[1]))