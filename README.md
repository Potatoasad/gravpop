# gravpop
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://potatoasad.github.io/gravpop/index.html)
[![PyPI version](https://badge.fury.io/py/gravpop.svg)](https://badge.fury.io/py/gravpop)


This library allows one to perform a gravitational wave population analysis, ([Hussain et al.](...), [Thrane et al.](https://arxiv.org/abs/1809.02293)) that allows exploration of population features even in narrow regions near the edges of a bounded domain. 

> *Feel free to jump to the tutorial [here](https://potatoasad.github.io/gravpop/Examples/gravpop_tutorial.html)*

The approach splits parameter space into two sectors:
- An __analytic sector__ ($\theta^a$), where the population model is represented as a weighted sum of multivariate truncated normal distributions, allowing for an analytical computation of the population likelihood.
- A __sampled sector__ ($\theta^s$), which accommodates more general population models and utilizes Monte Carlo estimates of the population likelihood.

This technique represents posterior samples using a truncated Gaussian mixture model (TGMM), where the population likelihood, $p(x)$, is expressed as a sum of truncated multivariate Gaussian components:

$$
p(x) = \sum_k w_k \, \phi_{[a,b]}(x \mid \mu_k, \Sigma_k).
$$

This form of the posterior allows analytic evaluation in the analytic sector, and falls back to using Monte-Carlo based estimation in the sampled sector.

For implementing the Truncated Gaussian Mixture Model fit, see [truncatedgaussianmixtures](https://github.com/Potatoasad/truncatedgaussianmixtures), a package designed to fit data to mixtures of truncated Gaussians.
