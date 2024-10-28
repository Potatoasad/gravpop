This library allows one to perform a gravitational wave population analysis (), inspired by methods from [Thrane et al.](https://arxiv.org/abs/1809.02293), with an extension based on a technique from [Hussain et al.](...) that allows exploration of population features even in narrow regions near the edges of a bounded domain.

The approach splits parameter space into two sectors:
- An __analytic sector__ ($\theta^a$), where the population model is represented as a weighted sum of multivariate truncated normal distributions, allowing for an analytical computation of the population likelihood.
- A __sampled sector__ ($\theta^s$), which accommodates more general population models and utilizes Monte Carlo estimates of the population likelihood.

This technique represents posterior samples using a truncated Gaussian mixture model (TGMM), where the population likelihood, $p(x)$, is expressed as a sum of truncated multivariate Gaussian components:

$$
p(x) = \sum_k w_k \, \phi_{[a,b]}(x \mid \mu_k, \Sigma_k).
$$

For implementing this, see [truncatedgaussianmixtures](https://github.com/Potatoasad/truncatedgaussianmixtures), a package designed to fit data to mixtures of truncated Gaussians.
