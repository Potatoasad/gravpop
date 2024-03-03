# Spin Tests
from gravpop import *

## Test Non Analytic Spin that it integrates to one
def test_non_analytic_spin_integrates_to_one():
    TG = TruncatedGaussian1D(a=0, b=1, var_name='x')
    params = {'mu' : 0.0, 'sigma': 0.1}
    test = lambda x: np.exp(TG({'x' : jnp.array([x])}, params))

    xs = np.linspace(0,1,10_000)
    ys = np.array([test(x) for x in xs])
    #plt.plot(xs, ys)

    assert np.isclose( np.trapz(y=ys, x=xs), 1.0, rtol=1e-3)


## Test Analytic Spin matches expected result from monte carlo estimate
def test_analytic_spin_matches_monte_carlo_estimate():
    import scipy
    TG = TruncatedGaussian1D(a=0, b=1)
    TGA = TruncatedGaussian1DAnalytic(a=0, b=1)
    mu_0, sigma_0 = 0.5, 0.1
    data = {'x' : scipy.stats.truncnorm.rvs(loc=0.5, scale=0.1, a=(0-mu_0)/sigma_0, b=(1-mu_0)/sigma_0, size=800_000)}
    data['x_mu_kernel'] = mu_0
    data['x_sigma_kernel'] = sigma_0


    params = {'mu' : 0.5, 'sigma': 0.1}
    assert jnp.isclose( TG(data, params), TGA(data, params) , rtol=1e-3)