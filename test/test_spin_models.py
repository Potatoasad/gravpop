# Spin Tests
from gravpop import *
import gwpopulation
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import numpy as np

## Test Non Analytic Spin that it integrates to one
def test_non_analytic_spin_integrates_to_one():
    TG = TruncatedGaussian1D(a=0, b=1, var_name='x')
    params = {'mu' : 0.0, 'sigma': 0.1}
    test = lambda x: TG({'x' : jnp.array([x])}, params)

    xs = np.linspace(0,1,10_000)
    ys = np.array([test(x) for x in xs])
    #plt.plot(xs, ys)

    assert np.isclose( np.trapz(y=ys, x=xs), 1.0, rtol=1e-1)


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


class TestSampledSpinModels:

    def setup_method(self, method):
        self.evaluation_point = {'mu_chi': 2.0, 'sigma_chi':0.9, 'xi_spin':0.1, 'sigma_spin':0.8}

    def create_magnitude_dataset(self):
        a1_grid = Grid1D("a_1", minimum=0, maximum=1, N=100)
        a2_grid = Grid1D("a_2", minimum=0, maximum=1, N=100)
        a_grid = Grid([a1_grid, a2_grid])
        return a_grid.data

    def create_orientations_dataset(self):
        cos_tilt_1_grid = Grid1D("cos_tilt_1", minimum=-1, maximum=1, N=100)
        cos_tilt_2_grid = Grid1D("cos_tilt_2", minimum=-1, maximum=1, N=100)
        cos_tilt_grid = Grid([cos_tilt_1_grid, cos_tilt_2_grid])
        return cos_tilt_grid.data

    def gwpopulation_iid_magnitude_model(self, data, params):
        cols = ['alpha_chi', 'beta_chi']
        return gwpopulation.models.spin.iid_spin_magnitude_beta(data, **{k:v for k,v in params.items() if k in cols})

    def gwpopulation_iid_orientations_model(self, data, params):
        cols = ['xi_spin', 'sigma_spin']
        return gwpopulation.models.spin.iid_spin_orientation_gaussian_isotropic(data, **{k:v for k,v in params.items() if k in cols})

    def gravpop_iid_magnitude_model(self, data, params):
        Mag = BetaSpinMagnitudeIID(parameterization="mu_sigma", var_names = ['a_1', 'a_2'])
        return Mag(data, params)

    def gravpop_iid_orientations_model(self, data, params):
        Orient = GaussianIsotropicSpinOrientationsIID(var_names = ['cos_tilt_1', 'cos_tilt_2'])
        return Orient(data, params)

    def test_compare_magnitudes(self):
        mag_data = self.create_magnitude_dataset()
        Z_gwpop = self.gwpopulation_iid_magnitude_model(mag_data, self.evaluation_point)
        Z_gravpop = self.gravpop_iid_magnitude_model(mag_data, self.evaluation_point)

        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax1.contourf(mag_data['a_1'], mag_data['a_2'], jnp.log(Z_gravpop))
        ax2.contourf(mag_data['a_1'], mag_data['a_2'], jnp.log(Z_gwpop))
        ax1.set_xlabel(r"$a_1$"); ax2.set_xlabel(r"$a_1$");
        ax1.set_ylabel(r"$a_2$"); ax2.set_ylabel(r"$a_2$");
        ax1.set_title(r"$\log P_{gravpop}$")
        ax2.set_title(r"$\log P_{gwpopulation}$")
        ax3.hist(jnp.abs(Z_gravpop/(jnp.abs(Z_gwpop) + 1e-16)).flatten(), bins=10);
        ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
        plt.suptitle("Log probability comparison for spin magnitude models")
        fig.savefig("./test/images/spin_magnitude_model_comparison.png")
        assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs

    def test_compare_orientations(self):
        orientation_data = self.create_orientations_dataset()
        Z_gwpop = self.gwpopulation_iid_orientations_model(orientation_data, self.evaluation_point)
        Z_gravpop = self.gravpop_iid_orientations_model(orientation_data, self.evaluation_point)

        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax1.contourf(orientation_data['cos_tilt_1'], orientation_data['cos_tilt_2'], jnp.log(Z_gravpop))
        ax2.contourf(orientation_data['cos_tilt_1'], orientation_data['cos_tilt_2'], jnp.log(Z_gwpop))
        ax1.set_xlabel(r"$cos(\theta_1)$"); ax2.set_xlabel(r"$cos(\theta_1)$");
        ax1.set_ylabel(r"$cos(\theta_2)$"); ax2.set_ylabel(r"$cos(\theta_2)$");
        ax1.set_title(r"$\log P_{gravpop}$")
        ax2.set_title(r"$\log P_{gwpopulation}$")
        ax3.hist(jnp.abs(Z_gravpop/(jnp.abs(Z_gwpop) + 1e-16)).flatten() - 1, bins=10);
        ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop} - 1$")
        plt.suptitle("Log probability comparison for spin orientation models")
        fig.savefig("./test/images/spin_orientations_model_comparison.png")
        assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs






