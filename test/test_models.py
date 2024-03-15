import pytest

from gravpop import *
import gravpop
import gwpopulation
from .testing_utils import *

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


# Define your test class
class TestCompareToGwpopulation:
    # Define setup method if needed
    def setup_method(self, method):
        ## GLOBAL POINTS
        self.evaluation_point = GLOBAL_DICTIONARY_OF_NICE_VALUES
        #self.evaluation_point = GLOBAL_DICTIONARY_OF_EVIL_VALUES
        self.prior = PRIOR_DICT 

        #### Mass Models
        self.mass_params =  ['alpha','beta','mmin','mmax','mpp','sigpp','lam','delta_m']
        self.mass_point = {k:v for k,v in self.evaluation_point.items() if k in self.mass_params}
        self.mass_grid = Grid([Grid1D(name='mass_1', minimum=2, maximum=100, N=200), 
                               Grid1D(name='mass_ratio', minimum=0, maximum=1  , N=200)]).data

        self.gwpop_mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
        self.gravpop_mass_model = gravpop.SmoothedTwoComponentPrimaryMassRatio(primary_mass_name='mass_1')

        #### Redshift Model
        self.redshift_params =  ['lamb']
        self.redshift_point = {k:v for k,v in self.evaluation_point.items() if k in self.redshift_params}

        self.evaluation_point_redshift = GLOBAL_DICTIONARY_OF_NICE_VALUES
        self.redshift_grid = Grid1D(name='redshift', minimum=0, maximum=1.9, N=200).data

        self.gwpop_redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)
        self.gravpop_redshift_model = gravpop.PowerLawRedshift(z_max=1.9)
    
    # Define teardown method if needed
    def teardown_method(self, method):
        pass
    
    # Write your test methods
    def test_mass_model_matches_gwpop(self):
        Z_gwpop = self.gwpop_mass_model(jax_to_numpy(self.mass_grid), **self.mass_point)
        Z_gravpop = self.gravpop_mass_model(self.mass_grid, self.evaluation_point)

        #fig, axes = plt.subplots(ncols=2, figsize=(10,5))
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax1.contourf(self.mass_grid['mass_1'], self.mass_grid['mass_ratio'], jnp.log(Z_gravpop))
        ax2.contourf(self.mass_grid['mass_1'], self.mass_grid['mass_ratio'], jnp.log(Z_gwpop))
        ax1.set_xlabel(r"$m_1$"); ax2.set_xlabel(r"$m_1$");
        ax1.set_ylabel(r"$q$"); ax2.set_ylabel(r"$q$");
        ax1.set_title(r"$\log P_{gravpop}$")
        ax2.set_title(r"$\log P_{gwpopulation}$")
        ax3.hist(jnp.abs(Z_gravpop/Z_gwpop).flatten(), bins=100);
        ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
        plt.suptitle("Log probability comparison for mass models")
        fig.savefig("./test/images/mass_model_comparison.png")
        assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs

    def test_mass_model_gradient(self):
        derivatives = model_gradient(self.gravpop_mass_model, self.mass_grid, self.mass_point)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        keys = list(derivatives.keys())
        n = len(keys)
        fig = plt.figure(layout="constrained", figsize=(10, 5*n))
        gs = GridSpec(n, 2, figure=fig)

        axes = []
        for i in range(n):
            row_axes = []
            for j in range(2):
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            axes.append(row_axes)

        for i, ax_row in enumerate(axes):
            ax = ax_row[0]
            M1, Q = self.mass_grid["mass_1"], self.mass_grid["mass_ratio"]
            contour = ax.contourf(M1,Q,derivatives[keys[i]])
            cbar = plt.colorbar(contour, ax=ax)
                
            # Add axis labels and title
            ax.set_xlabel(r"$m_1$")
            ax.set_ylabel(r"$q$")
            ax.set_title(f"{keys[i]} \n Gradient of model probability w.r.t to {keys[i]}")
            
            ax = ax_row[1]
            null_values_mask = jnp.isnan(derivatives[keys[i]])
            ax.imshow(null_values_mask, extent=(M1.min(), M1.max(),
                                                 Q.min(), Q.max()),
                      cmap='binary', alpha=0.5, origin='lower', aspect='auto')
            
           #ax.set_aspect('equal')
                
            # Add axis labels and title
            ax.set_xlabel(r"$m_1$")
            ax.set_ylabel(r"$q$")
            ax.set_title(f"{keys[i]} \n Locations where model probability gradient was null")
            
            
        total_null_points = jnp.sum(jnp.array([jnp.sum(jnp.isnan(derivatives[x])) for x in derivatives.keys()]))
        plt.suptitle(f"A total of {total_null_points} points had null gradients")
        fig.savefig("./test/images/mass_model_gradient_checks.png")
        assert not jnp.any(jnp.array([jnp.any(jnp.isnan(derivatives[x])) for x in derivatives.keys()]))

    def test_redshift_model_matches_gwpop(self):
        Z_gwpop = self.gwpop_redshift_model(jax_to_numpy(self.redshift_grid), **self.redshift_point)
        Z_gravpop = self.gravpop_redshift_model(self.redshift_grid, self.evaluation_point)

        fig = plt.figure(layout="constrained")
        gs = GridSpec(1, 1, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])

        lambs = np.linspace(-6,6,20)
        lambs = [{'lamb' : l} for l in lambs]
        zs = self.redshift_grid['redshift']

        fig,ax = plt.subplots(1)
        for lamb in lambs:
            Z_gwpop_inner = self.gwpop_redshift_model(jax_to_numpy(self.redshift_grid), **lamb)
            Z_gravpop_inner = self.gravpop_redshift_model(self.redshift_grid, lamb)
            dummy_1 = ax.plot(zs, Z_gravpop_inner, color="r", alpha=0.5, label="gravpop")
            dummy_2 = ax.plot(zs, Z_gwpop_inner, color="b", alpha=0.5, label="gwpopulation")

        ax.set_xlabel(r"$z$"); ax.set_ylabel(fr"$P(z | \kappa)$")
        plt.legend(handles=[dummy_1[0], dummy_2[0]])
        plt.title("Probability comparison for redshift models")
        fig.savefig("./test/images/redshift_model_comparison.png")
        assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs

    def test_redshift_model_gradient(self):
        derivatives = model_gradient(self.gravpop_redshift_model, self.redshift_grid, self.redshift_point)
        fig,axes = plt.subplots(nrows=2, figsize=(5,10))

        ax = axes[0]
        zs = self.redshift_grid['redshift']
        ax.plot(zs, derivatives['lamb'], color="r", alpha=0.5)
        ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$\frac{\partial P(z | \kappa)}{\partial \kappa}$")
        ax.set_title(r"$\frac{\partial P(z | \kappa)}{\partial \kappa}$")

        ax = axes[1]
        zs = self.redshift_grid['redshift']
        ax.scatter(zs, jnp.isnan(derivatives['lamb']), color="r", alpha=0.5, marker='o', facecolor='none', edgecolor='b')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not Null", "Null"])
        ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$\frac{\partial P(z | \kappa)}{\partial \kappa}$")
        ax.set_title("Check if gradient is null")

        fig.savefig("./test/images/redshift_model_gradient_checks.png")
        assert not jnp.any(jnp.array([jnp.any(jnp.isnan(derivatives[x])) for x in derivatives.keys()]))



# Define your test class
class TestMassRatioAtNegativeBeta:
    # Define setup method if needed
    def setup_method(self, method):
        ## GLOBAL POINTS
        self.evaluation_point = GLOBAL_DICTIONARY_OF_NICE_VALUES
        self.evaluation_point['beta'] = -1.0
        #self.evaluation_point = GLOBAL_DICTIONARY_OF_EVIL_VALUES
        self.prior = PRIOR_DICT 

        #### Mass Models
        self.mass_params =  ['alpha','beta','mmin','mmax','mpp','sigpp','lam','delta_m']
        self.mass_point = {k:v for k,v in self.evaluation_point.items() if k in self.mass_params}
        self.mass_grid = Grid([Grid1D(name='mass_1', minimum=2, maximum=100, N=200), 
                               Grid1D(name='mass_ratio', minimum=0, maximum=1  , N=200)]).data

        self.gwpop_mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
        self.gravpop_mass_model = gravpop.SmoothedTwoComponentPrimaryMassRatio(primary_mass_name='mass_1')

        #### Redshift Model
        self.redshift_params =  ['lamb']
        self.redshift_point = {k:v for k,v in self.evaluation_point.items() if k in self.redshift_params}

        self.evaluation_point_redshift = GLOBAL_DICTIONARY_OF_NICE_VALUES
        self.redshift_grid = Grid1D(name='redshift', minimum=0, maximum=1.9, N=200).data

        self.gwpop_redshift_model = gwpopulation.models.redshift.PowerLawRedshift(z_max=1.9)
        self.gravpop_redshift_model = gravpop.PowerLawRedshift(z_max=1.9)
    
    # Define teardown method if needed
    def teardown_method(self, method):
        pass
    
    # Write your test methods
    def test_mass_model_matches_gwpop(self):
        Z_gwpop = self.gwpop_mass_model(jax_to_numpy(self.mass_grid), **self.mass_point)
        Z_gravpop = self.gravpop_mass_model(self.mass_grid, self.evaluation_point)

        #fig, axes = plt.subplots(ncols=2, figsize=(10,5))
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax1.contourf(self.mass_grid['mass_1'], self.mass_grid['mass_ratio'], jnp.log(Z_gravpop))
        ax2.contourf(self.mass_grid['mass_1'], self.mass_grid['mass_ratio'], jnp.log(Z_gwpop))
        ax1.set_xlabel(r"$m_1$"); ax2.set_xlabel(r"$m_1$");
        ax1.set_ylabel(r"$q$"); ax2.set_ylabel(r"$q$");
        ax1.set_title(r"$\log P_{gravpop}$")
        ax2.set_title(r"$\log P_{gwpopulation}$")
        ax3.hist(jnp.abs(Z_gravpop/Z_gwpop).flatten(), bins=100);
        ax3.set_title(r"Histogram of $P_{gravpop}/P_{gwpop}$")
        plt.suptitle("Log probability comparison for mass models")
        fig.savefig("./test/images/mass_model_comparison_beta_negative.png")
        assert not jnp.any(jnp.isnan(Z_gravpop)) # TESTING NO NANs

    def test_mass_model_gradient(self):
        derivatives = model_gradient(self.gravpop_mass_model, self.mass_grid, self.mass_point)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        keys = list(derivatives.keys())
        n = len(keys)
        fig = plt.figure(layout="constrained", figsize=(10, 5*n))
        gs = GridSpec(n, 2, figure=fig)

        axes = []
        for i in range(n):
            row_axes = []
            for j in range(2):
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            axes.append(row_axes)

        for i, ax_row in enumerate(axes):
            ax = ax_row[0]
            M1, Q = self.mass_grid["mass_1"], self.mass_grid["mass_ratio"]
            contour = ax.contourf(M1,Q,derivatives[keys[i]])
            cbar = plt.colorbar(contour, ax=ax)
                
            # Add axis labels and title
            ax.set_xlabel(r"$m_1$")
            ax.set_ylabel(r"$q$")
            ax.set_title(f"{keys[i]} \n Gradient of model probability w.r.t to {keys[i]}")
            
            ax = ax_row[1]
            null_values_mask = jnp.isnan(derivatives[keys[i]])
            ax.imshow(null_values_mask, extent=(M1.min(), M1.max(),
                                                 Q.min(), Q.max()),
                      cmap='binary', alpha=0.5, origin='lower', aspect='auto')
            
           #ax.set_aspect('equal')
                
            # Add axis labels and title
            ax.set_xlabel(r"$m_1$")
            ax.set_ylabel(r"$q$")
            ax.set_title(f"{keys[i]} \n Locations where model probability gradient was null")
            
            
        total_null_points = jnp.sum(jnp.array([jnp.sum(jnp.isnan(derivatives[x])) for x in derivatives.keys()]))
        plt.suptitle(f"A total of {total_null_points} points had null gradients")
        fig.savefig("./test/images/mass_model_gradient_checks_beta_negative.png")
        assert not jnp.any(jnp.array([jnp.any(jnp.isnan(derivatives[x])) for x in derivatives.keys()]))
