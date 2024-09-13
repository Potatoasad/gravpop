import jax
import jax.numpy as jnp
import numpy as np
import astropy
from astropy.cosmology import Planck15
from ..generic import SampledPopulationModel, RedshiftPopulationModel
from ..utils import box

class Redshift(SampledPopulationModel, RedshiftPopulationModel):
    """A class representing redshifts and their associated probability distributions.

    Attributes:
        z_max (float): Maximum redshift.
        lamb_min (float): Minimum value for the hyperparameter.
        lamb_max (float): Maximum value for the hyperparameter.
        var_name (str): Name of the redshift variable.
        hyper_var_name (str): Name of the hyperparameter variable.
        lambs (numpy.ndarray): Array of hyperparameter values.
        zs (numpy.ndarray): Array of redshift values.
        dVdz_values (numpy.ndarray): Array of differential comoving volume values.

    Methods:
        dVdz(data): Computes the differential comoving volume for given redshift data.
        normalization(params): Computes the normalization factor for the probability distribution.
        probability(data, params): Computes the probability distribution for given data and parameters.
        __call__(data, params): Calls the probability method.
    """
    def __init__(self, var_names=['redshift'], hyper_var_names=['lamb'], z_max=3, lamb_min=-10, lamb_max=10):
        """Initialize the Redshift class.

        Args:
            z_max (float, optional): Maximum redshift (default is 3).
            lamb_min (float, optional): Minimum value for the hyperparameter (default is -10).
            lamb_max (float, optional): Maximum value for the hyperparameter (default is 10).
        """
        self.z_max = z_max
        self.var_names = var_names
        self.hyper_var_names = hyper_var_names
        self.var_name = var_names[0]
        self.hyper_var_name = hyper_var_names[0]
        self.lamb_min = lamb_min
        self.lamb_max = lamb_max
        self.lambs = jnp.linspace(self.lamb_min, self.lamb_max, 1000)
        self._precomputed_normalization = None
        
        self.zs = jnp.linspace(0, z_max, 1000)
        self.dVdz_values = Planck15.differential_comoving_volume(self.zs).value * 4 * np.pi

    def dVdz(self, data):
        """Compute the differential comoving volume.

        Args:
            data (dict): Dictionary containing the redshift data.

        Returns:
            numpy.ndarray: Array of differential comoving volume values.
        """
        return jnp.interp(data[self.var_name], self.zs, self.dVdz_values)

    def _normalization(self, params):
        """Compute the normalization factor.

        Args:
            params (dict): Dictionary containing the hyperparameter value.

        Returns:
            float: Normalization factor.
        """
        psi_of_z = self.psi_of_z({self.var_name: self.zs}, params)
        norm = jax.scipy.integrate.trapezoid(psi_of_z * self.dVdz_values / (1 + self.zs), self.zs)
        return norm

    def precomputed_normalization_array(self):
        """Precompute the normalization array."""
        if self._precomputed_normalization is None:
            self._precomputed_normalization = jnp.array([self._normalization({'lamb':lamb}) for lamb in self.lambs])
            return self._precomputed_normalization
        else:
            return self._precomputed_normalization

    def normalization(self, params):
        """Compute the normalization factor for given parameters.

        Args:
            params (dict): Dictionary containing the hyperparameter value.

        Returns:
            float: Normalization factor.
        """
        norms = self.precomputed_normalization_array()
        return jnp.interp(params[self.hyper_var_name], self.lambs, norms)

    def probability(self, data, params):
        """Compute the probability distribution.

        Args:
            data (dict): Dictionary containing the redshift data.
            params (dict): Dictionary containing the hyperparameter value.

        Returns:
            numpy.ndarray: Array of probability values.
        """
        normalisation = self._normalization(params)
        prob = self.dVdz(data) * self.psi_of_z(data, params) / (1 + data[self.var_name])
        in_bounds = box(data[self.var_name], 0, self.z_max)
        #return (prob  / normalisation) * in_bounds
        #print(prob, in_bounds,normalisation)
        #print(prob *in_bounds / normalisation)
        return prob *in_bounds / normalisation
        #return prob * in_bounds# * jnp.abs(params[self.hyper_var_name])

    def __call__(self, data, params):
        """Call the probability method.

        Args:
            data (dict): Dictionary containing the redshift data.
            params (dict): Dictionary containing the hyperparameter value.

        Returns:
            numpy.ndarray: Array of probability values.
        """
        return jnp.nan_to_num(self.probability(data, params),nan=0)

    def total_four_volume(self, params, analysis_time):
        return self.normalization(params) / 1e9 * analysis_time


class PowerLawRedshift(Redshift):
    """A subclass of Redshift representing power law redshift distributions.

    Methods:
        psi_of_z(data, params): Computes the power law function.
    """
    def psi_of_z(self, data, params):
        """Compute the power law function.

        Args:
            data (dict): Dictionary containing the redshift data.
            params (dict): Dictionary containing the hyperparameter value.

        Returns:
            numpy.ndarray: Array of power law values.
        """
        return ((1+data[self.var_name])**(params[self.hyper_var_name]))


