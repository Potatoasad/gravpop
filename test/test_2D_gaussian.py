import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import numpyro.distributions as dist
from typing import Dict
import jax
from gravpop import * 
import matplotlib.pyplot as plt
import scipy
import pytest

def test_covariance():
	sig_1, sig_2, rho = 1.2, 3.1, 0.1
	Sigma_mat = np.array([[sig_1**2, sig_1*sig_2*rho],[sig_1*sig_2*rho, sig_2**2]])
	Sigma_test = CovarianceMatrix2D(sig_1, sig_2, rho)