####################################################################################################################
####  THE CODE BELOW IS TAKEN ALMOST ENTIRELY WORD FOR WORD FROM 
####  https://colab.research.google.com/drive/1w2tI1-1LWzPSdG_jE0FXzdrs6VAsJwhv?usp=sharing
####  It is an implementation by gihub user flaviovdf (https://github.com/flaviovdf)
####  of the paper "A simple approximation for the bivariate normal integral" by Wen-Jen Tsay & Peng-Hsuan Ke
####  https://doi.org/10.1080/03610918.2021.1884718
####################################################################################################################

import jax.numpy as jnp
import jax
from jax import jit
from jax.scipy.special import erf

from jax.scipy.stats.norm import cdf as cdf1d

@jax.jit
def case1(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 + (line12 * line21) - (line22 * (line31 + line32))


@jax.jit
def case2(p, q):
    return cdf1d(p) * cdf1d(q)


@jax.jit
def case3(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12


@jax.jit
def case4(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)


@jax.jit
def case5(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 - (line12 * line21) + line22 * (line31 + line32)


@jit
def binorm_lower(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    p = (x1 - mu1) / sigma1
    q = (x2 - mu2) / sigma2

    a = -rho / jnp.sqrt(1 - rho * rho)
    b = p / jnp.sqrt(1 - rho * rho)
    
    cond1 = (a > 0) & (a * q + b >= 0)
    cond2 = (a == 0)
    cond3 = (a > 0) & (a * q + b < 0)
    cond4 = (a < 0) & (a * q + b >= 0)
    cond5 = (a < 0) & (a * q + b < 0)
    
    #print(cond1, cond2, cond3, cond4, cond5)
    
    return jnp.where(cond1, case1(p, q, rho, a, b),
             jnp.where(cond2, case2(p, q), 
                      jnp.where(cond3, case3(p, q, rho, a, b),
                               jnp.where(cond4, case4(p, q, rho, a, b),
                                        jnp.where(cond5, case5(p, q, rho, a, b), 
                                                 jnp.where(jnp.isposinf(p) & jnp.isposinf(q), jnp.ones_like(x1), 
                                                           jnp.zeros_like(x1)))))))

@jit
def binorm_upper(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    p = -(x1 - mu1) / sigma1
    q = -(x2 - mu2) / sigma2

    a = -rho / jnp.sqrt(1 - rho * rho)
    b = p / jnp.sqrt(1 - rho * rho)
    
    cond1 = (a > 0) & (a * q + b >= 0)
    cond2 = (a == 0)
    cond3 = (a > 0) & (a * q + b < 0)
    cond4 = (a < 0) & (a * q + b >= 0)
    cond5 = (a < 0) & (a * q + b < 0)
    
    #print(cond1, cond2, cond3, cond4, cond5)
    
    return jnp.where(cond1, case1(p, q, rho, a, b),
             jnp.where(cond2, case2(p, q), 
                      jnp.where(cond3, case3(p, q, rho, a, b),
                               jnp.where(cond4, case4(p, q, rho, a, b),
                                        jnp.where(cond5, case5(p, q, rho, a, b), 
                                                 jnp.where(jnp.isneginf(p) & jnp.isneginf(q), jnp.ones_like(x1), 
                                                           jnp.zeros_like(x1)))))))


@jit
def mvnorm2d_using_lower(mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, rho):
    #X1 = binorm_lower(b_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    #X0 = binorm_upper(a_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    #return X0-X1
    upper_right_lower = binorm_lower(b_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X < b
    upper_left_lower  = binorm_lower(a_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X < a_up
    lower_left_lower  = binorm_lower(a_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X < a
    lower_right_lower = binorm_lower(b_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X < b_down
    ## (b - b_down) + (a - a_up)
    return (upper_right_lower - lower_right_lower) + (lower_left_lower - upper_left_lower)

@jit
def mvnorm2d_using_upper(mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, rho):
    #X1 = binorm_lower(b_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    #X0 = binorm_upper(a_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    #return X0-X1
    upper_right_upper = binorm_upper(b_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X > b
    upper_left_upper  = binorm_upper(a_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X > a_up
    lower_left_upper  = binorm_upper(a_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X > a
    lower_right_upper = binorm_upper(b_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)  ## X > b_down
    ## (b - b_down) + (a - a_up)
    return (upper_right_upper - lower_right_upper) + (lower_left_upper - upper_left_upper)


@jit
def mvnorm2d(mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, rho):
    return jnp.where((mu_1 <= b_1) & (mu_2 <= b_2),  mvnorm2d_using_upper(mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, rho), 
                        jnp.where((mu_1 >= a_1) & (mu_2 >= a_2), mvnorm2d_using_lower(mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, rho),
                                 jnp.where((mu_1 <= a_1) & (mu_2 >= b_2), mvnorm2d_using_lower(b_1 + a_1 - mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, -rho),
                                          mvnorm2d_using_upper(b_1 + a_1 - mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, -rho))))



@jit
def mvnorm2d_deprecated(mu_1, mu_2, sigma_1, sigma_2, a_1, a_2, b_1, b_2, rho):
    #X1 = binorm_lower(b_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    #X0 = binorm_upper(a_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    #return X0-X1
    upper_right_lower = binorm_lower(b_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    upper_left_lower  = binorm_lower(a_1, b_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    lower_left_lower  = binorm_lower(a_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    lower_right_lower = binorm_lower(b_1, a_2, mu_1, mu_2, sigma_1, sigma_2, rho)
    return (upper_right_lower - lower_right_lower) - (upper_left_lower - lower_left_lower)


