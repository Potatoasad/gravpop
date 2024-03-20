import jax.numpy as jnp
from jax import grad

class BetaDistributionConverter:
    def __init__(self):
        pass
    
    @staticmethod
    def _convert_to_beta_parameters(parameters):
        added_keys = []
        converted = parameters.copy()

        def _convert(suffix):
            alpha = f"alpha_chi{suffix}"
            beta = f"beta_chi{suffix}"
            mu = f"mu_chi{suffix}"
            sigma = f"sigma_chi{suffix}"
            amax = f"amax{suffix}"

            if alpha not in parameters or beta not in parameters:
                needed = True
            elif converted[alpha] is None or converted[beta] is None:
                needed = True
            else:
                needed = False
                done = True

            if needed:
                if mu in converted and sigma in converted:
                    done = True
                    alpha_val, beta_val, _ = BetaDistributionConverter.mu_var_max_to_alpha_beta_max(
                        converted[mu], converted[sigma], converted[amax]
                    )
                    converted[alpha], converted[beta] = alpha_val, beta_val
                    added_keys.extend([alpha, beta])
                else:
                    done = False
            return done

        done = False
        for suffix in ["_1", "_2", ""]:
            done |= _convert(suffix)

        return converted, added_keys

    @staticmethod
    def alpha_beta_max_to_mu_var_max(alpha, beta, amax):
        mu = alpha / (alpha + beta) * amax
        var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax ** 2
        return mu, var, amax

    @staticmethod
    def mu_var_max_to_alpha_beta_max(mu, var, amax):
        mu /= amax
        var /= amax ** 2
        alpha = (mu ** 2 * (1 - mu) - mu * var) / var
        beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
        return alpha, beta, amax

    def convert_parameters(self, parameters_dict, remove=True):
        converted_dict, added_keys = self._convert_to_beta_parameters(parameters_dict)
        if remove:
            for key in added_keys:
                converted_dict.pop(key, None)
        return converted_dict