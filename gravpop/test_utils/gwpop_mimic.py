#### This is a direct rip from gwpop so I can easily test against it

import inspect
import json
import os
import sys
from importlib import import_module

import matplotlib

#matplotlib.use("agg")  # noqa

import dill
import numpy as np
import pandas as pd
from bilby import run_sampler
from bilby.core.prior import Constraint, LogUniform, ConditionalPriorDict
from bilby.core.utils import (
    infer_args_from_function_except_n_args,
    logger,
    decode_bilby_json,
)
from bilby.hyper.model import Model
from bilby_pipe.utils import convert_string_to_dict
from gwpopulation.backend import set_backend
from gwpopulation.conversions import convert_to_beta_parameters as convert_to_beta_parameters_gwpop
from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood
from gwpopulation.models.mass import (
    BrokenPowerLawPeakSmoothedMassDistribution,
    BrokenPowerLawSmoothedMassDistribution,
    MultiPeakSmoothedMassDistribution,
    SinglePeakSmoothedMassDistribution,
    two_component_primary_mass_ratio,
)
from gwpopulation.models.spin import (
    iid_spin,
    iid_spin_magnitude_beta,
    iid_spin_orientation_gaussian_isotropic,
    independent_spin_magnitude_beta,
    independent_spin_orientation_gaussian_isotropic,
)
from gwpopulation.utils import to_numpy
from scipy.stats import gamma
from tqdm.auto import trange

from gwpopulation_pipe import vt_helper
from gwpopulation_pipe.parser import create_parser as create_main_parser
from gwpopulation_pipe.utils import (
    get_path_or_local,
    prior_conversion,
    KNOWN_ARGUMENTS,
    MinimumEffectiveSamplesLikelihood,
)


MODEL_MAP = {
    "two_component_primary_mass_ratio": two_component_primary_mass_ratio,
    "iid_spin": iid_spin,
    "iid_spin_magnitude": iid_spin_magnitude_beta,
    "ind_spin_magnitude": independent_spin_magnitude_beta,
    "iid_spin_orientation": iid_spin_orientation_gaussian_isotropic,
    "two_comp_iid_spin_orientation": iid_spin_orientation_gaussian_isotropic,
    "ind_spin_orientation": independent_spin_orientation_gaussian_isotropic,
    "SmoothedMassDistribution": SinglePeakSmoothedMassDistribution,
    "SinglePeakSmoothedMassDistribution": SinglePeakSmoothedMassDistribution,
    "BrokenPowerLawSmoothedMassDistribution": BrokenPowerLawSmoothedMassDistribution,
    "MultiPeakSmoothedMassDistribution": MultiPeakSmoothedMassDistribution,
    "BrokenPowerLawPeakSmoothedMassDistribution": BrokenPowerLawPeakSmoothedMassDistribution,
}



from dataclasses import dataclass, field
from typing import Union, List, Optional


#DEFAULT_MASS = {'mass' : 'SmoothedMassDistribution', 
#                'redshift' : 'gwpopulation.models.redshift.PowerLawRedshift', 
#               'tilt' : 'iid_spin', 
#                'mag' : 'iid_spin_orientation'}

DEFAULT_MASS = dict(
            mass="SmoothedMassDistribution",#"two_component_primary_mass_ratio",
            mag="iid_spin_magnitude",
            tilt="iid_spin_orientation",
            redshift="gwpopulation.models.redshift.PowerLawRedshift",
        )


@dataclass
class GWPopLoader:
    prior_file : str
    vt_file : str
    posterior_file : Optional[str] = None
    backend : str  = 'jax'
    conversion_function = None
    rate : bool = False
    models : List[str] = field(default_factory=(lambda : DEFAULT_MASS.copy()))
    minimum_mass: float = 2
    maximum_mass : float = 100
    max_redshift : float = 1.9
    vt_models : List[str] = field(default_factory=(lambda : DEFAULT_MASS.copy()))
    vt_ifar_threshold : float = 1.0
    vt_snr_threshold : float = 11.0
    vt_function : str = "injection_resampling_vt"
    enforce_minimum_neffective_per_event : bool = False
    samples_per_posterior : int = 1000
    
    def __post_init__(self):
        self.posteriors = None
        self.prior = None
        self.selection = None
        self.likelihood = None


    def _load_model(self, model):
        args = self
        if model[-5:] == ".json":
            model = get_path_or_local(model)
            with open(model, "r") as ff:
                json_model = json.load(ff, object_hook=decode_bilby_json)
            try:
                cls = getattr(import_module(json_model["module"]), json_model["class"])
                _model = cls(**json_model.get("kwargs", dict()))
                logger.info(f"Using {cls} from {json_model['module']}.")
            except KeyError:
                logger.error(f"Failed to load {model} from json file.")
                raise
        elif "." in model:
            split_model = model.split(".")
            module = ".".join(split_model[:-1])
            function = split_model[-1]
            _model = getattr(import_module(module), function)
            logger.info(f"Using {function} from {module}.")
        elif model in MODEL_MAP:
            _model = MODEL_MAP[model]
            logger.info(f"Using {model}.")
        else:
            raise ValueError(f"Model {model} not found.")
        if inspect.isclass(_model):
            if "redshift" in model.lower():
                kwargs = dict(z_max=args.max_redshift)
            elif "mass" in model.lower():
                kwargs = dict(mmin=args.minimum_mass, mmax=args.maximum_mass)
            else:
                kwargs = dict()
            try:
                _model = _model(**kwargs)
                logger.info(f"Created {model} with arguments {kwargs}")
            except TypeError:
                logger.warning(f"Failed to instantiate {model} with arguments {kwargs}")
                _model = _model()
        return _model
    
    
    def load_prior(self):
        filename = get_path_or_local(self.prior_file)
        hyper_prior = ConditionalPriorDict(filename=filename)
        hyper_prior.conversion_function = convert_to_beta_parameters_gwpop
        if self.rate:
            hyper_prior["rate"] = LogUniform(
                minimum=1e-1,
                maximum=1e3,
                name="rate",
                latex_label="$R$",
                boundary="reflective",
            )
        self.prior = hyper_prior
        return hyper_prior
    
    def load_model(self):
        args = self
        if args.models is None:
            args.models = dict(
                mass="two_component_primary_mass_ratio",
                mag="iid_spin_magnitude",
                tilt="iid_spin_orientation",
                redshift="gwpopulation.models.redshift.PowerLawRedshift",
            )
        if args.backend == "jax":
            from gwpopulation.experimental.jax import NonCachingModel

            cls = NonCachingModel
        else:
            cls = Model
        model = cls([self._load_model(model) for model in args.models.values()])
        self.model = model
        return model
    
    def load_vt(self):
        args = self
        if args.vt_function == "" or args.vt_file == "None":
            result = vt_helper.dummy_selection
            self.selection = result
            return result
        if args.backend == "jax":
            from gwpopulation.experimental.jax import NonCachingModel

            cls = NonCachingModel
        else:
            cls = Model
        vt_model = cls([self._load_model(model) for model in args.vt_models.values()])
        try:
            vt_func = getattr(vt_helper, args.vt_function)
            result = vt_func(
                args.vt_file,
                model=vt_model,
                ifar_threshold=args.vt_ifar_threshold,
                snr_threshold=args.vt_snr_threshold,
            )
            self.selection = result
            return result
        except AttributeError:
            result = vt_helper.injection_resampling_vt(
                vt_file=args.vt_file,
                model=vt_model,
                ifar_threshold=args.vt_ifar_threshold,
                snr_threshold=args.vt_snr_threshold)
            self.selection = result
            return result

    def load_posteriors(self, posterior_file=None):
        import numpy as np
        import pandas as pd
        file = posterior_file or self.posterior_file
        posteriors = np.load(self.posterior_file, allow_pickle=True)
        self.posteriors = posteriors

    def create_likelihood(self, posteriors=None):
        args = self
        if posteriors is None:
            if self.posteriors is None:
                self.load_posteriors()
            posteriors = self.posteriors
        if args.rate:
            if args.enforce_minimum_neffective_per_event:
                raise ValueError(
                    "No likelihood available to enforce convergence of Monte Carlo integrals "
                    "while sampling over rate."
                )
            likelihood_class = RateLikelihood
        elif args.enforce_minimum_neffective_per_event:
            likelihood_class = MinimumEffectiveSamplesLikelihood
        else:
            likelihood_class = HyperparameterLikelihood
        likelihood = likelihood_class(
            posteriors,
            self.model,
            conversion_function=convert_to_beta_parameters_gwpop,
            selection_function=self.selection,
            max_samples=self.samples_per_posterior,
            cupy=False
        )
        self.likelihood = likelihood
        return likelihood