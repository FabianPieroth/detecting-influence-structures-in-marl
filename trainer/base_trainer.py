from typing import Dict

from environment.env_base import MathEnv
from environment.env_markov_chain_determiner import MarkovChainDeterminer
from impact_measure.impact_determiner import (
    AnalyticalImpactDeterminer,
    ImpactDeterminer,
)
from impact_measure.impact_estimator import ImpactEstimator
from impact_measure.impact_sampler import ImpactSampler
from logger.logger_module import Logger
from models.base_actor import BaseActor


class BaseTrainer(object):
    def __init__(self, env: MathEnv, config: Dict, logger: Logger):
        """
        Basic representation of a trainer.
        """
        self.env = env
        self.config = config
        self.actor = BaseActor(env, config)
        self.mc_determiner = MarkovChainDeterminer(self.env)
        self.logger = logger
        self.logger.log_dependency_matrix(self.env.get_dependency_matrix())
        self.impact_sampler = ImpactSampler(
            env, config, self.mc_determiner, config["ind_critic_type"]
        )
        self.analytical_impact_sampler = ImpactSampler(
            env, config, self.mc_determiner, "zhang_analytical_critic"
        )

        self.analytical_impact_determiner = None
        self.impact_determiner = None
        self.impact_estimator = None

    def init_impact_measure_objects(self):
        self._init_analytical_impact_determiner()
        self._init_impact_determiner()
        self._init_impact_estimator()

    def _init_analytical_impact_determiner(self):
        self.analytical_impact_determiner = AnalyticalImpactDeterminer(
            self.env,
            self.actor,
            self.analytical_impact_sampler,
            self.mc_determiner,
            self.logger,
        )

    def _init_impact_determiner(self):
        self.impact_determiner = ImpactDeterminer(
            self.env, self.actor, self.impact_sampler, self.mc_determiner, self.logger
        )

    def _init_impact_estimator(self):
        self.impact_estimator = ImpactEstimator(
            self.env, self.config, self.impact_sampler, self.logger
        )

    @classmethod
    def get_trainer_config(cls, project_root_dir: str) -> Dict:
        raise RuntimeError(
            "You have to override 'get_trainer_config()'. This method should return all needed configs."
        )

    def train(self):
        raise RuntimeError(
            "You have to override 'train()'. This method should implement the training phase."
        )
