from typing import Dict

import global_utils
from environment.env_base import MathEnv
from logger.logger_module import Logger
from trainer.algorithms.random_algo import RandomAgent
from trainer.algorithms.sarsa_linear_algo import SarsaLinearAlgorithm
from trainer.algorithms.sarsa_tabular_algo import SarsaTabularAlgorithm
from trainer.algorithms.zhang_analytical_critic_algo import (
    ZhangAnalyticalCriticAlgorithm,
)
from trainer.algorithms.zhang_centralized_algo import ZhangCentralizedAlgorithm
from trainer.algorithms.zhang_decentralized_algo import ZhangDecentralizedAlgorithm
from trainer.base_trainer import BaseTrainer


class TrainerEnvInitializer(object):
    def __init__(self):
        pass

    @staticmethod
    def get_analytical_critic_config(project_root_dir: str) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/models/configs/zhang_analytical_config.json"
        )

    @staticmethod
    def get_env_config(env_type: str, project_root_dir: str) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/environment/configs/" + env_type
        )

    @staticmethod
    def get_env(config: Dict) -> MathEnv:
        return MathEnv(config)

    @staticmethod
    def get_trainer_configs(trainer_type: str, project_root_dir: str) -> Dict:
        if trainer_type == "zhang_decentralized":
            return ZhangDecentralizedAlgorithm.get_trainer_config(project_root_dir)
        elif trainer_type == "zhang_centralized":
            return ZhangCentralizedAlgorithm.get_trainer_config(project_root_dir)
        elif trainer_type == "zhang_analytical_critic":
            return ZhangAnalyticalCriticAlgorithm.get_trainer_config(project_root_dir)
        elif trainer_type == "sarsa_tabular":
            return SarsaTabularAlgorithm.get_trainer_config(project_root_dir)
        elif trainer_type == "sarsa_linear":
            return SarsaLinearAlgorithm.get_trainer_config(project_root_dir)
        elif trainer_type == "random":
            return RandomAgent.get_trainer_config(project_root_dir)
        else:
            raise RuntimeError("You have to select a valid trainer type!")

    @staticmethod
    def get_trainer(
        trainer_type: str, env: MathEnv, config: Dict, logger: Logger
    ) -> BaseTrainer:
        if trainer_type == "zhang_decentralized":
            return ZhangDecentralizedAlgorithm(env, config, logger)
        elif trainer_type == "zhang_centralized":
            return ZhangCentralizedAlgorithm(env, config, logger)
        elif trainer_type == "zhang_analytical_critic":
            return ZhangAnalyticalCriticAlgorithm(env, config, logger)
        elif trainer_type == "sarsa_tabular":
            return SarsaTabularAlgorithm(env, config, logger)
        elif trainer_type == "sarsa_linear":
            return SarsaLinearAlgorithm(env, config, logger)
        elif trainer_type == "random":
            return RandomAgent(env, config, logger)
        else:
            raise RuntimeError("You have to select a valid trainer type!")

    @staticmethod
    def get_impact_estimator_config(project_root_dir: str) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/impact_measure/configs/impact_estimator_config.json"
        )

    @staticmethod
    def get_impact_sampler_config(project_root_dir: str) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/impact_measure/configs/impact_sampler_config.json"
        )
