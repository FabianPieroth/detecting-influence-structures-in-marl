from typing import Dict

import numpy as np

import global_utils
from environment.env_base import MathEnv
from logger.logger_module import Logger
from models.actors_and_critics.analytical_critic import (
    AnalyticalCritic,
    ZhangActorForAnalyticalCritic,
)
from trainer.base_trainer import BaseTrainer


class ZhangAnalyticalCriticAlgorithm(BaseTrainer):
    def __init__(self, env: MathEnv, config: dict, logger: Logger):
        super().__init__(env, config, logger)

        self.actor = self._init_actor()
        self.critic = self._init_critic()
        self.init_impact_measure_objects()

    @classmethod
    def get_trainer_config(cls, project_root_dir: str) -> Dict:
        analytical_config = cls._get_analytical_config(project_root_dir)
        return analytical_config

    @staticmethod
    def _get_analytical_config(project_root_dir) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/models/configs/zhang_analytical_config.json"
        )

    def _init_actor(self) -> ZhangActorForAnalyticalCritic:
        return ZhangActorForAnalyticalCritic(self.env, self.config)

    def _init_critic(self) -> AnalyticalCritic:
        return AnalyticalCritic(self.env, self.config, self.mc_determiner)

    def train(self):
        # create lists to estimate long-term-average return
        average_rewards_log, long_term_average_rewards_log = [], []

        # sample joint_action a_0
        state_t = self.env.get_current_state()
        actions_t = self.actor.act(state_t)

        im_analytical, tim_analytical = (
            self.analytical_impact_determiner.calc_analytical_impact_estimates()
        )

        self.critic.update_ltr()
        self.impact_sampler.update_ind_critics()

        for t in range(1, self.config["num_iterations"]):
            self.impact_estimator.update_impact_estimations(state_t, actions_t, t)
            # step from t -> t+1
            rewards_t_plus_1 = self.env.step(actions=actions_t)
            state_t_plus_1 = self.env.get_current_state()
            actions_t_plus_1 = self.actor.act(state_t_plus_1)
            # parameter updates
            # self.critic.update_state_action_markov_chain_params(
            #    self.actor.get_state_action_policy_distribution()
            # )
            # self.actor.actor_step(state_t, actions_t, self.critic)

            # ####################################### #
            # ############### LOGGING ############### #
            # ####################################### #
            average_rewards_log.append(rewards_t_plus_1.mean())
            long_term_average_rewards_log.append(np.mean(average_rewards_log))
            print(
                "t = "
                + str(t)
                + ", avg. reward: "
                + str(np.around(np.mean(average_rewards_log), decimals=2))
            )
            self.logger.log_scalar_value(
                "long_term_average_return", float(np.mean(average_rewards_log))
            )
            tim_estimate = self.impact_estimator.get_tim_estimates()
            im_estimate = self.impact_estimator.get_im_estimates()

            self.logger.log_impact_metrics(
                "analytical",
                "analytical",
                im_analytical,
                im_estimate,
                tim_analytical,
                tim_estimate,
            )

            # Update iteration counter t <- t+1
            self.logger.set_current_time_step(t)
            state_t, actions_t = state_t_plus_1, actions_t_plus_1
