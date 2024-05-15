from typing import Dict, List, Tuple

import numpy as np

import global_utils
from environment.env_base import MathEnv
from logger.logger_module import Logger
from models.actors_and_critics.sarsa_tabular_actor_critic import (
    SarsaTabularActor,
    SarsaTabularCritic,
)
from trainer.base_trainer import BaseTrainer


class SarsaTabularAlgorithm(BaseTrainer):
    def __init__(self, env: MathEnv, config: dict, logger: Logger):
        super().__init__(env, config, logger)

        self.actor = self._init_actor()
        self.critic = self._init_critic()
        self.actor.update_actor_q_table(self.critic.q_table)
        self.init_impact_measure_objects()

    @classmethod
    def get_trainer_config(cls, project_root_dir: str) -> Dict:
        sarsa_config = cls._get_sarsa_config(project_root_dir)
        return sarsa_config

    @staticmethod
    def _get_sarsa_config(project_root_dir) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/models/configs/sarsa_tabular_config.json"
        )

    def _init_actor(self) -> SarsaTabularActor:
        return SarsaTabularActor(self.env, self.config)

    def _init_critic(self) -> SarsaTabularCritic:
        return SarsaTabularCritic(self.env, self.config)

    def train(self):
        self.logger.start_logger(self.config)
        # create lists to estimate long-term-average return
        average_rewards_log = []
        long_term_average_rewards_log = []
        # sample joint_action a_0
        state_t = self.env.get_current_state()
        actions_t = self.actor.act(state_t)  # get a_0
        for t in range(1, self.config["num_iterations"]):
            # step from t=0 -> t=1
            rewards_t_plus_1 = self.env.step(actions=actions_t)
            state_t_plus_1 = self.env.get_current_state()
            actions_t_plus_1 = self.actor.act(state_t_plus_1)
            self.critic.critic_step(
                state_t, actions_t, state_t_plus_1, actions_t_plus_1, rewards_t_plus_1
            )
            self.actor.update_actor_q_table(self.critic.q_table)

            # logging
            average_rewards_log.append(rewards_t_plus_1.mean())
            long_term_average_rewards_log.append(np.mean(average_rewards_log))
            print(
                "t = " + str(t) + ", avg. reward: " + str(np.mean(average_rewards_log))
            )
            self.logger.log_scalar_value(
                "long_term_average_return", float(np.mean(average_rewards_log))
            )

            # t -> t+1
            self.logger.set_current_time_step(t)
            state_t, actions_t = state_t_plus_1, actions_t_plus_1
