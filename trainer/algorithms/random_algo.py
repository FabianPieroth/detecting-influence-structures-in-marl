from typing import Dict

import numpy as np

from environment.env_base import MathEnv
from logger.logger_module import Logger
from models.actors_and_critics.random_actor_and_critic import RandomActor, RandomCritic
from trainer.base_trainer import BaseTrainer


class RandomAgent(BaseTrainer):
    def __init__(self, env: MathEnv, config: Dict, logger: Logger):
        super().__init__(env, config, logger)
        self.critic = RandomCritic(env, config)
        self.actor = RandomActor(env, config)
        self.init_impact_measure_objects()

    @classmethod
    def get_trainer_config(cls, project_root_dir: str) -> Dict:
        return {}

    def train(self):
        # create lists to estimate long-term-average return
        average_rewards_log = []
        long_term_average_rewards_log = []

        state_t = self.env.get_current_state()
        # sample joint_action a_0
        actions_t = self.actor.act(state_t)
        # step from t=0 -> t=1

        rewards_t_plus_1 = self.env.step(actions=actions_t)

        for t in range(1, self.config["num_iterations"]):
            state_t_plus_1 = self.env.get_current_state()
            actions_t_plus_1 = self.actor.act(state_t_plus_1)
            # step from t+1 -> t+2
            rewards_t_plus_2 = self.env.step(actions=actions_t_plus_1)
            # logging
            average_rewards_log.append(rewards_t_plus_1.mean())
            long_term_average_rewards_log.append(np.mean(average_rewards_log))
            # print("t = " + str(t) + ", avg. reward: " + str(np.mean(average_rewards_log)))
            self.logger.log_scalar_value(
                "long_term_average_return", float(np.mean(average_rewards_log))
            )

            # Update iteration counter t <- t+1
            self.logger.set_current_time_step(t)
            rewards_t_plus_1 = rewards_t_plus_2
