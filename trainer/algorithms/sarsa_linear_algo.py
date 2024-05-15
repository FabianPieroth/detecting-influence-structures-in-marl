from typing import Dict, List

import numpy as np

import global_utils
from environment.env_base import MathEnv
from logger.logger_module import Logger
from trainer.base_trainer import BaseTrainer


class SarsaLinearAlgorithm(object):
    def __init__(self, env: MathEnv, config: dict, logger: Logger):
        self.env = env
        self.config = config
        self.logger = logger

        self.epsilon = self.config["epsilon"]
        self.q_learning_rate = self.config["q_learning_rate"]
        self.ltr_learning_rate = self.config["ltr_learning_rate"]
        self.q_function_weights = self._init_q_function_weights()
        self.ltr_estimate = self._init_ltr_estimate()

    @classmethod
    def get_trainer_config(cls, project_root_dir: str) -> Dict:
        sarsa_config = cls._get_sarsa_config(project_root_dir)
        return sarsa_config

    @staticmethod
    def _get_sarsa_config(project_root_dir) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/models/configs/sarsa_linear_config.json"
        )

    def _init_q_function_weights(self) -> np.array:
        return np.random.uniform(
            low=-1.0, high=1.0, size=self.env.size_state_action_emb
        )

    @staticmethod
    def _init_ltr_estimate():
        return 0.0

    def _act(self, state: int) -> List:
        if np.random.rand() > self.epsilon:
            action = self._get_greedy_action(state)
        else:
            action = np.random.randint(
                0, self.env.size_action_space, size=self.env.num_agents
            )
        return list(action)

    def _get_greedy_action(self, state: int) -> List:
        all_state_actions_emb_for_state = self.env.get_all_state_action_emb_for_state(
            state
        )
        q_values = self._calculate_state_q_values(all_state_actions_emb_for_state)
        return self._get_greedy_action_from_q_values(q_values)

    def _calculate_state_q_values(self, state_actions_embeddings: np.array) -> np.array:
        return (state_actions_embeddings * self.q_function_weights[np.newaxis, :]).sum(
            axis=1
        )

    def _get_greedy_action_from_q_values(self, q_values: np.array) -> List:
        return global_utils.get_number_representation_from_int(
            index=int(np.argmax(q_values)),
            decimal_base=self.env.size_action_space,
            length_representation=self.env.num_agents,
        )

    def _calculate_td_error(
        self,
        state_t: int,
        actions_t: List,
        state_t_plus_1: int,
        actions_t_plus_1: List,
        rewards_t_plus_1: np.array,
    ):
        q_difference = self._get_q_value(
            state_t_plus_1, actions_t_plus_1
        ) - self._get_q_value(state_t, actions_t)
        return np.mean(rewards_t_plus_1) - self.ltr_estimate + q_difference

    def _update_q_function_weights(
        self, state_t: int, actions_t: List, td_error: float
    ):
        self.q_function_weights += (
            self.q_learning_rate
            * td_error
            * self.env.get_state_action_emb(state_t, actions_t)
        )

    def _update_ltr_estimate(self, td_error: float):
        self.ltr_estimate += self.ltr_learning_rate * td_error

    def _get_q_value(self, state: int, actions: List) -> float:
        state_actions_emb = self.env.get_state_action_emb(state, actions)
        return state_actions_emb @ self.q_function_weights

    def _update_epsilon(self):
        self.epsilon *= self.config["epsilon_decay"]

    def train(self):
        self.logger.start_logger(self.config)
        # create lists to estimate long-term-average return
        average_rewards_log = []
        long_term_average_rewards_log = []
        # sample joint_action a_0
        state_t = self.env.get_current_state()
        actions_t = self._act(state_t)  # get a_0
        for t in range(1, self.config["num_iterations"]):
            # step from t=0 -> t=1
            rewards_t_plus_1 = self.env.step(actions=actions_t)  # r_1
            state_t_plus_1 = self.env.get_current_state()
            actions_t_plus_1 = self._act(state_t_plus_1)
            td_error = self._calculate_td_error(
                state_t, actions_t, state_t_plus_1, actions_t_plus_1, rewards_t_plus_1
            )
            self._update_q_function_weights(state_t, actions_t, td_error)
            self._update_ltr_estimate(td_error)

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
            self._update_epsilon()
            # update learning_rate: probably not needed!
