from typing import Dict, List

import numpy as np

from environment.env_base import MathEnv


class BaseCritic(object):
    def __init__(self, env: MathEnv, config: Dict):
        """
        Basic representation of a critic.
        """
        self.config = config
        self.env = env

    def get_ind_q_values_for_state_action(
        self, state: int, action_list: List[List[int]]
    ) -> np.ndarray:
        """
        :param state: state where joint-actions are taken
        :param action_list: List of joint-actions
        :return: shape=len(action_list) Q-values for given joint-actions in given state
        """
        raise RuntimeError(
            "You have to override 'get_ind_q_values_for_state_action()'!"
        )

    def update_parameters_in_one_step(self, **kwargs):
        raise NotImplementedError(
            "You have to override 'update_parameters_in_one_step()'!"
        )
