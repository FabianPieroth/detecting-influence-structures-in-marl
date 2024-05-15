from typing import Dict, List

import numpy as np

from environment.env_base import MathEnv


class BaseActor(object):
    def __init__(self, env: MathEnv, config: Dict):
        """
        Basic representation of an actor.
        """
        self.config = config
        self.env = env

    def get_policy_distribution(self) -> np.ndarray:
        """
        :return: Full policy distribution over all states and joint-actions
        """
        policy_distribution = np.zeros(
            (self.env.get_joint_action_size(), self.env.state_size)
        )
        for state in range(self.env.state_size):
            policy_distribution[:, state] = self.get_state_policy_distribution(state)
        return policy_distribution

    def get_state_policy_distribution(self, state: int) -> np.ndarray:
        """
        :param state:
        :return: Policy Distribution over all joint-actions in the given state
        """
        raise NotImplementedError(
            "You have to override 'get_state_policy_distribution()'!"
        )

    def act(self, state: int) -> List[int]:
        raise NotImplementedError(
            "You have to override 'act()'. This method returns a list of actions in the given state."
        )
