from typing import Dict, List

import numpy as np

from environment.env_base import MathEnv
from models.base_actor import BaseActor
from models.base_critic import BaseCritic


class RandomCritic(BaseCritic):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

    def get_ind_q_values_for_state_action(
        self, state: int, action_list: List[List[int]]
    ) -> np.ndarray:
        return np.random.uniform(size=len(action_list))


class IndRandomCritic(RandomCritic):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

    def update_parameters_in_one_step(self, *args):
        pass


class RandomActor(BaseActor):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

    def get_state_policy_distribution(self, state: int) -> np.ndarray:
        """
        :param state:
        :return: Policy Distribution over all joint-actions in the given state
        """
        return (
            np.ones(self.env.get_joint_action_size()) / self.env.get_joint_action_size()
        )

    def act(self, state: int) -> List[int]:
        return list(
            np.random.randint(self.env.size_action_space, size=self.env.num_agents)
        )
