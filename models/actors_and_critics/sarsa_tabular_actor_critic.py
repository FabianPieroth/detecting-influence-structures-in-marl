from typing import Dict, List, Tuple

import numpy as np

import global_utils
from environment.env_base import MathEnv
from models.base_actor import BaseActor
from models.base_critic import BaseCritic


class SarsaTabularCritic(BaseCritic):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)
        self.num_agents = env.num_agents
        self.size_action_space = env.size_action_space
        self.size_state_action_emb = env.size_state_action_emb

        self.q_learning_rate = self.config["q_learning_rate"]
        self.ltr_learning_rate = self.config["ltr_learning_rate"]
        self.init_q_table_type = self.config["init_q_table_type"]
        self.ltr_estimate = self._init_ltr_estimate()

        self.q_table = self._init_q_table(self.config["init_q_table_type"])

    @staticmethod
    def _init_ltr_estimate():
        return 0.0

    def _init_q_table(self, init_q_table_type="ones") -> np.array:
        if init_q_table_type == "ones":
            return np.ones(self._get_state_action_shape())
        elif init_q_table_type == "random":
            return np.random.random(self._get_state_action_shape())
        elif init_q_table_type == "zeros":
            return np.zeros(self._get_state_action_shape())

    def _get_state_action_shape(self) -> Tuple:
        size_joint_action_space = [
            self.env.size_action_space for i in range(self.env.num_agents)
        ]
        # add state_size
        size_joint_action_space.insert(0, self.env.state_size)
        return tuple(size_joint_action_space)

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

    def _update_q_table(self, state_t: int, actions_t: List, td_error: float):
        self.q_table[self._get_state_joint_action_index(state_t, actions_t)] += (
            self.q_learning_rate * td_error
        )

    def _update_ltr_estimate(self, td_error: float):
        self.ltr_estimate += self.ltr_learning_rate * td_error

    @staticmethod
    def _get_state_joint_action_index(state: int, actions: List) -> Tuple:
        actions_copy = actions.copy()
        # add state index at the front
        actions_copy.insert(0, state)
        return tuple(actions_copy)

    def _get_q_value(self, state: int, actions: List[int]) -> float:
        return self.q_table[self._get_state_joint_action_index(state, actions)]

    def critic_step(
        self,
        state_t: int,
        actions_t: List[int],
        state_t_plus_1: int,
        actions_t_plus_1: List[int],
        rewards_t_plus_1: np.ndarray,
    ):
        td_error = self._calculate_td_error(
            state_t, actions_t, state_t_plus_1, actions_t_plus_1, rewards_t_plus_1
        )
        self._update_q_table(state_t, actions_t, td_error)
        self._update_ltr_estimate(td_error)

    def get_ind_q_values_for_state_action(
        self, state: int, action_list: List[List[int]]
    ) -> np.ndarray:
        """
        :param state: state where joint-actions are taken
        :param action_list: List of joint-actions
        :return: shape=len(action_list) Q-values for given joint-actions in given state
        """
        q_values = np.zeros(len(action_list))
        for k, actions in enumerate(action_list):
            q_values[k] = self._get_q_value(state, actions)
        return q_values


class IndSarsaTabularCritic(SarsaTabularCritic):
    def __init__(self, env: MathEnv, config: Dict, agent_id: int):
        super().__init__(env, config)
        self.agent_id = agent_id

    def update_parameters_in_one_step(self, **kwargs):
        # parameter updates
        self.critic_step(
            kwargs["state_t"],
            kwargs["actions_t"],
            kwargs["state_t_plus_1"],
            kwargs["actions_t_plus_1"],
            np.array(kwargs["rewards_t_plus_1"][self.agent_id]),
        )


class SarsaTabularActor(BaseActor):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

        self.epsilon = self.config["epsilon"]
        self.q_table = None

    def act(self, state: int) -> List[int]:
        if np.random.rand() > self.epsilon:
            action = np.unravel_index(
                np.argmax(self.q_table[state, :]), self.q_table[state, :].shape
            )
        else:
            action = np.random.randint(
                0, self.env.size_action_space, size=self.env.num_agents
            )
        return list(action)

    def _update_epsilon(self):
        self.epsilon *= self.config["epsilon_decay"]

    def update_actor_q_table(self, q_table: np.ndarray):
        self.q_table = q_table
        self._update_epsilon()
