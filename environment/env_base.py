from typing import Dict, List, Tuple

import numpy as np

import global_utils


class MathEnv(object):
    def __init__(self, config: Dict):
        self.config = config
        self.num_agents = config["num_agents"]
        self.state_size = config["state_size"]
        self.size_action_space = config["size_action_space"]
        self.size_state_emb_for_policy = config["size_state_emb_for_policy"]
        self.size_state_action_emb = config["size_state_action_emb"]

        self.dependency_matrix = self._init_dependency_matrix()
        self.transition_matrix = self._init_transition_matrix()
        self.expected_reward_matrix = self._init_expected_reward_matrix()
        self.state_emb_for_policy = self._init_state_emb_for_policy()
        self.state_actions_emb = self._init_state_actions_emb()

        self.current_state = self._set_initial_state()

    def _set_initial_state(self) -> int:
        return np.random.randint(self.state_size)

    def _get_state_action_state_size(self) -> Tuple[int, int]:
        return self.get_state_action_size(), self.state_size

    def _get_state_action_num_agents_size(self) -> Tuple[int, int]:
        return self.get_state_action_size(), self.num_agents

    def _get_state_action_emb_size(self) -> Tuple[int, int]:
        return self.get_state_action_size(), self.size_state_action_emb

    def get_state_action_size(self) -> int:
        return self.get_joint_action_size() * self.state_size

    def get_joint_action_size(self) -> int:
        return self.size_action_space ** self.num_agents

    def get_transition_matrix(self) -> np.ndarray:
        return self.transition_matrix

    def get_expected_reward_matrix(self) -> np.ndarray:
        return self.expected_reward_matrix

    def get_dependency_matrix(self) -> np.ndarray:
        return self.dependency_matrix

    def _init_transition_matrix(self) -> np.ndarray:
        transition_matrix = np.random.uniform(size=self._get_state_action_state_size())
        # add constant to ensure that MDP is ergodic
        transition_matrix += 10e-5
        transition_matrix = global_utils.normalize_rows(transition_matrix)
        if np.min(self.dependency_matrix) == 1:
            return transition_matrix
        else:
            # This ensures that the impact measure is 0 if the entry in the dependency matrix is 0
            transition_matrix[:, :] = transition_matrix[0, :]
            return transition_matrix

    def _init_dependency_matrix(self) -> np.ndarray:
        if self.config["dependency_matrix_type"] == "full":
            return np.ones((self.num_agents, self.num_agents), dtype=int)
        elif (
            self.config["dependency_matrix_type"] == "connectivity_ratio"
            or self.config["dependency_matrix_type"] == "fix_num_additional_edges"
        ):
            return global_utils.sample_directed_matrix(
                self.config["dependency_matrix_type"],
                self.config["env_num_dependency_edges"],
                self.config["connectivity_ratio"],
                self.num_agents,
            )
        elif self.config["dependency_matrix_type"] == "block_diagonal_matrix":
            return global_utils.get_block_diagonal_matrix(
                self.num_agents, self.config["dependency_block_diagonal_size"]
            )
        elif self.config["dependency_matrix_type"] == "off_diagonals":
            return global_utils.get_off_diagonals_adj_matrix(
                self.num_agents, self.config["num_off_diagonals"]
            )
        else:
            raise ValueError(
                "No valid dependency_matrix_type selected! Choose: 'full', 'connectivity_matrix',"
                " 'fix_num_additional_edges', 'block_diagonal_matrix', 'off_diagonals'"
            )

    def _init_expected_reward_matrix(self) -> np.ndarray:
        reward_matrix = np.random.uniform(
            high=self.config["reward_upper_bound"],
            size=self._get_state_action_num_agents_size(),
        )
        reward_matrix = self._incorporate_dependency_matrix_into_expected_rewards(
            reward_matrix
        )
        return reward_matrix

    def _incorporate_dependency_matrix_into_expected_rewards(
        self, reward_matrix: np.ndarray
    ):
        if np.min(self.dependency_matrix) == 1:
            return reward_matrix
        else:
            all_joint_actions_minus_one_agent = [
                global_utils.cast_decimal_to_other_base_to_list(
                    integer,
                    base=self.size_action_space,
                    length_representation=self.num_agents - 1,
                )
                for integer in range(self.size_action_space ** (self.num_agents - 1))
            ]
            for i in range(self.dependency_matrix.shape[0]):
                for j in range(self.dependency_matrix.shape[1]):
                    if self.dependency_matrix[i, j] == 0:
                        for action_minus_i in all_joint_actions_minus_one_agent:
                            for state in range(self.state_size):
                                row_indices = []
                                for k in range(self.size_action_space):
                                    action_copy = action_minus_i.copy()
                                    action_copy.insert(i, k)
                                    row_indices.append(
                                        self.get_state_action_row_index(
                                            state, action_copy
                                        )
                                    )
                                reward_matrix[row_indices, j] = reward_matrix[
                                    row_indices[0], j
                                ]
            return reward_matrix

    def _init_state_emb_for_policy(self) -> np.ndarray:
        emb = np.random.uniform(
            size=(
                self.state_size,
                self.num_agents,
                self.size_state_emb_for_policy,
                self.size_action_space,
            )
        )
        if self.config["add_bias_to_emb"]:
            emb[:, :, 0, :] = 1.0
        return emb

    def _init_state_actions_emb(self) -> np.ndarray:
        state_actions_emb = np.random.uniform(size=self._get_state_action_emb_size())
        if self.config["add_bias_to_emb"]:
            state_actions_emb[:, 0] = 1.0  # set one feature constant as 'bias' term
        return state_actions_emb

    def _get_state_index(self, state: int) -> int:
        return state * (self.size_action_space ** self.num_agents)

    def get_state_action_row_index(self, state: int, actions: np.array) -> int:
        # block actions of agents and use state number as block index, block index by action representation
        action_index = global_utils.cast_to_decimal_from_array(
            actions, self.size_action_space
        )
        state_index = self._get_state_index(state)
        return state_index + action_index

    def _update_state(self, new_state: int):
        self.current_state = new_state

    def _state_transition(self, row_index: int) -> int:
        transition_prob = self.transition_matrix[row_index, :]
        return np.random.choice(len(transition_prob), p=transition_prob)

    def _sample_immediate_rewards(self, row_index: int) -> np.ndarray:
        expected_agent_rewards = self.expected_reward_matrix[row_index, :]
        reward_variance = np.array([0.0] * len(expected_agent_rewards))
        lower_bounds = expected_agent_rewards - reward_variance
        upper_bounds = expected_agent_rewards + reward_variance

        return np.random.uniform(low=lower_bounds, high=upper_bounds)

    def get_current_state(self) -> int:
        return self.current_state

    def get_state_emb_for_policy(self, state: int) -> np.array:
        # return array.shape = (num_agents, size_state_emb, size_action_space)
        return self.state_emb_for_policy[state, :, :, :]

    def get_state_action_emb(self, state: int, actions: List) -> np.ndarray:
        state_action_index = self.get_state_action_row_index(state, actions)
        return self.state_actions_emb[state_action_index, :]

    def get_all_joint_action_combinations(self):
        return [
            global_utils.cast_decimal_to_other_base_to_list(
                integer,
                base=self.size_action_space,
                length_representation=self.num_agents,
            )
            for integer in range(self.get_joint_action_size())
        ]

    def get_state_action_emb_from_list(
        self, state: int, actions_list: List[List[int]]
    ) -> np.ndarray:
        state_action_indices = [
            self.get_state_action_row_index(state, actions) for actions in actions_list
        ]
        return self.state_actions_emb[state_action_indices, :]

    def get_all_state_action_emb_for_state(self, state: int) -> np.ndarray:
        state_index, state_index_plus_one = self.get_consecutive_state_indices(state)
        return self.state_actions_emb[state_index:state_index_plus_one, :]

    def get_consecutive_state_indices(self, state: int) -> Tuple[int, int]:
        return self._get_state_index(state), self._get_state_index(state + 1)

    def step(self, actions: List) -> np.ndarray:
        state_action_row_index = self.get_state_action_row_index(
            self.current_state, actions
        )

        new_state = self._state_transition(state_action_row_index)
        self._update_state(new_state)

        immediate_rewards = self._sample_immediate_rewards(state_action_row_index)
        return immediate_rewards
