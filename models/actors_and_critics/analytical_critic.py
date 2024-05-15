from typing import Dict, List

import numpy as np

import global_utils
from environment.env_base import MathEnv
from environment.env_markov_chain_determiner import MarkovChainDeterminer
from models.base_actor import BaseActor
from models.base_critic import BaseCritic


class AnalyticalCritic(BaseCritic):
    def __init__(
        self, env: MathEnv, config: Dict, mc_determiner: MarkovChainDeterminer
    ):
        super().__init__(env, config)
        self.mc_determiner = mc_determiner
        self.transition_matrix = self.env.get_transition_matrix()
        self.expected_average_reward = self._get_expected_average_reward_vector(
            self.env.get_expected_reward_matrix()
        )
        self.ltr = 0.0
        self.analytical_epsilon_accuracy = self.config[
            "analytical_critic_epsilon_accuracy"
        ]

    @staticmethod
    def _get_expected_average_reward_vector(reward_matrix: np.array) -> np.array:
        return np.mean(reward_matrix, axis=1)

    def update_state_action_markov_chain_params(self, policy_distribution: np.array):
        self.mc_determiner.update_markov_chain_params(policy_distribution)
        self.update_ltr()

    def update_ltr(self):
        self.ltr = np.matmul(
            self.expected_average_reward,
            self.mc_determiner.get_stationary_distribution(),
        )

    def get_ind_q_values_for_state_action(
        self, state: int, actions_list: List[List[int]]
    ) -> np.ndarray:
        q_values = np.zeros(len(actions_list))
        expected_distributions = self._get_initial_distribution_vectors(
            state, actions_list
        )
        while (
            np.mean(
                np.sum(
                    np.abs(
                        expected_distributions
                        - self.mc_determiner.get_stationary_distribution()
                    ),
                    axis=1,
                )
            )
            > self.analytical_epsilon_accuracy
        ):
            q_values += self._get_expected_reward(expected_distributions) - self.ltr
            expected_distributions = self._get_next_expected_distributions(
                expected_distributions
            )
        return q_values

    def _get_initial_distribution_vectors(
        self, state: int, actions_list: List[List[int]]
    ) -> np.ndarray:
        state_action_distribution = np.zeros(
            (len(actions_list), self.env.get_state_action_size())
        )
        state_action_slice_indices = self._get_state_action_slice_indices(
            state, actions_list
        )
        np.put(state_action_distribution, state_action_slice_indices, 1.0)
        return state_action_distribution

    def _get_state_action_slice_indices(
        self, state: int, actions_list: List[List[int]]
    ):
        state_actions_row_indices = np.array(
            [
                self.env.get_state_action_row_index(state, actions)
                for actions in actions_list
            ]
        )
        base_state_indices = (
            np.array(range(len(actions_list))) * self.env.get_state_action_size()
        )
        state_action_slice_indices = state_actions_row_indices + base_state_indices
        return state_action_slice_indices

    def _get_expected_reward(self, expected_distributions: np.ndarray) -> np.ndarray:
        return np.matmul(expected_distributions, self.expected_average_reward)

    def _get_next_expected_distributions(
        self, expected_distributions: np.ndarray
    ) -> np.ndarray:
        next_state_distributions = np.matmul(
            expected_distributions, self.mc_determiner.get_p_theta()
        )
        assert all(np.isclose(next_state_distributions.sum(axis=1), 1.0))
        return next_state_distributions

    def get_single_q_value(self, state: int, actions: List[int]) -> float:
        q_value = self.get_ind_q_values_for_state_action(state, [actions])
        return float(q_value)

    def get_state_q_values(self, state: int) -> np.ndarray:
        all_state_actions_list = [
            global_utils.cast_decimal_to_other_base_to_list(
                integer,
                base=self.env.size_action_space,
                length_representation=self.env.num_agents,
            )
            for integer in range(self.env.get_joint_action_size())
        ]
        return self.get_ind_q_values_for_state_action(state, all_state_actions_list)

    def get_all_q_values(self) -> np.ndarray:
        q_values = np.zeros((self.env.state_size, self.env.get_joint_action_size()))
        for i in range(self.env.state_size):
            q_values[i, :] = self.get_state_q_values(i)
        return q_values


class IndAnalyticalCritic(AnalyticalCritic):
    def __init__(
        self,
        env: MathEnv,
        config: Dict,
        mc_determiner: MarkovChainDeterminer,
        agent_id: int,
    ):
        super().__init__(env, config, mc_determiner)
        self.agent_id = agent_id
        self.expected_average_reward = self._init_ind_expected_reward(
            self.env.get_expected_reward_matrix()
        )

    def _init_ind_expected_reward(self, reward_matrix: np.ndarray) -> np.ndarray:
        return reward_matrix[:, self.agent_id]

    def update_parameters_in_one_step(self, **kwargs):
        self.update_ltr()


class ZhangActorForAnalyticalCritic(BaseActor):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

        self.policy = self._init_policy_parameters()
        self.actor_step_counter = 0
        self.analytical_actor_step_size = self.config["analytical_actor_step_size"]
        self.analytical_actor_step_size_decay = self.config[
            "analytical_actor_step_size_decay"
        ]

    def _init_policy_parameters(self):
        return global_utils.init_learning_parameters(self.env.size_state_action_emb)

    def act(self, state: int) -> List:
        policy_distribution = self.get_state_policy_distribution(state)
        actions = self._sample_joint_action_from_policy(policy_distribution)
        return actions

    def get_state_policy_distribution(self, state: int) -> np.ndarray:
        all_state_action_emb = self.env.get_all_state_action_emb_for_state(state)
        return self._get_policy_distribution(all_state_action_emb)

    def _get_policy_distribution(self, state_action_embeddings: np.array) -> np.array:
        logits = (state_action_embeddings * self.policy[np.newaxis, :]).sum(axis=1)
        return global_utils.softmax(logits, single_axis=True)

    def _sample_joint_action_from_policy(self, policy_distribution: np.array) -> List:
        joint_action_index = np.random.choice(
            len(policy_distribution), p=policy_distribution
        )
        return global_utils.get_number_representation_from_int(
            index=joint_action_index,
            decimal_base=self.env.size_action_space,
            length_representation=self.env.num_agents,
        )

    def actor_step(self, state_t: int, actions_t: List, critic: AnalyticalCritic):
        advantage_function_sample = self._calculate_advantage_function(
            state_t, actions_t, critic
        )
        score_function_sample = self._calculate_score_function(state_t, actions_t)
        self.policy += (
            self.analytical_actor_step_size
            * advantage_function_sample
            * score_function_sample
        )
        self._update_step_size()

    def _calculate_advantage_function(
        self, state_t: int, actions_t: List, critic: AnalyticalCritic
    ) -> np.array:
        convex_comb_q_values_and_policy = self._get_convex_comb_q_values_and_policy(
            state_t, critic
        )
        return (
            critic.get_single_q_value(state_t, actions_t)
            - convex_comb_q_values_and_policy
        )

    def _get_convex_comb_q_values_and_policy(
        self, state_t: int, critic: AnalyticalCritic
    ) -> float:
        policy_distribution_t = self.get_state_policy_distribution(state_t)

        q_values_all_state_actions = critic.get_state_q_values(state_t)
        return policy_distribution_t @ q_values_all_state_actions

    def _calculate_score_function(self, state_t: int, actions_t: List) -> np.array:
        state_action_emb = self.env.get_state_action_emb(state_t, actions_t)

        all_state_action_emb = self.env.get_all_state_action_emb_for_state(state_t)
        policy_distribution_t = self.get_state_policy_distribution(state_t)
        convex_comb_emb_and_policy = (
            policy_distribution_t[:, np.newaxis] * all_state_action_emb
        ).sum(axis=0)

        return state_action_emb - convex_comb_emb_and_policy

    def _update_step_size(self):
        self.actor_step_counter += 1
        self.analytical_actor_step_size = (
            1.0 / self.actor_step_counter ** self.analytical_actor_step_size_decay
        )
