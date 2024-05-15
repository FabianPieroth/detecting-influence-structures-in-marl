from typing import Dict, List

import numpy as np

import global_utils
from environment.env_base import MathEnv
from models.base_actor import BaseActor
from models.base_critic import BaseCritic


class ZhangCentralizedCritic(BaseCritic):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

        self.ltr_estimate = self._init_ltr_estimate()
        self.critic_weights = self._init_critic_weights()
        self.critic_initial_step_size = self.config["critic_initial_step_size"]
        self.critic_step_size = 1.0
        self.critic_step_size_decay = self.config["critic_step_size_decay"]
        self.critic_step_counter = 0

    @staticmethod
    def _init_ltr_estimate():
        return 0.0

    def _init_critic_weights(self):
        return global_utils.init_learning_parameters(self.env.size_state_action_emb)

    def update_ltr_estimate(self, rewards: np.ndarray) -> np.ndarray:
        ltr_estimate_t = self.ltr_estimate
        self.ltr_estimate = (
            1.0 - self.critic_step_size
        ) * self.ltr_estimate + self.critic_step_size * np.mean(rewards)
        return ltr_estimate_t

    def critic_step(
        self,
        rewards_t_plus_1: np.ndarray,
        state_t: int,
        actions_t: List,
        state_t_plus_1: int,
        actions_t_plus_1: List,
        ltr_estimate: np.ndarray,
    ) -> np.ndarray:
        critic_weights_t = self.critic_weights
        td_error = self._calc_temporal_differencing_error(
            rewards_t_plus_1,
            state_t,
            actions_t,
            state_t_plus_1,
            actions_t_plus_1,
            ltr_estimate,
        )
        gradient_q_function_t = self._calc_gradient_q_function(state_t, actions_t)
        self._update_critic_weights(td_error, gradient_q_function_t)
        self._update_step_size()
        return critic_weights_t

    def _calc_temporal_differencing_error(
        self,
        rewards_t_plus_1: np.ndarray,
        state_t: int,
        actions_t: List,
        state_t_plus_1: int,
        actions_t_plus_1: List,
        ltr_parameter: np.ndarray,
    ) -> np.ndarray:
        q_values_t = self.calc_q_value(state_t, actions_t, self.critic_weights)
        q_values_t_plus_1 = self.calc_q_value(
            state_t_plus_1, actions_t_plus_1, self.critic_weights
        )
        return (
            np.mean(rewards_t_plus_1) - ltr_parameter + q_values_t_plus_1 - q_values_t
        )

    def calc_q_value(
        self, state: int, actions: List, weights: np.ndarray
    ) -> np.ndarray:
        state_action_emb = self.env.get_state_action_emb(state, actions)
        return state_action_emb @ weights

    def _calc_gradient_q_function(self, state: int, actions: List):
        return self.env.get_state_action_emb(state, actions)

    def _update_critic_weights(
        self, td_error: np.ndarray, gradient_q_function_t: np.ndarray
    ):
        self.critic_weights += self.critic_step_size * td_error * gradient_q_function_t

    def _update_step_size(self):
        self.critic_step_counter += 1
        self.critic_step_size = (
            self.critic_initial_step_size
            / self.critic_step_counter ** self.critic_step_size_decay
        )

    @staticmethod
    def calculate_state_q_values(
        all_state_action_emb: np.array, weights: np.ndarray
    ) -> np.array:
        return (all_state_action_emb * weights[np.newaxis, :]).sum(axis=1)

    def get_all_q_values(self) -> np.ndarray:
        q_values = np.zeros((self.env.state_size, self.env.get_joint_action_size()))
        for i in range(self.env.state_size):
            all_state_action_emb = self.env.get_all_state_action_emb_for_state(i)
            q_values[i, :] = self.calculate_state_q_values(
                all_state_action_emb, self.critic_weights
            )
        return q_values

    def get_ind_q_values_for_state_action(
        self, state: int, action_list: List[List[int]]
    ) -> np.ndarray:
        state_action_embeddings = self.env.get_state_action_emb_from_list(
            state, action_list
        )
        return np.sum(
            self.critic_weights[np.newaxis, :] * state_action_embeddings, axis=1
        )


class IndZhangCentralizedCritic(ZhangCentralizedCritic):
    def __init__(self, env: MathEnv, config: Dict, agent_id: int):
        super().__init__(env, config)
        self.agent_id = agent_id

    def update_parameters_in_one_step(self, **kwargs):
        ltr_estimate_t = self.update_ltr_estimate(
            kwargs["rewards_t_plus_1"][self.agent_id]
        )
        # parameter updates
        _ = self.critic_step(
            kwargs["rewards_t_plus_1"][self.agent_id],
            kwargs["state_t"],
            kwargs["actions_t"],
            kwargs["state_t_plus_1"],
            kwargs["actions_t_plus_1"],
            ltr_estimate_t,
        )


class ZhangCentralizedActor(BaseActor):
    def __init__(self, env: MathEnv, config: Dict):
        super().__init__(env, config)

        self.policy = self._init_policy_parameters()
        self.actor_step_counter = 0
        self.actor_step_size = 1.0
        self.actor_initial_step_size = self.config["actor_initial_step_size"]
        self.actor_step_size_decay = self.config["actor_step_size_decay"]

    def _init_policy_parameters(self):
        return global_utils.init_learning_parameters(self.env.size_state_action_emb)

    def act(self, state: int) -> List:
        all_state_action_emb = self.env.get_all_state_action_emb_for_state(state)
        policy_distribution = self._get_policy_distribution(all_state_action_emb)
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

    def actor_step(
        self,
        state_t: int,
        actions_t: np.array,
        critic: ZhangCentralizedCritic,
        critic_weights_t: np.array,
    ):
        advantage_function_sample = self._calculate_advantage_function(
            state_t, actions_t, critic, critic_weights_t
        )
        score_function_sample = self._calculate_score_function(state_t, actions_t)
        self.policy += (
            self.actor_step_size * advantage_function_sample * score_function_sample
        )
        self._update_step_size()

    def _calculate_advantage_function(
        self,
        state_t: int,
        actions_t: List,
        critic: ZhangCentralizedCritic,
        critic_weights_t: np.array,
    ) -> np.array:
        convex_comb_q_values_and_policy = self._get_convex_comb_q_values_and_policy(
            state_t, critic, critic_weights_t
        )
        return (
            critic.calc_q_value(state_t, actions_t, critic_weights_t)
            - convex_comb_q_values_and_policy
        )

    def _get_convex_comb_q_values_and_policy(
        self, state_t: int, critic: ZhangCentralizedCritic, critic_weights_t: np.array
    ) -> float:
        all_state_action_emb = self.env.get_all_state_action_emb_for_state(state_t)
        policy_distribution_t = self._get_policy_distribution(all_state_action_emb)

        q_values_joint_action = critic.calculate_state_q_values(
            all_state_action_emb, critic_weights_t
        )
        return policy_distribution_t @ q_values_joint_action

    def _calculate_score_function(self, state_t: int, actions_t: List) -> np.array:
        state_action_emb = self.env.get_state_action_emb(state_t, actions_t)

        all_state_action_emb = self.env.get_all_state_action_emb_for_state(state_t)
        policy_distribution_t = self._get_policy_distribution(all_state_action_emb)
        convex_comb_emb_and_policy = (
            policy_distribution_t[:, np.newaxis] * all_state_action_emb
        ).sum(axis=0)

        return state_action_emb - convex_comb_emb_and_policy

    def _update_step_size(self):
        self.actor_step_counter += 1
        self.actor_step_size = (
            self.actor_initial_step_size
            / self.actor_step_counter ** self.actor_step_size_decay
        )
