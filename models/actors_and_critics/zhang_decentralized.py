from typing import Dict, List

import numpy as np

import global_utils
from environment.env_base import MathEnv
from models.actors_and_critics.zhang_centralized import ZhangCentralizedCritic
from models.base_actor import BaseActor
from models.base_critic import BaseCritic


class ZhangDecentralizedCritic(BaseCritic):
    def __init__(self, env: MathEnv, config: dict):
        super().__init__(env, config)
        self.num_agents = env.num_agents
        self.size_action_space = env.size_action_space
        self.size_state_action_emb = env.size_state_action_emb

        self.ltr_estimates = self._init_ltr_estimates()
        self.ind_critic_parameters = self._init_ind_critic_parameters()
        self.global_critic_parameters = self._init_global_critic_parameters()
        self.critic_initial_step_size = self.config["critic_initial_step_size"]
        self.critic_step_size = 1.0
        self.critic_step_size_decay = self.config["critic_step_size_decay"]
        self.critic_step_counter = 0

    def _init_ltr_estimates(self):
        return 2.0 * np.ones(
            self.num_agents
        )  # np.random.uniform(low=0.0, high=1.0, size=self.num_agents)

    def _init_ind_critic_parameters(self):
        return global_utils.init_learning_parameters(
            (self.num_agents, self.size_state_action_emb)
        )

    def _init_global_critic_parameters(self):
        return global_utils.init_learning_parameters(
            (self.num_agents, self.size_state_action_emb)
        )

    def update_ltr_estimates(self, rewards: np.array) -> np.array:
        ltr_estimates_t = self.ltr_estimates
        self.ltr_estimates = (
            1.0 - self.critic_step_size
        ) * self.ltr_estimates + self.critic_step_size * rewards
        return ltr_estimates_t

    def critic_step(
        self,
        rewards_t_plus_1: np.array,
        state_t: int,
        actions_t: List,
        state_t_plus_1: int,
        actions_t_plus_1: List,
        ltr_estimates: np.array,
    ):
        td_error = self._calc_temporal_differencing_error(
            rewards_t_plus_1,
            state_t,
            actions_t,
            state_t_plus_1,
            actions_t_plus_1,
            ltr_estimates,
        )
        gradient_q_function_t = self._calc_gradient_q_function(state_t, actions_t)
        self._update_ind_critic_parameters(td_error, gradient_q_function_t)
        self._update_step_size()

    def _calc_temporal_differencing_error(
        self,
        rewards_t_plus_1: np.array,
        state_t: int,
        actions_t: List,
        state_t_plus_1: int,
        actions_t_plus_1: List,
        ltr_parameters: np.array,
    ) -> np.array:
        q_values_t = self._calc_q_values(state_t, actions_t)
        q_values_t_plus_1 = self._calc_q_values(state_t_plus_1, actions_t_plus_1)
        return rewards_t_plus_1 - ltr_parameters + q_values_t_plus_1 - q_values_t

    def _calc_q_values(self, state: int, actions: List) -> np.array:
        state_action_emb = self.env.get_state_action_emb(state, actions)
        return global_utils.vector_product_state_action_function(
            parameters=self.global_critic_parameters, embedding=state_action_emb
        )

    def _calc_gradient_q_function(self, state: int, actions: List):
        return self.env.get_state_action_emb(state, actions)

    def _update_ind_critic_parameters(
        self, td_error: np.array, gradient_q_function_t: np.array
    ):
        self.ind_critic_parameters = (
            self.global_critic_parameters
            + self.critic_step_size
            * td_error[:, np.newaxis]
            * gradient_q_function_t[np.newaxis, :]
        )

    def _update_step_size(self):
        self.critic_step_counter += 1
        self.critic_step_size = (
            self.critic_initial_step_size
            / self.critic_step_counter ** self.critic_step_size_decay
        )

    def consensus_step(self, connectivity_matrix: np.array):
        self.global_critic_parameters = (
            connectivity_matrix[:, :, np.newaxis]
            * self.ind_critic_parameters[np.newaxis, :, :]
        ).sum(axis=1)


class IndZhangDecentralizedCritic(ZhangCentralizedCritic):
    def __init__(self, env: MathEnv, config: Dict, agent_id: int):
        super().__init__(env, config)
        # ######### ATTENTION! ########### #
        # we use the centralized critic, as
        # this is equivalent to ind. estimations!
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


class ZhangDecentralizedActor(BaseActor):
    def __init__(self, env: MathEnv, config: dict):
        super().__init__(env, config)
        self.num_agents = env.num_agents
        self.size_action_space = env.size_action_space
        self.size_state_emb_for_policy = env.size_state_emb_for_policy

        self.policy = self._init_policy_parameters()
        self.actor_step_counter = 0
        self.actor_step_size = 1.0
        self.actor_initial_step_size = self.config["actor_initial_step_size"]
        self.actor_step_size_decay = self.config["actor_step_size_decay"]

    def _init_policy_parameters(self):
        return global_utils.init_learning_parameters(
            (self.num_agents, self.size_state_emb_for_policy)
        )

    def act(self, state: int) -> List:
        state_emb = self.env.get_state_emb_for_policy(state)
        policy_distributions = self._get_policy_distributions(state_emb)
        actions = self._sample_actions_from_policy(policy_distributions)
        return actions

    def get_state_policy_distribution(self, state: int) -> np.ndarray:
        """
        :param state:
        :return: Policy Distribution over all joint-actions in the given state
        """
        policy_distribution = np.zeros(self.env.get_joint_action_size())
        state_emb = self.env.get_state_emb_for_policy(state)
        policy_distributions = self._get_policy_distributions(state_emb)
        for i, actions in enumerate(self.env.get_all_joint_action_combinations()):
            policy_distribution[i] = np.prod(
                policy_distributions[np.arange(len(policy_distributions)), actions]
            )
        assert np.isclose(
            np.sum(policy_distribution), 1.0
        ), "Check that we are returning a policy distribution!"
        return policy_distribution

    def _get_policy_distributions(self, state_emb: np.array) -> np.array:
        logits = (state_emb * self.policy[:, :, np.newaxis]).sum(axis=1)
        return global_utils.softmax(logits)

    def _sample_actions_from_policy(self, policy_distributions: np.array) -> List:
        if np.isnan(policy_distributions.sum()):
            print("test")
        actions = [0] * self.num_agents
        for i in range(self.num_agents):
            actions[i] = np.random.choice(
                self.size_action_space, p=policy_distributions[i, :]
            )
        return actions

    def actor_step(
        self, state_t: int, actions_t: np.array, critic: ZhangDecentralizedCritic
    ):
        advantage_function_sample = self._calculate_advantage_function(
            state_t, actions_t, critic
        )
        score_function_sample = self._calculate_score_function(state_t, actions_t)
        self.policy += (
            self.actor_step_size
            * advantage_function_sample[:, np.newaxis]
            * score_function_sample
        )
        self._update_step_size()

    def _calculate_advantage_function(
        self, state_t: int, actions_t: List, critic: ZhangDecentralizedCritic
    ) -> np.array:
        # TODO: Super ugly, write this without the loops
        marginalized_q_values = np.zeros(self.num_agents)
        state_emb_t = self.env.get_state_emb_for_policy(state_t)
        policy_distribution_t = self._get_policy_distributions(state_emb_t)
        q_values_t = critic._calc_q_values(state_t, actions_t)
        for i in range(self.num_agents):
            for j in range(self.size_action_space):
                actions_copy = actions_t.copy()
                actions_copy[i] = j
                q_value_j = critic._calc_q_values(state_t, actions_copy)
                marginalized_q_values[i] += policy_distribution_t[i, j] * q_value_j[i]
        return q_values_t - marginalized_q_values

    def _calculate_score_function(self, state_t: int, actions_t: List) -> np.array:
        state_emb_for_policy_t = self.env.get_state_emb_for_policy(state_t)
        state_emb_for_policy_by_action_t = global_utils.select_state_emb_for_policy_by_specific_action(
            state_emb_for_policy_t, actions_t
        )
        policy_distribution_t = self._get_policy_distributions(state_emb_for_policy_t)
        policy_weighted_state_emb_for_policy = (
            state_emb_for_policy_t * policy_distribution_t[:, np.newaxis, :]
        ).sum(axis=2)
        return state_emb_for_policy_by_action_t - policy_weighted_state_emb_for_policy

    def _update_step_size(self):
        self.actor_step_counter += 1
        self.actor_step_size = (
            self.actor_initial_step_size
            / self.actor_step_counter ** self.actor_step_size_decay
        )
