from typing import Dict, List

import numpy as np

import global_utils
from environment.env_base import MathEnv
from environment.env_markov_chain_determiner import MarkovChainDeterminer
from models.actors_and_critics.analytical_critic import IndAnalyticalCritic
from models.actors_and_critics.random_actor_and_critic import IndRandomCritic
from models.actors_and_critics.sarsa_tabular_actor_critic import IndSarsaTabularCritic
from models.actors_and_critics.zhang_centralized import IndZhangCentralizedCritic
from models.actors_and_critics.zhang_decentralized import IndZhangDecentralizedCritic
from models.base_critic import BaseCritic


class ImpactSampler(object):
    def __init__(
        self,
        env: MathEnv,
        config: Dict,
        mc_determiner: MarkovChainDeterminer,
        critic_type: str,
    ):
        """
        Basic representation of a critic.
        """
        self.config = config
        self.env = env
        self.mc_determiner = mc_determiner
        self.critic_type = critic_type
        self.ind_critic_list = self._init_ind_critic_list()

    def _init_ind_critic_list(self) -> List[BaseCritic]:
        return [
            self._get_ind_critic_for_agent(agent)
            for agent in range(self.env.num_agents)
        ]

    def _get_ind_critic_for_agent(self, agent_id: int) -> BaseCritic:
        if self.critic_type == "random":
            return IndRandomCritic(self.env, self.config)
        elif self.critic_type == "zhang_analytical_critic":
            return IndAnalyticalCritic(
                self.env, self.config, self.mc_determiner, agent_id
            )
        elif self.critic_type == "zhang_centralized":
            return IndZhangCentralizedCritic(self.env, self.config, agent_id)
        elif self.critic_type == "zhang_decentralized":
            return IndZhangDecentralizedCritic(self.env, self.config, agent_id)
        elif self.critic_type == "sarsa_tabular":
            return IndSarsaTabularCritic(self.env, self.config, agent_id)
        else:
            raise ValueError(
                "No valid individual critic type for: "
                + self.critic_type
                + " implemented!"
            )

    def update_ind_critics(self, **kwargs):
        for ind_critic in self.ind_critic_list:
            ind_critic.update_parameters_in_one_step(**kwargs)

    def _get_impact_samples_for_each_ind_agent(
        self, state: int, counterfactual_actions: List[List[int]], agent: int
    ) -> np.ndarray:
        """
        :param state:
        :param counterfactual_actions: all possible single action deviations from given joint-action
        :return: impact_samples, shape=num_agents
        """
        counterfactual_q_values = self.ind_critic_list[
            agent
        ].get_ind_q_values_for_state_action(state, counterfactual_actions)
        reshaped_counterfactual_q_values = counterfactual_q_values.reshape(
            (self.env.num_agents, self.env.size_action_space)
        )
        return np.max(reshaped_counterfactual_q_values, axis=1) - np.min(
            reshaped_counterfactual_q_values, axis=1
        )

    def get_impact_samples(self, state: int, action: List[int]) -> np.ndarray:
        """
        :param state:
        :param action:
        :return: impact_samples, shape=(num_agents, num_agents), column-wise impact samples, which are later
                 row-normalised to get the coordination matrix
        """
        counterfactual_actions = self.get_possible_action_deviations(action)
        impact_samples = np.zeros((self.env.num_agents, self.env.num_agents))
        for agent in range(self.env.num_agents):
            impact_samples[:, agent] = self._get_impact_samples_for_each_ind_agent(
                state, counterfactual_actions, agent
            )
        return impact_samples

    def get_possible_action_deviations(self, action: List[int]) -> List[List[int]]:
        """
        :param action: a joint-action
        :return: list of joint-actions of all possible single agent counter factual actions
        """
        action_list = [
            action.copy()
            for i in range(self.env.num_agents * self.env.size_action_space)
        ]
        counter = 0
        for i in range(self.env.num_agents):
            for j in range(self.env.size_action_space):
                action_list[counter][i] = j
                counter += 1

        return action_list

    def get_state_impact_samples(self, state: int) -> np.ndarray:
        """
        :return: impact_samples for all all joint-actions in given state
                (shape=(joint_action_size, num_agents, num_agents))
        """
        state_impact_samples = np.zeros(
            (self.env.get_joint_action_size(), self.env.num_agents, self.env.num_agents)
        )
        joint_action_combinations = self.env.get_all_joint_action_combinations()
        for i, action in enumerate(joint_action_combinations):
            state_impact_samples[i, :, :] = self.get_impact_samples(state, action)
        return state_impact_samples

    def get_all_impact_samples(self) -> np.ndarray:
        """
        :return: impact_samples for all state-action pairs (shape=(state_size * joint_action_size, num_agents, num_agents))
        """
        all_impact_samples = np.zeros(
            (
                self.env.state_size * self.env.get_joint_action_size(),
                self.env.num_agents,
                self.env.num_agents,
            )
        )
        for state in range(self.env.state_size):
            state_index, state_index_plus_1 = self.env.get_consecutive_state_indices(
                state
            )
            all_impact_samples[
                state_index:state_index_plus_1, :, :
            ] = self.get_state_impact_samples(state)

        return all_impact_samples
