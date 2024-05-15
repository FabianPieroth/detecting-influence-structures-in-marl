import numpy as np

import global_utils
from environment.env_base import MathEnv


class MarkovChainDeterminer(object):
    def __init__(self, env: MathEnv):
        self.env = env
        self.transition_matrix = self.env.get_transition_matrix()
        self.p_theta = np.zeros(
            (self.env.get_state_action_size(), self.env.get_state_action_size())
        )
        self.stationary_dist = np.zeros(self.env.get_state_action_size())

    def update_markov_chain_params(self, policy_distribution: np.ndarray):
        self._update_p_theta(policy_distribution)
        self._update_stationary_distribution()

    def _update_p_theta(self, policy_distribution: np.ndarray):
        for i in range(self.env.state_size):
            state_index, state_index_plus_1 = self.env.get_consecutive_state_indices(i)
            p_theta_for_state_i = (
                self.transition_matrix[:, i][:, np.newaxis]
                * policy_distribution[:, i][np.newaxis, :]
            )
            self.p_theta[:, state_index:state_index_plus_1] = p_theta_for_state_i

    def _update_stationary_distribution(self):
        self.stationary_dist = global_utils.get_stationary_distribution_with_sanity_checks(
            self.p_theta.T
        )

    def get_p_theta(self) -> np.ndarray:
        return self.p_theta

    def get_stationary_distribution(self) -> np.ndarray:
        return self.stationary_dist
