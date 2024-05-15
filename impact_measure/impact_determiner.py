from typing import Tuple

import numpy as np

from environment.env_base import MathEnv
from environment.env_markov_chain_determiner import MarkovChainDeterminer
from impact_measure.impact_sampler import ImpactSampler
from logger.logger_module import Logger
from models.base_actor import BaseActor


class ImpactDeterminer(object):
    def __init__(
        self,
        env: MathEnv,
        actor: BaseActor,
        impact_sampler: ImpactSampler,
        mc_determiner: MarkovChainDeterminer,
        logger: Logger,
    ):
        self.env = env
        self.actor = actor
        self.impact_sampler = impact_sampler
        self.mc_determiner = mc_determiner
        self.logger = logger

    def calculate_impact_measures(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Calculate impact measures with impact determiner!")
        im = self.get_impact_measure_for_all_states()
        tim = self.get_total_impact_measure()
        return im, tim

    def get_impact_measure(self, state: int) -> np.ndarray:
        """
        :return: im for given state (shape=(num_agents, num_agents))
        """
        state_policy_distribution = self.actor.get_state_policy_distribution(state)
        state_impact_samples = self.impact_sampler.get_state_impact_samples(state)

        return np.sum(
            state_policy_distribution[:, np.newaxis, np.newaxis] * state_impact_samples,
            axis=0,
        )

    def get_impact_measure_for_all_states(self) -> np.ndarray:
        """
        :return: im for all states (shape=(state_size, num_agents, num_agents))
        """
        all_impact_measures = np.zeros(
            (self.env.state_size, self.env.num_agents, self.env.num_agents)
        )
        for state in range(self.env.state_size):
            all_impact_measures[state, :, :] = self.get_impact_measure(state)

        return all_impact_measures

    def get_total_impact_measure(self) -> np.ndarray:
        """
        :return: tim (shape=(num_agents, num_agents))
        """
        policy_distribution = self.actor.get_policy_distribution()
        self.mc_determiner.update_markov_chain_params(policy_distribution)
        stationary_distribution = self.mc_determiner.get_stationary_distribution()
        all_impact_samples = self.impact_sampler.get_all_impact_samples()
        return np.sum(
            stationary_distribution[:, np.newaxis, np.newaxis] * all_impact_samples,
            axis=0,
        )


class AnalyticalImpactDeterminer(ImpactDeterminer):
    def __init__(
        self,
        env: MathEnv,
        actor: BaseActor,
        impact_sampler: ImpactSampler,
        mc_determiner: MarkovChainDeterminer,
        logger: Logger,
    ):
        super().__init__(env, actor, impact_sampler, mc_determiner, logger)

    def calc_analytical_impact_estimates(self):
        self.mc_determiner.update_markov_chain_params(
            self.actor.get_policy_distribution()
        )
        self.impact_sampler.update_ind_critics()

        im_analytical, tim_analytical = self.calculate_impact_measures()
        return im_analytical, tim_analytical
