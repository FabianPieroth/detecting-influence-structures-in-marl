from typing import Dict, List

import numpy as np

from environment.env_base import MathEnv
from impact_measure.impact_sampler import ImpactSampler
from logger.logger_module import Logger


class ImpactEstimator(object):
    def __init__(
        self, env: MathEnv, config: Dict, impact_sampler: ImpactSampler, logger: Logger
    ):
        self.env = env
        self.config = config
        self.impact_sampler = impact_sampler
        self.logger = logger
        self.im_estimates = self._init_im_estimates()
        self.tim_estimates = self._init_tim_estimates()

        self.im_step_size = 1.0
        self.tim_step_size = 1.0

        self.im_step_size_decay = self.config["im_step_size_decay"]
        self.im_initial_step_size = self.config["im_initial_step_size"]
        self.tim_step_size_decay = self.config["tim_step_size_decay"]
        self.tim_initial_step_size = self.config["tim_initial_step_size"]

    def _init_im_estimates(self) -> np.ndarray:
        return (
            np.ones((self.env.state_size, self.env.num_agents, self.env.num_agents))
            / self.env.num_agents
        )

    def _init_tim_estimates(self) -> np.ndarray:
        return np.ones((self.env.num_agents, self.env.num_agents)) / self.env.num_agents

    def update_impact_estimations(self, state: int, action: List[int], time_step: int):
        impact_sample = self.impact_sampler.get_impact_samples(state, action)
        self._update_step_sizes(time_step)
        self._update_im_estimates(state, impact_sample)
        self._update_tim_estimates(impact_sample)

    def update_impact_sampler(self, **kwargs):
        self.impact_sampler.update_ind_critics(**kwargs)

    def _update_step_sizes(self, time_step: int):
        self.im_step_size = (
            self.im_initial_step_size / time_step ** self.im_step_size_decay
        )
        self.tim_step_size = (
            self.tim_initial_step_size / time_step ** self.tim_step_size_decay
        )

    def _update_im_estimates(self, state: int, impact_sample: np.ndarray):
        self.im_estimates[state] = self._stochastic_iteration_update(
            self.im_step_size, self.im_estimates[state], impact_sample
        )

    def _update_tim_estimates(self, impact_sample: np.ndarray):
        self.tim_estimates = self._stochastic_iteration_update(
            self.tim_step_size, self.tim_estimates, impact_sample
        )

    @staticmethod
    def _stochastic_iteration_update(
        step_size: float, estimate: np.ndarray, sample: np.ndarray
    ) -> np.ndarray:
        return (1.0 - step_size) * estimate + step_size * sample

    def get_im_estimates(self) -> np.ndarray:
        return self.im_estimates

    def get_tim_estimates(self) -> np.ndarray:
        return self.tim_estimates
