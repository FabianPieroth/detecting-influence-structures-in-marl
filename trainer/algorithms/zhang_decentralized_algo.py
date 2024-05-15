from collections import deque
from typing import Dict

import numpy as np

import global_utils
from environment.env_base import MathEnv
from logger.logger_module import Logger
from models.actors_and_critics.zhang_decentralized import (
    ZhangDecentralizedActor,
    ZhangDecentralizedCritic,
)
from networks.communication_network import NetworkSampler
from trainer.base_trainer import BaseTrainer


class ZhangDecentralizedAlgorithm(BaseTrainer):
    def __init__(self, env: MathEnv, config: dict, logger: Logger):
        super().__init__(env, config, logger)

        self.actor = self._init_actor()
        self.critic = self._init_critic()
        self.networksampler = self._init_networksampler()
        self.init_impact_measure_objects()

    @classmethod
    def get_trainer_config(cls, project_root_dir: str) -> Dict:
        analytical_config = cls._get_decentralized_config(project_root_dir)
        network_sampler_config = cls._get_networksampler_config(project_root_dir)
        combined_dicts = global_utils.integrate_and_overwrite_dicts(
            analytical_config, network_sampler_config
        )
        return combined_dicts

    @staticmethod
    def _get_decentralized_config(project_root_dir) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/models/configs/zhang_decentralized_config.json"
        )

    @staticmethod
    def _get_networksampler_config(project_root_dir) -> Dict:
        return global_utils.load_json(
            project_root_dir + "/networks/configs/networksampler_config.json"
        )

    def _init_actor(self) -> ZhangDecentralizedActor:
        return ZhangDecentralizedActor(self.env, self.config)

    def _init_critic(self) -> ZhangDecentralizedCritic:
        return ZhangDecentralizedCritic(self.env, self.config)

    def _init_networksampler(self) -> NetworkSampler:
        return NetworkSampler(self.env, self.config)

    def train(self):
        # create lists to estimate long-term-average return
        average_rewards_log = []
        average_reward_rolling_windows = deque(
            maxlen=self.config["log_rolling_window_size"]
        )
        self.logger.init_policy_params_to_log(self.actor.policy)

        # time-step 0
        actions_t, state_t, rewards_t_plus_1 = self.env_act_and_step()

        im_analytical, tim_analytical = (
            self.calculate_and_log_impact_measure_time_step_0()
        )

        if self.config["pretrain_ind_critics"]:
            self.pretrain_impact_estimation(
                actions_t, im_analytical, rewards_t_plus_1, state_t, tim_analytical
            )

        for t in range(1, self.config["num_iterations"]):
            # update long-term-return (ltr) estimates and return past ltr estimates for later
            ltr_estimates_t = self.critic.update_ltr_estimates(rewards_t_plus_1)
            actions_t_plus_1, state_t_plus_1, rewards_t_plus_2 = self.env_act_and_step()
            # ######################### Impact Measure ######################### #
            if self.config["pretrain_im_estimation"]:
                num_pretrain_steps = self.config["pretrain_num_steps"]
            else:
                num_pretrain_steps = 0
            self.impact_estimator.update_impact_estimations(
                state_t, actions_t, t + num_pretrain_steps
            )
            tim_estimate = self.impact_estimator.get_tim_estimates()
            im_estimate = self.impact_estimator.get_im_estimates()
            self.impact_estimator.update_impact_sampler(
                **{
                    "rewards_t_plus_1": rewards_t_plus_1,
                    "state_t": state_t,
                    "actions_t": actions_t,
                    "state_t_plus_1": state_t_plus_1,
                    "actions_t_plus_1": actions_t_plus_1,
                }
            )
            connectivity_sampler_params = {
                "state": state_t,
                "im_estimate": im_estimate,
                "tim_estimate": tim_estimate,
                "im_analytical": im_analytical,
                "tim_analytical": tim_analytical,
            }
            # ######################## Parameter Updates ######################## #
            self.critic.critic_step(
                rewards_t_plus_1,
                state_t,
                actions_t,
                state_t_plus_1,
                actions_t_plus_1,
                ltr_estimates_t,
            )
            if not self.config["disable_actor_step"]:
                self.actor.actor_step(state_t, actions_t, self.critic)

            if self.config["add_noise_to_actor_params"]:
                global_utils.add_noise_to_actor_params(
                    self.actor, self.config["noise_strength_decay"], t
                )
            connectivity_matrix = self.networksampler.draw_connectivity_matrix(
                **connectivity_sampler_params
            )
            self.critic.consensus_step(connectivity_matrix)

            # ############################## Logging ############################# #

            average_rewards_log.append(rewards_t_plus_1.mean())
            average_reward_rolling_windows.append(rewards_t_plus_1.mean())
            if t % 500 == 0:
                print(
                    "t = "
                    + str(t)
                    + ", avg. reward: "
                    + str(np.mean(average_rewards_log))
                    + ", min. connectivity_matrix: "
                    + str(np.min(connectivity_matrix[connectivity_matrix > 0.0]))
                )
            self.logger.log_ltr(
                average_reward_rolling_windows, average_rewards_log, connectivity_matrix
            )
            self.logger.log_norm_of_policy(self.actor.policy)
            self.logger.log_impact_metrics(
                "analytical",
                self.config["ind_critic_type"],
                im_analytical,
                im_estimate,
                tim_analytical,
                tim_estimate,
            )
            # ##################### Update iteration counter t <- t+1 ##################### #
            if self.config["calculate_tim_analytically_in_every_step"]:
                im_analytical, tim_analytical = (
                    self.analytical_impact_determiner.calc_analytical_impact_estimates()
                )
            self.logger.set_current_time_step(t)
            state_t = state_t_plus_1
            actions_t = actions_t_plus_1
            rewards_t_plus_1 = rewards_t_plus_2

    def calculate_and_log_impact_measure_time_step_0(self):
        im_analytical, tim_analytical = (
            self.analytical_impact_determiner.calc_analytical_impact_estimates()
        )
        """im_analytical, tim_analytical = (
                np.zeros((self.env.num_agents, self.env.num_agents)),
                np.zeros((self.env.num_agents, self.env.num_agents)),
            )"""
        tim_estimate = self.impact_estimator.get_tim_estimates()
        im_estimate = self.impact_estimator.get_im_estimates()

        comparison_name = "analytical"
        estimate_name = self.config["ind_critic_type"]

        if self.config["pretrain_im_estimation"]:
            comparison_name = "pretrain_" + comparison_name
            estimate_name = "pretrain_" + estimate_name

        self.logger.log_impact_metrics(
            comparison_name,
            estimate_name,
            im_analytical,
            im_estimate,
            tim_analytical,
            tim_estimate,
        )
        return im_analytical, tim_analytical

    def pretrain_impact_estimation(
        self, actions_t, im_analytical, rewards_t_plus_1, state_t, tim_analytical
    ):
        print(
            "Start the pretraining phase for the ind critics with im-pretraining: "
            + str(self.config["pretrain_im_estimation"])
        )
        for t_pre in range(1, self.config["pretrain_num_steps"]):
            actions_t_plus_1, state_t_plus_1, rewards_t_plus_2 = self.env_act_and_step()
            # ######################### Impact Measure ######################### #
            if self.config["pretrain_im_estimation"]:
                self.impact_estimator.update_impact_estimations(
                    state_t, actions_t, t_pre
                )
            tim_estimate = self.impact_estimator.get_tim_estimates()
            im_estimate = self.impact_estimator.get_im_estimates()
            self.impact_estimator.update_impact_sampler(
                **{
                    "rewards_t_plus_1": rewards_t_plus_1,
                    "state_t": state_t,
                    "actions_t": actions_t,
                    "state_t_plus_1": state_t_plus_1,
                    "actions_t_plus_1": actions_t_plus_1,
                }
            )
            if self.config["pretrain_im_estimation"]:
                self.logger.log_impact_metrics(
                    "pretrain_analytical",
                    "pretrain_" + self.config["ind_critic_type"],
                    im_analytical,
                    im_estimate,
                    tim_analytical,
                    tim_estimate,
                )
            # ############################## Print Info ############################# #
            print(
                "Pretraining phase step : "
                + str(t_pre)
                + "/"
                + str(self.config["pretrain_num_steps"])
            )
            self.logger.set_current_time_step(t_pre)
            state_t = state_t_plus_1
            actions_t = actions_t_plus_1
            rewards_t_plus_1 = rewards_t_plus_2

    def env_act_and_step(self):
        state_t = self.env.get_current_state()
        actions_t = self.actor.act(state_t)
        # step from t+1 -> t+2
        rewards_t_plus_1 = self.env.step(actions=actions_t)
        return actions_t, state_t, rewards_t_plus_1
