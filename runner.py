import multiprocessing
from pathlib import Path

import numpy as np

import global_utils
from environment.env_base import MathEnv
from logger.logger_module import Logger
from trainer.base_trainer import BaseTrainer
from trainer.trainer_env_initializer import TrainerEnvInitializer


class Runner(object):
    def __init__(self, config: dict):
        self.config = config

        self._set_random_seed()
        self.env = self._create_env()
        self.logger = self._init_logger()
        self.trainer = self._init_trainer()

    def _set_random_seed(self):
        global_utils.init_random_seed(self.config["seed"])

    def _create_env(self) -> MathEnv:
        return TrainerEnvInitializer.get_env(self.config)

    def _init_logger(self) -> Logger:
        return Logger(self.config)

    def _init_trainer(self) -> BaseTrainer:
        return TrainerEnvInitializer.get_trainer(
            self.config["trainer"], self.env, self.config, self.logger
        )

    def run(self):
        # Start the training process
        self.trainer.train()


def main():
    project_root_dir = str(Path().resolve().parents[0]) + "/detecting-influence-structures-in-marl"

    runner_config = {
        "seed": 0,
        "num_iterations": 15001,
        "env_config": "medium.json",
        "project_root_dir": project_root_dir,
        # trainer = 'zhang_decentralized', 'zhang_centralized', 'zhang_analytical_critic', 'sarsa_tabular',
        # 'sarsa_linear', 'random'
        "disable_actor_step": False,
        "add_noise_to_actor_params": False,
        "calculate_tim_analytically_in_every_step": False,
        "noise_strength_decay": 0.9995,
        "pretrain_ind_critics": False,
        "pretrain_im_estimation": False,
        "pretrain_num_steps": 10000,
        "trainer": "zhang_decentralized",
        "ind_critic_type": "zhang_decentralized",
        "log_rolling_window_size": 100,
        "log_neptune": False,
        "log_tim_agent_0": True,
        "neptune_log_interval": 1,
        "neptune_image_log_interval": 1000,
        "neptune_ind_impact_metrics": False,
        "neptune_impact_error_metrics": True,
        "neptune_project_name": "",
        "neptune_experiment_name": "metropolis-seed-test",
        "log_local": True,
        "log_local_folder_name": "metropolis-seed-test",
    }

    exp_list = []

    # ########################### Scenario I ########################### #
    # TIM Estimation with fixed actor and critic estimation
    for i in range(2):
        for num_add_edges in [0, 2, 6, 10]:
            experiment_config = {
                "seed": i,
                "num_iterations": 50001,
                "log_neptune": False,
                "disable_actor_step": True,
                "add_noise_to_actor_params": False,
                "noise_strength_decay": 0.9985,
                "trainer": "zhang_decentralized",
                "calculate_tim_analytically_in_every_step": False,
                "ind_critic_type": "sarsa_tabular",
                "pretrain_ind_critics": False,
                "pretrain_im_estimation": False,
                "neptune_log_interval": 1,
                "env_config": "small.json",
                "connectivity_matrix_sampling_type": "full_consensus",
                "communication_matrix_sampling_type": "full",
                "comm_num_additional_edges": 5,
                "env_num_dependency_edges": num_add_edges,
                "num_off_diagonals": 1,
                "pretrain_num_steps": 10001,
                "actor_step_size_decay": 0.088,
                "critic_step_size_decay": 0.039,
                "size_state_action_emb": 80,
                "size_state_emb_for_policy": 20,
                "actor_initial_step_size": 0.924,
                "critic_initial_step_size": 0.128,
                "im_step_size_decay": 0.539,
                "im_initial_step_size": 0.740,
                "tim_step_size_decay": 0.726,
                "tim_initial_step_size": 0.471,
                "q_learning_rate": 0.036,
                "ltr_learning_rate": 0.036,
                "dependency_matrix_type": "fix_num_additional_edges",
                "dependency_block_diagonal_size": 2,
                "reward_upper_bound": 2,
                "neptune_project_name": "",
                "neptune_experiment_name": "hyperparameter-im-estimation",
                "log_local": True,
                "log_local_folder_name": "static_policy/impact-estimation-num-edges-"
                + str(num_add_edges),
            }

            exp_list.append((experiment_config, project_root_dir, runner_config))

    # ########################### Scenario II ########################### #
    # TIM and SIM for learning policy
    """for i in range(50):
        experiment_config = {
            "seed": i,
            "num_iterations": 10001,
            "log_neptune": False,
            "disable_actor_step": False,
            "add_noise_to_actor_params": False,
            "noise_strength_decay": 0.9985,
            "trainer": "zhang_decentralized",
            "calculate_tim_analytically_in_every_step": True,
            "ind_critic_type": "sarsa_tabular",
            "pretrain_ind_critics": False,
            "pretrain_im_estimation": False,
            "neptune_log_interval": 1,
            "env_config": "small.json",
            "connectivity_matrix_sampling_type": "full_consensus",
            "communication_matrix_sampling_type": "full",
            "comm_num_additional_edges": 5,
            "env_num_dependency_edges": 10,
            "num_off_diagonals": 1,
            "pretrain_num_steps": 1,
            "actor_step_size_decay": 0.088,
            "critic_step_size_decay": 0.039,
            "size_state_action_emb": 80,
            "size_state_emb_for_policy": 20,
            "actor_initial_step_size": 0.924,
            "critic_initial_step_size": 0.128,
            "im_step_size_decay": 0.539,
            "im_initial_step_size": 0.740,
            "tim_step_size_decay": 0.726,
            "tim_initial_step_size": 0.471,
            "q_learning_rate": 0.036,
            "ltr_learning_rate": 0.036,
            "dependency_matrix_type": "fix_num_additional_edges",
            "dependency_block_diagonal_size": 3,
            "reward_upper_bound": 2,
            "neptune_project_name": "",
            "neptune_experiment_name": "hyperparameter-im-estimation",
            "log_local": True,
            "log_local_folder_name": "learning_policy/impact-estimation",
        }

        exp_list.append((experiment_config, project_root_dir, runner_config))"""

    num_cores = 4  # multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(global_utils.start_single_experiment, exp_list, 1)

    print("Done")


if __name__ == "__main__":
    main()
