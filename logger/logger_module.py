import os
import sys
from typing import Deque, Dict, List, Tuple

import matplotlib.pyplot as plt
import neptune
import numpy as np

import global_utils


class Logger(object):
    def __init__(self, config: dict):
        self.config = config
        self.log_neptune = config["log_neptune"]
        self.log_local = config["log_local"]
        self.local_log_dict = {}
        self.neptune_project_name = self.config["neptune_project_name"]

        self.current_time_step = 0
        self.policy_params = None

        if self.log_neptune:
            self.neptune_project_name = config["neptune_project_name"]
            self.neptune_experiment_name = config["neptune_experiment_name"]
            self._init_neptune_client()
        self.start_logger(self.config)

    def _init_neptune_client(self):
        try:
            neptune.init(self.neptune_project_name)
        except neptune.api_exceptions.ProjectNotFound:
            sys.stderr.write(
                "Project name does not exist. Please create the project with the name '"
                + self.neptune_project_name.split("/")[1]
                + "' for user '"
                + self.neptune_project_name.split("/")[0]
                + "' in the web application. Exiting!"
            )
            sys.exit(-1)

    def start_logger(self, *args: Dict):
        if self.log_neptune:
            self._create_neptune_experiment(args)
        else:
            print(
                "The parameter log_neptune is set to false, no results will be logged!"
            )

    def _create_neptune_experiment(self, experiment_params: Tuple[Dict]):
        params_to_track = global_utils.flatten_tuple_of_dicts(experiment_params)
        neptune.create_experiment(
            name=self.neptune_experiment_name, params=params_to_track
        )
        self._push_neptune_tags(self._select_neptune_tags(params_to_track))

    @staticmethod
    def _push_neptune_tags(neptune_tags: List[str]):
        for tag in neptune_tags:
            neptune.append_tag(tag)

    @staticmethod
    def _select_neptune_tags(params_to_track: Dict) -> List[str]:
        neptune_tags = [
            params_to_track["trainer"],
            params_to_track["env_config"],
            "seed: " + str(params_to_track["seed"]),
            "dependency_matrix_type: " + params_to_track["dependency_matrix_type"],
            "connectivity_matrix_sampling_type: "
            + params_to_track["connectivity_matrix_sampling_type"],
            "communication_matrix_sampling_type: "
            + params_to_track["communication_matrix_sampling_type"],
        ]
        return neptune_tags

    def set_current_time_step(self, time_step: int):
        self.current_time_step = time_step

    def log_scalar_value(self, log_name: str, param: float):
        if self.log_neptune and (
            self.current_time_step % self.config["neptune_log_interval"] == 0
            or self.current_time_step == 1
        ):
            neptune.log_metric(log_name, param)
        if self.log_local:
            self.log_local_metric(log_name, param)

    def log_ltr(
        self,
        average_reward_rolling_windows: Deque[float],
        average_rewards_log: List[float],
        connectivity_matrix: np.ndarray,
    ):
        avg_reward = average_rewards_log[-1]
        ltr = float(np.mean(average_rewards_log))
        ltr_window = float(np.mean(average_reward_rolling_windows))
        num_active_edges = float(np.sum(connectivity_matrix > 0))
        self.log_scalar_value("avg_reward", avg_reward)
        self.log_scalar_value("long_term_average_return", ltr)
        self.log_scalar_value("long_term_average_return_window", ltr_window)
        self.log_scalar_value("ltr_per_num_active_edges", ltr / num_active_edges)
        self.log_scalar_value(
            "ltr_window_per_num_active_edges", ltr_window / num_active_edges
        )
        self.log_scalar_value("num_active_edges", num_active_edges)

    def log_norm_of_policy(self, policy_params: np.ndarray):
        policy_norm_t_plus_1 = global_utils.calculate_one_norm(policy_params)
        norm_difference_t_to_t_plus_1 = global_utils.calculate_one_norm(
            self.policy_params - policy_params
        )

        self.log_scalar_value("norm_policy_params", policy_norm_t_plus_1)
        self.log_scalar_value(
            "norm_changes_in_policy_t_to_t_plus_1", norm_difference_t_to_t_plus_1
        )

        self.policy_params = np.copy(policy_params)

    def init_policy_params_to_log(self, policy_params: np.ndarray):
        self.policy_params = np.copy(policy_params)

    def log_dependency_matrix(self, env_dependency_matrix: np.ndarray):
        self.log_coordination_matrix(
            env_dependency_matrix, 0, "Analytical", "DependencyMatrix", "EnvValue"
        )

    def _log_ind_impact_metrics(
        self,
        comparison_name: str,
        estimate_name: str,
        im_comparison_value: np.ndarray,
        im_estimate: np.ndarray,
        tim_comparison_value: np.ndarray,
        tim_estimate: np.ndarray,
    ):
        self.log_2dim_array(tim_estimate, "tim_estimate_" + estimate_name)
        self.log_2dim_array(tim_comparison_value, "tim_" + comparison_name)
        self.log_state_plus_2dim_array(im_estimate, "im_estimate_" + estimate_name)
        self.log_state_plus_2dim_array(im_comparison_value, "im_" + comparison_name)

    def _log_tim_agent_0(
        self,
        comparison_name: str,
        estimate_name: str,
        tim_comparison_value: np.ndarray,
        tim_estimate: np.ndarray,
    ):
        self.log_first_row_of_2dim_array(tim_estimate, "tim_estimate_" + estimate_name)
        self.log_first_row_of_2dim_array(tim_comparison_value, "tim_" + comparison_name)

    def _log_impact_error_metrics(
        self,
        comparison_name: str,
        estimate_name: str,
        im_comparison_value: np.ndarray,
        im_estimate: np.ndarray,
        tim_comparison_value: np.ndarray,
        tim_estimate: np.ndarray,
    ):
        self.log_matrix_norm(
            im_estimate,
            im_comparison_value,
            "im_estimation_error_" + estimate_name + "_" + comparison_name,
        )
        self.log_matrix_norm(
            tim_estimate,
            tim_comparison_value,
            "tim_estimation_error_" + estimate_name + "_" + comparison_name,
        )

    def log_impact_metrics(
        self,
        comparison_name: str,
        estimate_name: str,
        im_comparison_value: np.ndarray,
        im_estimate: np.ndarray,
        tim_comparison_value: np.ndarray,
        tim_estimate: np.ndarray,
    ):
        """
        Handle the logging of impact metrics
        :param comparison_name: Which values it should be compared (e.g. 'analytical')
        :param estimate_name: Name of estimation method
        :param im_comparison_value:
        :param im_estimate:
        :param tim_comparison_value:
        :param tim_estimate:
        """
        normalized_im_comparison_value = global_utils.normalize_to_coordination_matrix(
            im_comparison_value
        )
        normalized_im_estimate = global_utils.normalize_to_coordination_matrix(
            im_estimate
        )
        normalized_tim_comparison_value = global_utils.normalize_to_coordination_matrix(
            tim_comparison_value
        )
        normalized_tim_estimate = global_utils.normalize_to_coordination_matrix(
            tim_estimate
        )

        if self.config["neptune_ind_impact_metrics"]:
            self._log_ind_impact_metrics(
                comparison_name,
                estimate_name,
                im_comparison_value,
                im_estimate,
                tim_comparison_value,
                tim_estimate,
            )
            self._log_ind_impact_metrics(
                "normalized_" + comparison_name,
                "normalized_" + estimate_name,
                normalized_im_comparison_value,
                normalized_im_estimate,
                normalized_tim_comparison_value,
                normalized_tim_estimate,
            )
        if self.config["log_tim_agent_0"]:
            self._log_tim_agent_0(
                comparison_name, estimate_name, tim_comparison_value, tim_estimate
            )

        if self.config["neptune_impact_error_metrics"]:
            self._log_impact_error_metrics(
                comparison_name,
                estimate_name,
                im_comparison_value,
                im_estimate,
                tim_comparison_value,
                tim_estimate,
            )
            self._log_impact_error_metrics(
                "normalized_" + comparison_name,
                "normalized_" + estimate_name,
                normalized_im_comparison_value,
                normalized_im_estimate,
                normalized_tim_comparison_value,
                normalized_tim_estimate,
            )
            # log individual matrix norms
            self.log_matrix_norm(
                im_estimate, np.zeros(im_estimate.shape), "im_estimation_norm"
            )
            self.log_matrix_norm(
                im_comparison_value,
                np.zeros(im_comparison_value.shape),
                "im_analytical_norm",
            )
            self.log_matrix_norm(
                tim_estimate, np.zeros(tim_estimate.shape), "tim_estimation_norm"
            )
            self.log_matrix_norm(
                tim_comparison_value,
                np.zeros(tim_comparison_value.shape),
                "tim_analytical_norm",
            )

        if (
            self.log_neptune
            and self.current_time_step % self.config["neptune_image_log_interval"] == 0
            or self.current_time_step == 0
        ):
            self.log_coordination_matrix(
                tim_estimate, self.current_time_step, estimate_name, "estimation", "TIM"
            )
            self.log_coordination_matrix(
                tim_comparison_value,
                self.current_time_step,
                estimate_name,
                "analytical",
                "TIM",
            )

            self.log_coordination_matrix(
                normalized_tim_estimate,
                self.current_time_step,
                estimate_name,
                "estimation",
                "normalized_TIM",
            )
            self.log_coordination_matrix(
                normalized_tim_comparison_value,
                self.current_time_step,
                estimate_name,
                "analytical",
                "normalized_TIM",
            )

    def log_matrix_norm(self, matrix_1: np.ndarray, matrix_2: np.ndarray, name: str):
        one_distance = np.mean(np.abs(matrix_1 - matrix_2))
        self.log_scalar_value(name, float(one_distance))

    def log_state_plus_2dim_array(self, matrix: np.ndarray, name: str):
        assert len(matrix.shape) == 3, "matrix has to be 3-dimensional"
        for state in range(matrix.shape[0]):
            self.log_2dim_array(matrix[state, :, :], name + "_state_" + str(state))

    def log_2dim_array(self, matrix: np.ndarray, name: str):
        assert len(matrix.shape) == 2, "matrix has to be 2-dimensional"
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                log_name = name + "_" + str(row) + "_to_" + str(col)
                self.log_scalar_value(log_name, matrix[row, col])

    def log_first_row_of_2dim_array(self, matrix: np.ndarray, name: str):
        assert len(matrix.shape) == 2, "matrix has to be 2-dimensional"
        for col in range(matrix.shape[1]):
            log_name = name + "_" + str(0) + "_to_" + str(col)
            self.log_scalar_value(log_name, matrix[0, col])

    def log_coordination_matrix(
        self,
        matrix: np.ndarray,
        time_step: int,
        critic_type: str,
        estimation_type: str,
        measure_type: str,
    ):
        if self.log_neptune:

            plot_title = self.get_plot_title_for_matrices(
                measure_type, estimation_type, critic_type, time_step
            )

            # figure = plt.figure()

            im = plt.imshow(
                matrix, interpolation="none", vmin=0, vmax=1, aspect="equal"
            )

            ax = plt.gca()

            # Major ticks
            ax.set_xticks(np.arange(0, self.config["num_agents"], 1))
            ax.set_yticks(np.arange(0, self.config["num_agents"], 1))

            # Labels for major ticks
            ax.set_xticklabels(np.arange(0, self.config["num_agents"], 1))
            ax.set_yticklabels(np.arange(0, self.config["num_agents"], 1))

            # Minor ticks
            ax.set_xticks(np.arange(-0.5, self.config["num_agents"], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.config["num_agents"], 1), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

            if measure_type != "EnvValue":
                plt.colorbar()
            plt.title(plot_title)
            log_name = (
                "coordination_matrix_for_"
                + critic_type
                + "_"
                + estimation_type
                + "_"
                + measure_type
            )
            description = (
                "Estimation_type_"
                + estimation_type
                + "_critic_"
                + critic_type
                + "_for_"
                + measure_type
                + "_time_step_"
                + str(time_step)
            )
            figure = im.get_figure()
            neptune.log_image(x=figure, log_name=log_name, description=description)
            plt.close()

    def get_plot_title_for_matrices(
        self, measure_type: str, estimation_type: str, critic_type: str, time_step: int
    ) -> str:
        if measure_type == "EnvValue":
            return self.get_plot_title_for_dependency_matrix()
        elif estimation_type == "analytical":
            return r"$TI^{ana}(\pi_{\theta})$ matrix"
        elif estimation_type == "estimation" and critic_type == "sarsa_tabular":
            return r"$TI^{\eta_t}_t(\pi_{\theta})$ matrix for t = " + str(time_step)
        else:
            return "Coordination Matrix for : " + measure_type

    def get_plot_title_for_dependency_matrix(self) -> str:
        if self.config["dependency_matrix_type"] == "off_diagonals":
            return (
                r"Influence structure: connected chains $D_{offsets}=$"
                + r"$"
                + str(self.config["num_off_diagonals"])
                + r"$"
            )
        elif self.config["dependency_matrix_type"] == "block_diagonal_matrix":
            return (
                r"Influence structure: disconnected groups $G_{size}=$"
                + r"$"
                + str(self.config["dependency_block_diagonal_size"])
                + r"$"
            )
        elif self.config["dependency_matrix_type"] == "fix_num_additional_edges":
            return (
                r"Influence structure: randomly sampled $L_{add}^{dir}=$"
                + r"$"
                + str(2 * self.config["env_num_dependency_edges"])
                + r"$"
            )
        else:
            return "Binary dependencies among agents"

    def del_neptune(self):
        if self.log_neptune:
            neptune.stop()

    def log_local_metric(self, log_name: str, param: float):
        if log_name not in self.local_log_dict:
            self.local_log_dict[log_name] = []
        self.local_log_dict[log_name].append(param)

    def log_to_local_folder(self):
        if self.log_local:
            save_dir = (
                self.config["project_root_dir"]
                + "/local_logs/"
                + self.config["log_local_folder_name"]
            )
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            filename = "seed_" + str(self.config["seed"])
            self.local_log_dict["config"] = self.config
            global_utils.save_json(save_dir + "/" + filename, self.local_log_dict)
