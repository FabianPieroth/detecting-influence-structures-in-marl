import sys

import numpy as np

import global_utils
from environment.env_base import MathEnv


class NetworkSampler(object):
    def __init__(self, env: MathEnv, config: dict):
        self.config = config
        self.env = env
        self.static_communication_graph = False
        self.use_static_connectivity_matrix = False
        self.connectivity_matrix_sampling_type = config[
            "connectivity_matrix_sampling_type"
        ]
        self.communication_matrix_sampling_type = config[
            "communication_matrix_sampling_type"
        ]
        self.communication_matrix = np.ones(
            (self.env.num_agents, self.env.num_agents), dtype=int
        )
        self.connectivity_matrix = np.eye(self.env.num_agents)

    def sample_communication_matrix(self, matrix: np.ndarray = None):
        if self.communication_matrix_sampling_type == "full":
            self.communication_matrix = np.ones(
                (self.env.num_agents, self.env.num_agents), dtype=int
            )
        elif (
            self.communication_matrix_sampling_type == "connectivity_ratio"
            or self.communication_matrix_sampling_type == "fix_num_additional_edges"
        ):
            self.communication_matrix = (
                self._get_randomly_sampled_communication_matrix()
            )
        elif self.communication_matrix_sampling_type == "static_communication_graph":
            if not self.static_communication_graph:
                self.communication_matrix = (
                    self._get_randomly_sampled_communication_matrix()
                )
            self.static_communication_graph = True
        elif "largest_values" in self.communication_matrix_sampling_type:
            if not self.static_communication_graph:
                self.communication_matrix = self._get_masking_matrix_for_largest_values(
                    matrix
                )
                if "static" in self.communication_matrix_sampling_type:
                    self.static_communication_graph = True
        else:
            raise ValueError(
                "No valid communication_matrix_sampling_type selected! Use 'full', 'connectivity_ratio',"
                "'fix_num_additional_edges', 'static_communication_graph', 'largest_values', 'largest_values_connectivity_ratio'"
            )

    def _get_masking_matrix_for_largest_values(self, matrix: np.ndarray) -> np.ndarray:
        num_agents = self.communication_matrix.shape[0]
        masking_matrix = np.eye(num_agents, dtype=int)
        non_symmetric_matrix = np.zeros(matrix.shape, dtype=int)
        largest_values_indices = []
        num_edges = global_utils.calc_num_sampling_edges(
            self.communication_matrix_sampling_type,
            self.config["comm_num_additional_edges"],
            self.config["connectivity_ratio"],
            self.env.num_agents,
        )
        linear_indices = matrix.argsort(axis=None)[-(2 * num_edges + num_agents) :][
            ::-1
        ]
        diagonal_linear_indices = np.ravel_multi_index(
            np.diag_indices(num_agents), matrix.shape
        )
        # filter diagonal and symmetric indices
        linear_indices = [
            index for index in linear_indices if index not in diagonal_linear_indices
        ]
        for linear_index in linear_indices:
            if len(largest_values_indices) >= num_edges:
                break
            elif (
                global_utils.get_transposed_linear_index(linear_index, matrix.shape)
                not in largest_values_indices
            ):
                largest_values_indices.append(linear_index)
        if len(largest_values_indices) > 0:
            non_symmetric_matrix[
                np.unravel_index(largest_values_indices, matrix.shape)
            ] = 1
        return masking_matrix + non_symmetric_matrix + non_symmetric_matrix.T

    def _get_randomly_sampled_communication_matrix(self):
        adj_matrix = global_utils.sample_adjacency_matrix_randomly(
            self.communication_matrix_sampling_type,
            self.config["comm_num_additional_edges"],
            self.config["comm_connectivity_ratio"],
            self.env.num_agents,
        )
        return np.diag(np.ones(self.env.num_agents, dtype=int)) + adj_matrix

    def _get_full_consensus_cm(self) -> np.ndarray:
        return np.ones((self.env.num_agents, self.env.num_agents)) / self.env.num_agents

    def _get_absolute_laplacian(self) -> np.ndarray:
        adj_matrix = global_utils.sample_adjacency_matrix_randomly(
            self.communication_matrix_sampling_type,
            self.config["comm_num_additional_edges"],
            self.config["ct_connectivity_ratio"],
            self.env.num_agents,
        )
        node_degrees = np.sum(adj_matrix, axis=1)
        node_degrees[
            node_degrees == 0
        ] = 1  # make sure that there exists at least a self connection
        degree_matrix = np.diag(node_degrees)
        return np.abs(degree_matrix - adj_matrix)

    def _get_row_normalized_laplacian(self) -> np.ndarray:
        return global_utils.normalize_rows(self._get_absolute_laplacian())

    def _get_doubly_stochastic_laplacian(self) -> np.ndarray:
        absolute_laplacian = self._get_absolute_laplacian()
        return global_utils.convert_matrix_to_doubly_stochastic(absolute_laplacian)

    def _get_metropolis_weights(self, base_matrix: np.ndarray) -> np.ndarray:
        adj_matrix = base_matrix - np.diag(np.ones(self.env.num_agents, dtype=int))
        connectivity_matrix = np.zeros(adj_matrix.shape)
        degree_matrix = adj_matrix.sum(axis=1)
        non_zero_indices = np.where(adj_matrix == 1)
        for i, j in zip(non_zero_indices[0], non_zero_indices[1]):
            connectivity_matrix[i, j] = 1.0 / (
                1.0 + max(degree_matrix[i], degree_matrix[j])
            )
        connectivity_matrix += np.diag(
            np.ones(degree_matrix.shape[0]) - connectivity_matrix.sum(axis=1)
        )
        return connectivity_matrix

    def _get_tim_to_doubly_stochastic(self, tim_est: np.ndarray) -> np.ndarray:
        masked_tim_est = self._apply_communication_graph_masking(tim_est)
        return global_utils.convert_matrix_to_doubly_stochastic(masked_tim_est)

    def _get_im_to_doubly_stochastic(
        self, im_est: np.ndarray, state: int
    ) -> np.ndarray:
        masked_im_est = self._apply_communication_graph_masking(im_est[state, :, :])
        return global_utils.convert_matrix_to_doubly_stochastic(masked_im_est)

    def _apply_communication_graph_masking(self, matrix: np.ndarray) -> np.ndarray:
        self.sample_communication_matrix(matrix)
        return self.communication_matrix * matrix

    def draw_connectivity_matrix(self, **kwargs) -> np.ndarray:
        if not self.use_static_connectivity_matrix:
            self._update_connectivity_matrix(**kwargs)
            if self.config["use_static_connectivity_matrix"]:
                self.use_static_connectivity_matrix = True
        return self.connectivity_matrix

    def _update_connectivity_matrix(self, **kwargs):
        if self.connectivity_matrix_sampling_type == "full_consensus":
            self.connectivity_matrix = self._get_full_consensus_cm()
        elif self.connectivity_matrix_sampling_type == "row_norm_laplacian":
            self.connectivity_matrix = self._get_row_normalized_laplacian()
        elif self.connectivity_matrix_sampling_type == "doubly_stochastic_laplacian":
            self.connectivity_matrix = self._get_doubly_stochastic_laplacian()
        elif self.connectivity_matrix_sampling_type == "sample_metropolis_weights":
            self.sample_communication_matrix()
            self.connectivity_matrix = self._get_metropolis_weights(
                self.communication_matrix
            )
        elif self.connectivity_matrix_sampling_type == "tim_estimation":
            self.connectivity_matrix = self._get_tim_to_doubly_stochastic(
                kwargs["tim_estimate"]
            )
        elif self.connectivity_matrix_sampling_type == "tim_estimation_for_comm_graph":
            self.sample_communication_matrix(kwargs["tim_estimate"])
            self.connectivity_matrix = self._get_metropolis_weights(
                self.communication_matrix
            )
        elif self.connectivity_matrix_sampling_type == "im_estimation":
            self.connectivity_matrix = self._get_im_to_doubly_stochastic(
                kwargs["im_estimate"], kwargs["state"]
            )
        elif self.connectivity_matrix_sampling_type == "tim_analytical":
            self.connectivity_matrix = self._get_tim_to_doubly_stochastic(
                kwargs["tim_analytical"]
            )
        elif self.connectivity_matrix_sampling_type == "im_analytical":
            self.connectivity_matrix = self._get_im_to_doubly_stochastic(
                kwargs["im_analytical"], kwargs["state"]
            )
        elif self.connectivity_matrix_sampling_type == "block_diagonal_matrix":
            block_matrix = global_utils.get_block_diagonal_matrix(
                self.env.num_agents, self.config["dependency_block_diagonal_size"]
            )
            self.connectivity_matrix = self._get_metropolis_weights(block_matrix)
        elif self.connectivity_matrix_sampling_type == "off_diagonals":
            off_diagonals_adj_matrix = global_utils.get_off_diagonals_adj_matrix(
                self.env.num_agents, self.config["num_off_diagonals"]
            )
            self.connectivity_matrix = self._get_metropolis_weights(
                off_diagonals_adj_matrix
            )
        elif self.connectivity_matrix_sampling_type == "dependency_matrix":
            dependency_matrix = self.env.dependency_matrix
            self.connectivity_matrix = self._get_metropolis_weights(dependency_matrix)
        else:
            sys.exit(
                'No valid sampling_type selected! Please choose between: "full_consensus", "row_norm_laplacian", '
                '"doubly_stochastic_laplacian", "tim_estimation", "im_estimation", "tim_analytical", "im_analytical",'
                '"block_diagonal_matrix", "off_diagonals", "dependency_matrix", "sample_metropolis_weights"'
            )
