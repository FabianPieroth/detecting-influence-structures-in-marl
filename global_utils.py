import collections
import difflib
import gc
import json
import math
import os
import random
import traceback
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg as sla
from sinkhorn_knopp import sinkhorn_knopp as skp

from models.actors_and_critics.zhang_decentralized import ZhangDecentralizedActor
from runner import Runner
from trainer.trainer_env_initializer import TrainerEnvInitializer


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return


def ret_all_files_in_folder(folder_path, full_names=True):
    files = [s for s in os.listdir(folder_path) if filter_hidden_files(s)]
    if full_names:
        files = [folder_path + "/" + s for s in files]
    return files


def filter_hidden_files(string):
    bool1 = ".DS_" in string
    bool2 = "._" in string
    bool3 = "descriptions.csv" in string
    bool4 = "extra" in string

    return not any([bool1, bool2, bool3, bool4])


def ret_file_name_from_path(file_path: str) -> str:
    return file_path.split("/")[-1].split(".")[0]


def load_json(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}


def save_json(filename: str, data):
    with open(filename + ".json", "w") as fp:
        json.dump(data, fp)


def get_differences_in_strings(string_list: List[str]) -> str:
    if len(string_list) == 1:
        filtered_string_element = "".join(
            e for e in string_list[0] if e.isalnum() or e == " "
        )
        return filtered_string_element
    else:
        # filter special characters
        start_string = "".join(e for e in string_list[0] if e.isalnum() or e == " ")
        previous_string = start_string
        return_string = start_string
        for str_element in string_list[1:]:
            filtered_string_element = "".join(
                e for e in str_element if e.isalnum() or e == " "
            )
            str_diff_list = [
                li[-1]
                for li in difflib.ndiff(previous_string, filtered_string_element)
                if li[0] != " " and "+" in li
            ]
            previous_string = filtered_string_element
            return_string += "_AND_" + "".join(str_diff_list)
        return return_string


def start_single_experiment(params: Tuple[Dict, str, Dict]):
    experiment_config, project_root_dir, runner_config = params
    runner_config.update(experiment_config)
    trainer_config = TrainerEnvInitializer.get_trainer_configs(
        runner_config["trainer"], project_root_dir
    )
    ind_critic_config = TrainerEnvInitializer.get_trainer_configs(
        runner_config["ind_critic_type"], project_root_dir
    )
    env_config = TrainerEnvInitializer.get_env_config(
        runner_config["env_config"], project_root_dir
    )
    impact_estimator_config = TrainerEnvInitializer.get_impact_estimator_config(
        project_root_dir
    )
    impact_sampler_config = TrainerEnvInitializer.get_impact_sampler_config(
        project_root_dir
    )
    analytical_critic_config = TrainerEnvInitializer.get_analytical_critic_config(
        project_root_dir
    )
    # Combine configs; set runner_config as last!
    combined_config = integrate_and_overwrite_dicts(
        trainer_config,
        ind_critic_config,
        env_config,
        impact_estimator_config,
        impact_sampler_config,
        analytical_critic_config,
        runner_config,
    )
    runner = Runner(combined_config)
    try_except_experiment_with_traceback(runner)
    reset_runner(runner)  # Needed if several experiments should be done in one run


def try_except_experiment_with_traceback(runner: Runner):
    try:
        runner.run()
    except Exception as err:
        try:
            raise TypeError("Test Error?")
        except:
            pass

        traceback.print_tb(err.__traceback__)


def plot_sequence(seq: np.ndarray):
    plt.plot(seq)
    plt.ylabel("Average Reward")
    plt.show()


def flatten_tuple_of_dicts(tuple_of_dicts: Tuple[Dict]) -> Dict:
    flat_dict = {}
    for dictionary in tuple_of_dicts:
        flat_dict.update(flatten_dict(dictionary))
    return flat_dict


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(dict(v), new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def integrate_and_overwrite_dicts(*args) -> Dict:
    """
    Extend the sub_dict by all key pairs of the main_dict and
    overwrites values of existing keys by the ones in main_dict
    """
    combined_dict = {}
    for config in args:
        combined_dict.update(config)
    return combined_dict


def reset_runner(runner: Runner):
    # save local logs if parameter is set to true
    runner.logger.log_to_local_folder()
    # reset neptune so that a new experiment can be logged!
    runner.logger.del_neptune()
    # delete runner object
    del runner
    gc.collect()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]


def normalize_to_coordination_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    :param matrix: shape=(state_size, num_agents, num_agents) or shape=(num_agents, num_agents)
    :return: row normalized
    """
    if len(matrix.shape) == 3:
        row_sums = matrix.sum(axis=2)
        normalized = matrix / row_sums[:, :, np.newaxis]
    else:
        normalized = normalize_rows(matrix)
    np.nan_to_num(normalized, copy=False)
    return normalized


def convert_matrix_to_doubly_stochastic(matrix: np.ndarray) -> np.ndarray:
    sk = skp.SinkhornKnopp(max_iter=1000, epsilon=1e-3)
    return sk.fit(matrix)


def get_mask_for_largest_values(
    matrix: np.ndarray, connectivity_ratio: int
) -> np.ndarray:
    pass


def cast_to_decimal_from_array(
    number_representation: np.ndarray, decimal_base: int
) -> int:
    strings = [str(integer) for integer in number_representation]
    string_representation = "".join(strings)
    return int(string_representation, decimal_base)


def get_number_representation_from_int(
    index: int, decimal_base: int, length_representation: int
) -> List:
    representation_shape = tuple([decimal_base for i in range(length_representation)])
    return list(np.unravel_index(index, representation_shape))


def cast_decimal_to_other_base_to_list(
    integer: int, base: int, length_representation: int
) -> List:
    string_representation = decimal_to_string_in_other_base(integer, base)
    string_representation = pad_string_to_representation_length(
        length_representation, string_representation
    )

    return [int(string) for string in string_representation]


def pad_string_to_representation_length(length_representation, string_representation):
    if len(string_representation) < length_representation:
        string_representation = (
            "0" * (length_representation - len(string_representation))
            + string_representation
        )
    assert len(string_representation) <= length_representation
    return string_representation


def assign_correct_chars_to_num(num):
    # To return char for a value. For example
    # '2' is returned for 2. 'A' is returned
    # for 10. 'B' for 11
    if 0 <= num <= 9:
        return chr(num + ord("0"))
    else:
        return chr(num - 10 + ord("A"))


def decimal_to_string_in_other_base(
    integer: int, base: int, resulting_string: str = ""
):
    # Convert input number is given base
    # by repeatedly dividing it by base
    # and taking remainder
    while integer > 0:
        resulting_string += assign_correct_chars_to_num(integer % base)
        integer = int(integer / base)
        # Reverse the result
    resulting_string = resulting_string[::-1]
    return resulting_string


def get_transposed_linear_index(linear_index: int, matrix_shape: Tuple[int]) -> int:
    return int(
        np.ravel_multi_index(
            tuple(reversed(np.unravel_index(linear_index, matrix_shape))), matrix_shape
        )
    )


def softmax(logits: np.ndarray, single_axis=False) -> np.ndarray:
    e_logits = np.exp(logits - np.max(logits))
    if single_axis:
        return e_logits / e_logits.sum()
    else:
        return e_logits / e_logits.sum(axis=1)[:, np.newaxis]


def vector_product_state_action_function(
    parameters: np.ndarray, embedding: np.ndarray
) -> np.ndarray:
    return (parameters * embedding[np.newaxis, :]).sum(axis=1)


def select_state_emb_for_policy_by_specific_action(
    array: np.ndarray, indices: List
) -> np.ndarray:
    sliced_array = np.zeros((array.shape[0], array.shape[1]))
    for i in range(len(indices)):
        sliced_array[i, :] = array[i, :, indices[i]]
    return sliced_array


def get_stationary_distribution_with_sanity_checks(matrix: np.ndarray) -> np.ndarray:
    vector = scipy.linalg.null_space(matrix - np.eye(matrix.shape[0])).squeeze()
    stationary_distribution = vector / vector.sum()
    assert np.isclose(
        stationary_distribution.sum(), 1.0
    ), "This vector should represent a probability distribution"
    assert all(
        np.isclose(
            np.matmul(stationary_distribution, matrix.T), stationary_distribution
        )
    ), "Check if left eigenvector is correct"
    return stationary_distribution


def init_learning_parameters(param_shape) -> np.ndarray:
    if isinstance(param_shape, int):
        return sample_uniformly(calc_sample_bound(param_shape), param_shape)
    elif len(param_shape) == 2:
        return sample_uniformly(calc_sample_bound(param_shape[1]), param_shape)
    else:
        raise NotImplementedError(
            "Currently only two different kinds of param shapes are implemented"
        )


def sample_uniformly(bound: float, shape) -> np.ndarray:
    return np.random.uniform(low=-bound, high=bound, size=shape)


def calc_sample_bound(size_in_features: int) -> float:
    return float(np.sqrt(1.0 / size_in_features))


def calc_num_edges_from_connectivity_ratio(ratio_param: int, num_agents: int) -> int:
    """
    Sampling method of the communication matrix by Zhang
    :param ratio_param: the ratio is calculated as ratio_param / num_agents
    :param num_agents:
    :return: num_edges = ratio_param / 2 * (num_agents - 1)
    """
    return math.floor(ratio_param / 2) * (num_agents - 1)


def calc_num_sampling_edges(
    sampling_type: str, num_additional_edges: int, ratio_param: int, num_agents: int
) -> int:
    if "connectivity_ratio" in sampling_type:
        return calc_num_edges_from_connectivity_ratio(ratio_param, num_agents)
    else:
        return num_additional_edges


def sample_off_diagonal_elements(num_edges: int, size_matrix: int) -> List[List[int]]:
    assert num_edges <= size_matrix * (
        size_matrix - 1
    ), "One cannot draw more than there is to draw!"
    off_diagonal_indices = get_off_diagonal_indices(size_matrix)
    return random.sample(off_diagonal_indices, num_edges)


def get_off_diagonal_indices(size_matrix: int) -> List[List[int]]:
    indices_1, indices_2 = np.unravel_index(
        list(range(size_matrix ** 2)), (size_matrix, size_matrix)
    )
    return [
        [index_1, index_2]
        for index_1, index_2 in zip(indices_1, indices_2)
        if not index_1 == index_2
    ]


def sample_upper_diagonal_elements(num_edges: int, size_matrix: int) -> List[List[int]]:
    upper_diagonal_indices = np.transpose(np.triu_indices(size_matrix, k=1)).tolist()
    return random.sample(upper_diagonal_indices, num_edges)


def sample_adjacency_matrix_randomly(
    communication_matrix_type: str,
    num_additional_edges: int,
    ratio_param: int,
    num_agents: int,
) -> np.ndarray:
    num_edges = calc_num_sampling_edges(
        communication_matrix_type, num_additional_edges, ratio_param, num_agents
    )
    upper_diagonal_indices = sample_upper_diagonal_elements(num_edges, num_agents)
    upper_diagonal_elements_transpose = np.array(upper_diagonal_indices).T.tolist()
    upper_diagonal_matrix = np.zeros((num_agents, num_agents), dtype=int)
    if len(upper_diagonal_elements_transpose) > 0:
        upper_diagonal_matrix[tuple(upper_diagonal_elements_transpose)] = 1
    return upper_diagonal_matrix + upper_diagonal_matrix.T


def sample_directed_adjacency_matrix(
    dependency_matrix_type: str,
    env_num_dependency_edges: int,
    ratio_param: int,
    num_agents: int,
) -> np.ndarray:
    num_edges = 2 * calc_num_sampling_edges(
        dependency_matrix_type, env_num_dependency_edges, ratio_param, num_agents
    )
    off_diagonal_indices = sample_off_diagonal_elements(num_edges, num_agents)
    off_diagonal_elements_transpose = np.array(off_diagonal_indices).T.tolist()
    adj_matrix = np.zeros((num_agents, num_agents), dtype=int)
    if len(off_diagonal_elements_transpose) > 0:
        adj_matrix[tuple(off_diagonal_elements_transpose)] = 1
    return adj_matrix


def sample_directed_matrix(
    dependency_matrix_type: str,
    env_num_dependency_edges: int,
    ratio_param: int,
    num_agents: int,
) -> np.ndarray:
    comm_matrix = np.eye(num_agents, dtype=int)
    adj_matrix = sample_directed_adjacency_matrix(
        dependency_matrix_type, env_num_dependency_edges, ratio_param, num_agents
    )
    return comm_matrix + adj_matrix


def get_block_diagonal_matrix(matrix_size: int, block_size: int) -> np.ndarray:
    block_diagonal_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    assert block_size >= 1, "The block size needs to be positive!"
    if block_size >= matrix_size:
        return np.ones(block_diagonal_matrix.shape, dtype=int)
    num_blocks, _ = divmod(matrix_size, block_size)
    for k in range(num_blocks):
        block_diagonal_matrix[
            k * block_size : (k + 1) * block_size, k * block_size : (k + 1) * block_size
        ] = 1
    block_diagonal_matrix[num_blocks * block_size :, num_blocks * block_size :] = 1

    return block_diagonal_matrix


def get_off_diagonals_adj_matrix(
    matrix_size: int, num_off_diagonals: int
) -> np.ndarray:
    adj_matrix = np.eye(matrix_size)
    for i in range(1, num_off_diagonals + 1):
        upper_off_diagonal = np.diag(np.ones(matrix_size - i), i)
        lower_off_diagonal = np.diag(np.ones(matrix_size - i), -i)
        adj_matrix += upper_off_diagonal + lower_off_diagonal
    return adj_matrix


def get_corresponding_metric_name(
    metric_name: str, json_file: Dict, num_folder: int
) -> str:
    if "tim_estimation" in metric_name:
        metric_name_list = [
            name
            for name in list(json_file.keys())
            if metric_name in name and "normalized" not in name
        ]
        return metric_name_list[0]
    elif "im_estimation" in metric_name:
        metric_name_list = [
            name
            for name in list(json_file.keys())
            if metric_name in name and "normalized" not in name
        ]
        return metric_name_list[0]
    elif metric_name == "tim_analytical_0":
        return metric_name + "_to_" + str(num_folder)
    else:
        return metric_name


def add_noise_to_actor_params(
    actor: ZhangDecentralizedActor, noise_strength_decay: float, time_step: int
):
    noise = np.random.standard_normal(actor.policy.shape)
    actor.policy += noise_strength_decay ** time_step * noise


def calculate_one_norm(parameters: np.ndarray) -> float:
    return float(np.mean(np.abs(parameters)))
