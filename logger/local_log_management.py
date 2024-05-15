import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sys.path.append(os.path.realpath("."))

import global_utils
from logger.local_neptune_log_management import get_mean_std_df

sns.set(style="darkgrid")
sns.set_context(
    "paper",
    rc={
        "font.size": 35,
        "axes.titlesize": 25,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    },
)

from typing import List, Tuple


def get_combined_df(
    read_folder: str, metric_name: str, normalize_by_last_value: bool, num_folder: int
) -> pd.DataFrame:
    file_names = global_utils.ret_all_files_in_folder(read_folder)
    df_list = []

    for file_name in file_names:
        json_file = global_utils.load_json(file_name)

        adjusted_metric_name = global_utils.get_corresponding_metric_name(
            metric_name, json_file, num_folder
        )

        data_frame = pd.DataFrame(
            json_file[adjusted_metric_name],
            columns=[global_utils.ret_file_name_from_path(file_name)],
        )
        if normalize_by_last_value:
            data_frame = normalize_df_by_last_value(data_frame)
        df_list.append(data_frame)
    return pd.concat(df_list, axis=1, join="inner")


def normalize_df_by_last_value(df: pd.DataFrame) -> pd.DataFrame:
    # take the last value of the series and normalize the whole series by it
    col_name = df.columns[0]
    last_value = df[col_name].iloc[-1]
    df[col_name] = df[col_name].div(last_value)
    return df


def get_mean_std_df_from_folder(
    read_folder: str,
    metric_name: str,
    normalize_by_last_value: bool,
    num_folder: int,
    plot_mean_std: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_df = get_combined_df(
        read_folder, metric_name, normalize_by_last_value, num_folder
    )
    return get_upper_middle_and_lower_value_for_df(combined_df, plot_mean_std)


def get_upper_middle_and_lower_value_for_df(
    combined_df: pd.DataFrame, mean_std: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mean_std:
        mean_df, std_df = get_mean_std_df(combined_df)
        upper_df = mean_df + std_df
        lower_df = mean_df - std_df
        return upper_df, mean_df, lower_df
    else:
        return get_quantiles_from_combined_df(combined_df)


def get_quantiles_from_combined_df(
    combined_df: pd.DataFrame, quantile: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    median_df = combined_df.quantile(q=0.5, axis=1)
    upper_quantile_df = combined_df.quantile(q=quantile, axis=1)
    lower_quantile_df = combined_df.quantile(q=1 - quantile, axis=1)
    return upper_quantile_df, median_df, lower_quantile_df


def get_mean_std_df_list(
    read_folders: List[str],
    metric_names: List[str],
    normalize_by_last_value: bool,
    plot_mean_std: bool,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    mean_std_list = []

    for k, read_folder in enumerate(read_folders):
        if len(metric_names) == len(read_folders):
            l = k
        else:
            l = 0
        mean_std_list.append(
            get_mean_std_df_from_folder(
                read_folder, metric_names[l], normalize_by_last_value, k, plot_mean_std
            )
        )

    return mean_std_list


def append_folder_path_to_folder_names(
    folder_path: str, folder_names: List[str]
) -> List[str]:
    return [folder_path + folder_name for folder_name in folder_names]


def add_metric_type_to_title(title: str, metric_name: str) -> str:
    if metric_name == "long_term_average_return":
        return title
    elif metric_name == "long_term_average_return_window":
        return title
    elif metric_name == "tim_estimation_norm":
        return "Matrix norm of TIM Estimation"
    elif metric_name == "tim_analytical_norm":
        return "Matrix norm of TIM Analytical"
    else:
        print("No standard metric chosen: " + metric_name)
        return title


def get_correct_length_x_axis(x_coord: np.ndarray, df_data: pd.DataFrame) -> np.ndarray:
    if len(df_data) < len(x_coord):
        return x_coord[: len(df_data)]
    elif len(df_data) > len(x_coord):
        return np.array([float(k) for k in range(len(df_data))])
    else:
        return x_coord


def create_plot_from_df_list(
    labels: List[str],
    metric_name: str,
    title: str,
    save_dir: str,
    df_list: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    save_plot=False,
    normalize_by_last_value: bool = False,
    set_plot_title: bool = False,
    yaxis_label: str = "",
    num_plot: int = 0,
    plot_mean_std: bool = False,
    second_y_axis_data: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
    second_y_axis_metric: str=None,
    second_y_axis_label: str=None,
):
    if normalize_by_last_value:
        title += " Normalized by last Value"
    if not plot_mean_std:
        title += " quantiles plot"
    color_list = ["black", "red", "blue", "green", "purple", "orange", "brown"]
    upper_df_0, mean_df_0, lower_df_0 = df_list[0]
    ax = sns.lineplot(data=mean_df_0, color=color_list[0], label=labels[0])
    l1 = ax.lines[0]
    x_coord = l1.get_xydata()[:, 0]
    ax.fill_between(
        x=x_coord, y1=lower_df_0, y2=upper_df_0, color=color_list[0], alpha=0.3
    )
    for k in range(1, len(labels)):
        upper_df, mean_df, lower_df = df_list[k]
        ax = sns.lineplot(data=mean_df, color=color_list[k], label=labels[k])
        x_coord_adjusted = get_correct_length_x_axis(x_coord, lower_df)
        ax.fill_between(
            x=x_coord_adjusted, y1=lower_df, y2=upper_df, color=color_list[k], alpha=0.2
        )

    if set_plot_title:
        ax.set_title(title)
    ax.set_xlabel("time step t")
    ax.set_ylabel(yaxis_label)

    if second_y_axis_data:
        secondary_axis_color = "blue"
        ax2 = ax.twinx()
        upper_df, mean_df, lower_df = second_y_axis_data[0]
        x_coord_adjusted = get_correct_length_x_axis(x_coord, lower_df)
        ax2.plot(
            x_coord_adjusted,
            np.array(mean_df),
            color=secondary_axis_color,
            label=second_y_axis_metric,
        )
        ax2.fill_between(
            x=x_coord_adjusted,
            y1=lower_df,
            y2=upper_df,
            color=secondary_axis_color,
            alpha=0.2,
        )
        ax2.set_ylabel(second_y_axis_label, color=secondary_axis_color)
        ax2.tick_params(axis="y", labelcolor=secondary_axis_color)
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=(1.11, 0.01))
        ax.legend_ = None

    if not second_y_axis_data:
        handle_legend_placement(metric_name[0])

    save_plot_to_file(labels, metric_name[0], num_plot, save_dir, save_plot, title)


def handle_legend_placement(metric_name):
    if (
        "tim_estimation_error" in metric_name
        or "im_estimation_error" in metric_name
        or "norm_changes" in metric_name
    ):
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")


def save_plot_to_file(labels, metric_name, num_plot, save_dir, save_plot, title):
    if save_plot:
        save_name = global_utils.get_differences_in_strings(labels)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        plt.savefig(
            save_dir
            + "/"
            + metric_name
            + "_"
            + title
            + "_"
            + save_name
            + str(num_plot)
            + ".png",
            bbox_inches="tight",
            dpi=600,
        )
    else:
        plt.show()
    plt.close()


def main():
    project_root_dir = str(Path().resolve().parents[0])

    scenario_name = "static_policy"

    save_dir = project_root_dir + "/detecting-influence-structures-in-marl/local_logs/" + scenario_name + "/"

    path_to_read_folder = (
        project_root_dir + "/detecting-influence-structures-in-marl/local_logs/" + scenario_name + "/"
    )

    normalize_by_last_value = False

    read_folder_names_list = [
        [
            "impact-estimation-num-edges-0",
            "impact-estimation-num-edges-2",
            "impact-estimation-num-edges-6",
            "impact-estimation-num-edges-10",
        ],
        [
            "impact-estimation-num-edges-0",
            "impact-estimation-num-edges-2",
            "impact-estimation-num-edges-6",
            "impact-estimation-num-edges-10",
        ]
    ]

    labels_list = [
        [
            r"$L_{add}=0$",
            r"$L_{add}=4$",
            r"$L_{add}=12$",
            r"$L_{add}=20$",
        ],
        [
            r"$L_{add}=0$",
            r"$L_{add}=4$",
            r"$L_{add}=12$",
            r"$L_{add}=20$",
        ]
    ]

    title_list = ["TIM estimation error", "SIM estimation error"]

    metric_names_list = [["tim_estimation_error"], ["im_estimation_error"]]
    yaxis_label_list = [
        r"$||TI(\pi_{\theta}, \eta_t) - TI^{ana}(\pi_{\theta})||_1$",
        r"$||SI(\cdot, \pi_{\theta}, \eta_t) - SI^{ana}(\cdot, \pi_{\theta})||_1$",
    ]

    second_y_axis_metric = r"$\frac{1}{100}\sum_{k=t-99}^{t} r_{k+1}$"
    second_y_axis_label = ""

    set_plot_title_bool_list = [False, False, False, False]

    save_plot = True

    add_second_y_axis = False

    for (
        read_folder_names,
        labels,
        title,
        metric_names,
        yaxis_label,
        k_iter,
        set_plot_tile,
    ) in zip(
        read_folder_names_list,
        labels_list,
        title_list,
        metric_names_list,
        yaxis_label_list,
        list(range(50)),
        set_plot_title_bool_list,
    ):
        for plot_mean_std in [False]:
            title_adj = add_metric_type_to_title(title, metric_names[0])

            read_folders = append_folder_path_to_folder_names(
                path_to_read_folder, read_folder_names
            )

            df_list = get_mean_std_df_list(
                read_folders, metric_names, normalize_by_last_value, plot_mean_std
            )

            if add_second_y_axis:
                df_list_second_axis = get_mean_std_df_list(
                    read_folders, metric_names_list[1], normalize_by_last_value, False
                )
            else:
                df_list_second_axis = None

            create_plot_from_df_list(
                labels,
                metric_names,
                title_adj,
                save_dir,
                df_list,
                save_plot,
                normalize_by_last_value,
                set_plot_title=set_plot_tile,
                yaxis_label=yaxis_label,
                num_plot=k_iter,
                plot_mean_std=plot_mean_std,
                second_y_axis_data=df_list_second_axis,
                second_y_axis_metric=second_y_axis_metric,
                second_y_axis_label=second_y_axis_label
            )
            print("Finished plot:" + title)

    print("Done")


if __name__ == "__main__":
    main()
