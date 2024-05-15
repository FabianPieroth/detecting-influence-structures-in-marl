from pathlib import Path

import matplotlib.pyplot as plt
import neptune
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

from typing import List, Tuple


def append_id_to_log_names(log_names: List[str], prefix: str):
    return [prefix + "_" + log_name for log_name in log_names]


def get_df_from_experiments(log_names, project, tag_list):
    experiments = project.get_experiments(tag=tag_list)
    df_list = []
    for experiment in experiments:
        df = experiment.get_numeric_channels_values(*log_names)
        df.columns = ["x"] + append_id_to_log_names(log_names, experiment.id)
        df.set_index("x")
        df_list.append(df)
    return pd.concat(df_list, axis=1, join="inner").drop(columns=["x"])


def get_mean_std_from_experiments(metric_name, project, tag_list_1):
    combined_df = get_df_from_experiments([metric_name], project, tag_list_1)
    return get_mean_std_df(combined_df)


def get_mean_std_df(combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mean_df = combined_df.mean(axis=1)
    std_df = combined_df.std(axis=1)
    return mean_df, std_df


def create_plot_from_neptune_exp(
    label_1: str,
    label_2: str,
    metric_name: str,
    neptune_project_name: str,
    tag_list_1: List[str],
    tag_list_2: List[str],
    save_dir: str,
    save_plot=False,
):
    project = neptune.init(neptune_project_name)
    mean_df_1, std_df_1 = get_mean_std_from_experiments(
        metric_name, project, tag_list_1
    )
    mean_df_2, std_df_2 = get_mean_std_from_experiments(
        metric_name, project, tag_list_2
    )
    create_plot(
        label_1,
        label_2,
        metric_name,
        save_dir,
        mean_df_1,
        std_df_1,
        mean_df_2,
        std_df_2,
        save_plot,
    )


def create_plot(
    label_1: str,
    label_2: str,
    metric_name: str,
    save_dir: str,
    mean_df_1,
    std_df_1,
    mean_df_2,
    std_df_2,
    save_plot=False,
):
    ax = sns.lineplot(data=mean_df_1, color="blue", label=label_1)
    l1 = ax.lines[0]
    x_coord = l1.get_xydata()[:, 0]
    ax.fill_between(
        x=x_coord,
        y1=mean_df_1 - std_df_1,
        y2=mean_df_1 + std_df_1,
        color="blue",
        alpha=0.3,
    )

    ax = sns.lineplot(data=mean_df_2, color="red", label=label_2)
    ax.fill_between(
        x=x_coord,
        y1=mean_df_2 - std_df_2,
        y2=mean_df_2 + std_df_2,
        color="red",
        alpha=0.3,
    )
    if save_plot:
        plt.savefig(
            save_dir + "/" + metric_name + "_" + label_1 + "_" + label_2 + ".png",
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close()


def main():
    save_dir = str(Path().resolve().parents[1]) + "/Tests/seaborn_unordered"

    neptune_project_name = "fabianpieroth/decentralized-im-hypersearch"

    tag_list_1 = [
        "dependency_matrix_type: full",
        "communication_matrix_sampling_type: largest_values",
        "connectivity_matrix_sampling_type: im_estimation",
    ]

    tag_list_2 = [
        "dependency_matrix_type: full",
        "communication_matrix_sampling_type: largest_values",
        "connectivity_matrix_sampling_type: tim_estimation",
    ]

    label_1 = "IM Estimation with largest values"
    label_2 = "TIM Estimation with largest values"

    metric_name = "long_term_average_return"

    create_plot_from_neptune_exp(
        label_1,
        label_2,
        metric_name,
        neptune_project_name,
        tag_list_1,
        tag_list_2,
        save_dir,
        save_plot=True,
    )

    print("Done")


if __name__ == "__main__":
    main()
