import numpy as np
import matplotlib.pyplot as plt


def plot_matrix(matrix: np.ndarray):

    # figure = plt.figure()
    plt.matshow(matrix)

    #im = plt.imshow(
     #   matrix, interpolation="none", vmin=0, vmax=1, aspect="equal"
    #)

    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, matrix.shape[0], 1))
    ax.set_yticks(np.arange(0, matrix.shape[1], 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, matrix.shape[0]+1, 1))
    ax.set_yticklabels(np.arange(1, matrix.shape[1]+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(r"$TI^{ana}(\pi)$ matrix with $L_{add}^{dir}=8$", fontsize=20, y=-0.11)
    plt.savefig("./exemplary_TIM_matrix.png", dpi=600)


def main():
    example_TIM = np.array([
        [0.8, 0.0, 0.59, 0.77, 0.0],
        [0.59, 0.7, 0.0, 0.0, 0.75],
        [0.0, 0.6, 0.8, 0.0, 0.0],
        [0.0, 0.65, 0.1, 0.65, 0.0],
        [0.63, 0.0, 0.0, 0.0, 0.66],
    ])
    plot_matrix(example_TIM)


if __name__ == "__main__":
    main()