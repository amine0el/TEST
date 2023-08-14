import os
import glob
import numpy as np
from scipy import stats
from general import PATH
from seaborn import set_theme
import matplotlib.pyplot as plt

set_theme()


def get_data(path):
    """
    Gets the means and confidence intervals of the Js, Vs, and Hs from the
    path provided (e.g. one particular value of learning
    rate and n_features tested with multiple seeds)
    """
    data = {}

    # J
    Js = get_data_with_str(path, "J")
    J_mean, J_upper_conf, J_lower_conf = calculate_bounds(Js)
    data["J"] = {
        "mean": J_mean,
        "upper_conf": J_upper_conf,
        "lower_conf": J_lower_conf,
    }

    # V
    Vs = get_data_with_str(path, "V")
    V_mean, V_upper_conf, V_lower_conf = calculate_bounds(Vs)
    data["V"] = {
        "mean": V_mean,
        "upper_conf": V_upper_conf,
        "lower_conf": V_lower_conf,
    }

    # H
    Hs = get_data_with_str(path, "H")
    H_mean, H_upper_conf, H_lower_conf = calculate_bounds(Hs)
    data["H"] = {"mean": H_mean, "upper_conf": H_upper_conf, "lower_conf": H_lower_conf}

    return data


def get_data_with_str(results_path, string):
    """
    Gets all the .npy arrays containing a certain string in their filename
    """
    arrays = []
    pattern = os.path.join(results_path, f"*{string}*.npy")
    file_paths = glob.glob(pattern)
    for path in file_paths:
        arrays.append(np.load(path))
    return arrays


def calculate_bounds(arrays):
    means = []
    lower_confidence_bounds = []
    upper_confidence_bounds = []
    num_elements = len(arrays[0])
    for i in range(num_elements):
        elements = [array[i] for array in arrays]
        mean = np.mean(elements)
        std_dev = np.std(elements)
        n = len(elements)
        t_value = stats.t.ppf(0.95, n - 1)
        std_error = std_dev / np.sqrt(n)
        margin_of_error = t_value * std_error
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        means.append(mean)
        lower_confidence_bounds.append(lower_bound)
        upper_confidence_bounds.append(upper_bound)
    return means, upper_confidence_bounds, lower_confidence_bounds


def main():

    # param = "test"
    # value = "test"

    # Get data
    results_path_1 = "/Users/amineelblidi/Documents/Bachlor vorbereitung/code/TEST/data/Hypervolume/pql"
    data_1 = get_data(results_path_1)

    # results_path_2 = "/home/kurse/kurs00067/me35goky/HW3/logs/results/tune_ppo/0.05"
    # data_2 = get_data(results_path_2)

    # results_path_3 = "/home/kurse/kurs00067/me35goky/HW3/logs/results/tune_ppo/0.5"
    # data_3 = get_data(results_path_3)

    # results_path_4 = "/home/kurse/kurs00067/me35goky/HW3/logs/results/tune_general_params/n_features_512/learning_rate_0.5"
    # data_4 = get_data(results_path_4)

    # results_path_5 = "/home/kurse/kurs00067/me35goky/HW3/logs/results/tune_general_params/n_features_512/learning_rate_1e-05"
    # data_5 = get_data(results_path_5)

    x = range(len(data_1["J"]["mean"]))

    # Plot
    fig, ax = plt.subplots(2, 1)

    # First dataset J, V & H
    ax[0].plot(x, data_1["J"]["mean"], color="red", linestyle="-", label="α=0.1: J")
    ax[0].fill_between(
        x,
        data_1["J"]["lower_conf"],
        data_1["J"]["upper_conf"],
        color="red",
        alpha=0.2,
    )
    ax[0].plot(x, data_1["V"]["mean"], color="red", linestyle="--", label="α=0.1: V")
    ax[0].fill_between(
        x,
        data_1["V"]["lower_conf"],
        data_1["V"]["upper_conf"],
        color="red",
        alpha=0.2,
    )
    ax[1].plot(x, data_1["H"]["mean"], color="red", linestyle="--", label="α=0.1: H")
    ax[1].fill_between(
        x,
        data_1["H"]["lower_conf"],
        data_1["H"]["upper_conf"],
        color="red",
        alpha=0.2,
    )

    # Second dataset J, V & H
    ax[0].plot(
        x, data_2["J"]["mean"], color="blue", linestyle="-", label="α=0.05: J"
    )
    ax[0].fill_between(
        x,
        data_2["J"]["lower_conf"],
        data_2["J"]["upper_conf"],
        color="blue",
        alpha=0.2,
    )
    ax[0].plot(
        x, data_2["V"]["mean"], color="blue", linestyle="--", label="α=0.05: V"
    )
    ax[0].fill_between(
        x,
        data_2["V"]["lower_conf"],
        data_2["V"]["upper_conf"],
        color="blue",
        alpha=0.2,
    )
    ax[1].plot(
        x, data_2["H"]["mean"], color="blue", linestyle="--", label="α=0.05: H"
    )
    ax[1].fill_between(
        x,
        data_2["H"]["lower_conf"],
        data_2["H"]["upper_conf"],
        color="blue",
        alpha=0.2,
    )

    # Third dataset J, V & H
    ax[0].plot(
        x, data_3["J"]["mean"], color="green", linestyle="-", label="α=0.5: J"
    )
    ax[0].fill_between(
        x,
        data_3["J"]["lower_conf"],
        data_3["J"]["upper_conf"],
        color="green",
        alpha=0.2,
    )
    ax[0].plot(
        x, data_3["V"]["mean"], color="green", linestyle="--", label="α=0.5: V"
    )
    ax[0].fill_between(
        x,
        data_3["V"]["lower_conf"],
        data_3["V"]["upper_conf"],
        color="green",
        alpha=0.2,
    )
    ax[1].plot(
        x, data_3["H"]["mean"], color="green", linestyle="--", label="α=0.5: H"
    )
    ax[1].fill_between(
        x,
        data_3["H"]["lower_conf"],
        data_3["H"]["upper_conf"],
        color="green",
        alpha=0.2,
    )

    # # 4 dataset J, V & H
    # ax[0].plot(
    #     x, data_4["J"]["mean"], color="yellow", linestyle="-", label="α=0.5: J"
    # )
    # ax[0].fill_between(
    #     x,
    #     data_4["J"]["lower_conf"],
    #     data_4["J"]["upper_conf"],
    #     color="green",
    #     alpha=0.2,
    # )
    # ax[0].plot(
    #     x, data_4["V"]["mean"], color="yellow", linestyle="--", label="α=0.5: V"
    # )
    # ax[0].fill_between(
    #     x,
    #     data_4["V"]["lower_conf"],
    #     data_4["V"]["upper_conf"],
    #     color="green",
    #     alpha=0.2,
    # )
    # ax[1].plot(
    #     x, data_4["H"]["mean"], color="yellow", linestyle="--", label="α=0.5: H"
    # )
    # ax[1].fill_between(
    #     x,
    #     data_4["H"]["lower_conf"],
    #     data_4["H"]["upper_conf"],
    #     color="green",
    #     alpha=0.2,
    # )

    # # 5 dataset J, V & H
    # ax[0].plot(
    #     x, data_5["J"]["mean"], color="pink", linestyle="-", label="α=0.00001: J"
    # )
    # ax[0].fill_between(
    #     x,
    #     data_5["J"]["lower_conf"],
    #     data_5["J"]["upper_conf"],
    #     color="green",
    #     alpha=0.2,
    # )
    # ax[0].plot(
    #     x, data_5["V"]["mean"], color="pink", linestyle="--", label="α=0.00001: V"
    # )
    # ax[0].fill_between(
    #     x,
    #     data_5["V"]["lower_conf"],
    #     data_5["V"]["upper_conf"],
    #     color="green",
    #     alpha=0.2,
    # )
    # ax[1].plot(
    #     x, data_5["H"]["mean"], color="pink", linestyle="--", label="α=0.00001: H"
    # )
    # ax[1].fill_between(
    #     x,
    #     data_5["H"]["lower_conf"],
    #     data_5["H"]["upper_conf"],
    #     color="green",
    #     alpha=0.2,
    # )

    # Plot info & formatting
    ax[1].set_xlabel("Epoch")
    ax[0].set_ylabel("Value")
    ax[1].set_ylabel("Entropy")
    fig.suptitle("ppo tuning")
    ax[0].legend(prop={"size": 8})
    ax[1].legend(prop={"size": 8})
    fig.set_size_inches(8, 6)

    # Save plot
    plot_dir = PATH+ "/plots"
    fn = plot_dir + "/tuning_ppo.jpg"
    plt.savefig(fn)


if __name__ == "__main__":
    main()
