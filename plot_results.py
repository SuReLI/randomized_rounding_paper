import random
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    """
    Compute the mean and confidence interval of the the input data array-like.
    :param data: (array-like)
    :param confidence: probability of the mean to lie in the interval
    :return: (tuple) mean, interval upper-endpoint, interval lower-endpoint
    """

    a = 1.0 * np.array(data)
    n = len(a)
    # m, se = np.mean(a), scipy.stats.sem(a)
    # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    m = sum(a)/len(a)
    se = np.sqrt(sum((a - m)**2)/n)
    h = 1.96 * se / np.sqrt(n)
    return m, m-h, m+h


def plot_results(abscisse, results, algorithm_list, colors, formating, x_log=True, y_log=True, interval=False):
    # Plot the results constained in results with confidence intervalles

    figure = plt.figure()
    plt.rcParams.update({'font.size': 13})
    if x_log : plt.xscale("log")
    if y_log : plt.yscale("log")
    ax = figure.gca()

    for algorithm_name in algorithm_list:
        algorithm_label = algorithm_name if algorithm_name != 'Simulated annaeling' else 'SA'

        if interval:
            plt.plot(abscisse, results[algorithm_name][0], formating[algorithm_name], label=algorithm_label, color=colors[algorithm_name])
            plt.fill_between(abscisse, results[algorithm_name][1], results[algorithm_name][2], alpha=0.25, facecolor=colors[algorithm_name], edgecolor=colors[algorithm_name])
            ax.legend(loc='lower left', framealpha=0.5)
            ax.set_xlabel("Nb nodes")
            # ax.set_xlabel("Nb commodites")
            ax.set_ylabel("Performance")
            # ax.set_ylabel("Computing time")

        else:
            plt.plot(abscisse, results[algorithm_name], label=algorithm_name, color=colors[algorithm_name])
            ax.legend(loc='lower right')
            ax.set_xlabel("Nb nodes")
            ax.set_ylabel("Performance")

    return figure

def plot_dataset(global_path, dataset_name, abscisse):
    # Plot the results for a dataset : results are stored in the result_file.p of the dataset directory

    # Open the result file
    result_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/result_file.p", "rb" )
    result_dict = pickle.load(result_file)
    result_file.close()

    algorithm_list = list(result_dict.keys())

    colors = {"SRR" : '#1f77b4', "RR" : '#ff7f0e', "SRR congestion" : '#1f77b4',
                "RR congestion" : '#ff7f0e', 'Simulated annaeling' : '#2ca02c', "MILP solver" : '#d62728',
                "CSRR" : '#9467bd', "SRR unsorted" : "#000000"}

    formating = {"SRR" : '-s', "RR" : '-^', "SRR congestion" : '-_',
                "RR congestion" : '-', 'Simulated annaeling' : '-o', "MILP solver" : ':',
                "CSRR" : '-', "SRR unsorted" : "--"}

    results_performance = {algorithm_name : ([], [], []) for algorithm_name in result_dict}
    results_compututing_time = {algorithm_name : ([], [], []) for algorithm_name in result_dict}

    # Extract the results and compute the mean and confidence intervals
    for algorithm_name in result_dict:
        temp_dict = {}
        for instance_name in result_dict[algorithm_name]:
            size = int(instance_name.split('_')[2])
            if size not in temp_dict:
                temp_dict[size] = []
            temp_dict[size].extend(result_dict[algorithm_name][instance_name])

        for size in sorted(list(temp_dict.keys())):
            performance_list, _, _, computing_time_list = zip(*temp_dict[size])
            performance_mean, performance_low, performance_up = mean_confidence_interval(list(performance_list))
            results_performance[algorithm_name][0].append(performance_mean)
            results_performance[algorithm_name][1].append(performance_low)
            results_performance[algorithm_name][2].append(performance_up)

            computing_time_mean, computing_time_low, computing_time_up = mean_confidence_interval(list(computing_time_list))
            results_compututing_time[algorithm_name][0].append(computing_time_mean)
            results_compututing_time[algorithm_name][1].append(computing_time_low)
            results_compututing_time[algorithm_name][2].append(computing_time_up)

    # Choose the abscisse of the plot
    if abscisse == "nb_nodes":
        abscisse = list(temp_dict.keys())

    # Plot the figures
    performance_figure = plot_results(abscisse, results_performance, algorithm_list, colors, formating, interval=True)
    computing_time_figure = plot_results(abscisse, results_compututing_time, algorithm_list, colors, formating, interval=True)
    plt.show()


if __name__ == "__main__":
    global_path = None
    assert global_path, "Complete global_path with the path to the main directory"
    dataset_name = "commodity_scaling_dataset"
    # abscisse = [422.4, 843.0, 1597.0, 2410.2, 3766.3, 5747.9, 8243.5, 11874.7, 18992.1, 26952.5]
    abscisse = "nb_nodes"
    plot_dataset(global_path, dataset_name, abscisse)
