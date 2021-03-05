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
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_instance_parameters(global_path, dataset_name):

        instance_name_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/instance_name_file.p", "rb" )
        instance_name_list = pickle.load(instance_name_file)
        instance_name_file.close()

        nb_nodes_list = []
        nb_arcs_list = []
        nb_commodities_list = []

        for instance_name in instance_name_list:
            instance_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/" + instance_name + ".p", "rb" )
            graph, commodity_list = pickle.load(instance_file) # read an instance
            instance_file.close()

            nb_nodes_list.append(len(graph))
            nb_arcs_list.append(sum(len(neighbor_dict) for neighbor_dict in graph))
            nb_commodities_list.append(len(commodity_list))

        # print the parameters for each instance of the dataset
        nb_commodities_list = sorted(nb_commodities_list)
        print(nb_commodities_list)
        print(nb_nodes_list)
        print(nb_arcs_list)


def plot_results(abscisse, results, algorithm_list, colors, formating, title, x_log=True, y_log=True, interval=True, x_label="Nb_nodes", y_label="Performance", legend_position="upper left"):
    figure = plt.figure()
    plt.rcParams.update({'font.size': 13})
    if x_log : plt.xscale("log")
    if y_log : plt.yscale("log")
    ax = figure.gca()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks([30, 50, 100, 200, 400], [30, 50, 100, 200, 400])

    for algorithm_name in algorithm_list:
        if interval:
            plt.plot(abscisse, results[algorithm_name][0], formating[algorithm_name], label=algorithm_label, color=colors[algorithm_name]) # print the main curve
            plt.fill_between(abscisse, results[algorithm_name][1], results[algorithm_name][2], alpha=0.25, facecolor=colors[algorithm_name], edgecolor=colors[algorithm_name]) # print the intervals around the main curve
            ax.legend(loc=legend_position, framealpha=0.3)

        else:
            plt.plot(abscisse, results[algorithm_name], label=algorithm_name, color=colors[algorithm_name])
            ax.legend(loc=legend_position, framealpha=0.3)

    return figure


def plot_dataset(global_path, dataset_name, algorithm_list=None, x_label="Nb nodes", legend_position="upper left"):
    # This function reads the results of a dataset, aggregates the results of instances with the same parameters and call the plotting function

    result_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/result_file.p", "rb" )
    # result_file = open(global_path + "/MCNF_solver/instance_files_dynamic/" + dataset_name + "/result_file.p", "rb" )
    result_dict = pickle.load(result_file)
    result_file.close()

    if algorithm_list is None:
        algorithm_list = list(result_dict.keys())

    # Color for each algorithm
    colors = {"SRR" : '#1f77b4', "RR" : '#ff7f0e', "SRR congestion" : '#1f77b4',
                "RR congestion" : '#ff7f0e', 'SA' : '#2ca02c', "MILP solver" : '#d62728',
                "CSRR" : '#9467bd', "SRR unsorted" : "#000000", "RR sorted" : "#eeee00",
                 'SA 2' : '#2ca02c', 'VNS' : '#d62728', 'VNS 2' : '#d62728', 'Ant colony' : '#000000'}

     # Line style for each algorithm
    formating = {"SRR" : '-s', "RR" : '-^', "SRR congestion" : '-_',
                "RR congestion" : '-', 'SA' : '-o', "MILP solver" : ':',
                "CSRR" : '-', "SRR unsorted" : "--", "RR sorted" : "-d",
                'SA 2' : '--', 'VNS' : '-o', 'VNS 2' : '--', 'Ant colony' : '-o'}

    # Color for each algorithm
    # colors = {"SRR arc node" : '#1f77b4', "SRR arc path" : '#ff7f0e', "SRR restricted" : '#ff7f0e',
    #             "B&B restricted medium" : '#2ca02c', 'Partial B&B restricted' : '#2ca02c', "SRR path-combination" : '#d62728',
    #             "SRR path-combination restricted" : '#d62728', 'SRR arc path no penalization' : '#ff7f0e', 'B&B restricted short' : '#2ca02c',
    #             'B&B restricted long' : '#2ca02c', 'SRR path-combination no penalization' : '#d62728', 'SRR path-combination timestep' : '#9467bd',
    #             'SRR arc node no penalization' : '#1f77b4', 'SRR path-combination commodity' : '#eeee00'}

    # Line style for each algorithm
    # formating = {"SRR arc node" : '-', "SRR arc path" : '-', "SRR restricted" : '-s',
    #             "B&B restricted medium" : '-', 'Partial B&B restricted' : '-o', "SRR path-combination" : '-',
    #             "SRR path-combination restricted" : '-s', 'SRR arc path no penalization' : '-o', 'B&B restricted short' : '-s',
    #             'B&B restricted long' : '-o', 'SRR path-combination no penalization' : '-o', 'SRR path-combination timestep' : '-',
    #              'SRR arc node no penalization' : '-o', 'SRR path-combination commodity' : '-'}

    results_performance = {algorithm_name : ([], [], []) for algorithm_name in result_dict}
    results_compututing_time = {algorithm_name : ([], [], []) for algorithm_name in result_dict}
    results_total_overload = {algorithm_name : ([], [], []) for algorithm_name in result_dict}

    for algorithm_name in algorithm_list:
        temp_dict = {}
        for instance_name in result_dict[algorithm_name]:
            size = int(instance_name.split('_')[1]) # use for graph_scaling_dataset
            # size = int(instance_name.split('_')[2]) # use for graph_scaling_dataset_random and commodity_scaling_dataset
            if size not in temp_dict:
                temp_dict[size] = []
            temp_dict[size].extend(result_dict[algorithm_name][instance_name])

        for size in sorted(list(temp_dict.keys())):
            # result_list = [res if res[0] is not None else (0, 0, 10, 10**5, 10**4) for res in temp_dict[size]]
            result_list = [res if res[0] is not None else (2, 0, 1200) for res in temp_dict[size]] #If an algorithm could not finish in time we give it a bad result

            performance_list, total_overload, computing_time_list = zip(*result_list)
            # _, _, performance_list, total_overload_list, computing_time_list = zip(*result_list)
            # total_overload_list = [x + 1 for x in total_overload_list]
            # performance_list = [x - 1 for x in performance_list]

            # Aggregation of the performance : mean and bound of the confidence interval
            performance_mean, performance_low, performance_up = mean_confidence_interval(list(performance_list))
            results_performance[algorithm_name][0].append(performance_mean)
            results_performance[algorithm_name][1].append(max(10**-6, performance_low)) # prevent bad plotting in log scales
            results_performance[algorithm_name][2].append(performance_up)

            # Aggregation of the computing time : mean and bound of the confidence interval
            computing_time_mean, computing_time_low, computing_time_up = mean_confidence_interval(list(computing_time_list))
            results_compututing_time[algorithm_name][0].append(computing_time_mean)
            results_compututing_time[algorithm_name][1].append(computing_time_low)
            results_compututing_time[algorithm_name][2].append(computing_time_up)

            # plt.plot([size]*len(total_overload_list), total_overload_list, '+', color=colors[algorithm_name])

            # total_overload_mean, total_overload_low, total_overload_up = mean_confidence_interval(list(total_overload_list))
            # results_total_overload[algorithm_name][0].append(total_overload_mean)
            # results_total_overload[algorithm_name][1].append(max(1, total_overload_low))
            # results_total_overload[algorithm_name][2].append(total_overload_up)

    # abscisse = [182.23, 362.88, 685.2, 1038.48, 1615.56, 2462.05, 3512.71, 5048.89, 8138.71, 11644.12]
    # abscisse = [63, 125.0, 234.0, 350.2, 540.3, 800.9, 1200.5, 1730.7, 2750.1, 3900.5]
    abscisse = list(temp_dict.keys())

    #Call to the plotting function for the differents metrics (performance, computing time, ...)
    performance_figure = plot_results(abscisse, results_performance, algorithm_list, colors, formating, "Performance vs number of nodes", x_label=x_label, y_label="Performance", legend_position=legend_position)
    computing_time_figure = plot_results(abscisse, results_compututing_time, algorithm_list, colors, formating, "Computing time vs number of nodes", x_label=x_label, y_label="Computing time", legend_position=legend_position)
    # total_overload_figure = plot_results(abscisse, results_total_overload, algorithm_list, colors, formating, "Total overload vs number of nodes", x_label=x_label, y_label="Total overload")
    plt.show()


if __name__ == "__main__":
    global_path = "/home/francois/Desktop"

    # dataset_name = "graph_scaling_dataset"
    # dataset_name = "graph_scaling_dataset_small_commodities"
    # dataset_name = "graph_scaling_dataset_random"
    dataset_name = "commodity_scaling_dataset"
    # dataset_name = "small_instance_dataset"

    # dataset_name = "graph_scaling_dataset_easy"
    # dataset_name = "graph_scaling_dataset_hard"
    # dataset_name = "graph_scaling_dataset_random"
    # dataset_name = "commodity_scaling_dataset"

    plot_instance_parameters(global_path, dataset_name)

    algorithm_list = []
    # algorithm_list.append("SRR arc node")
    # algorithm_list.append("SRR arc path")
    # algorithm_list.append("SRR arc node no penalization")
    # algorithm_list.append("SRR arc path no penalization")
    # algorithm_list.append("SRR restricted")
    # algorithm_list.append("B&B restricted short")
    # algorithm_list.append("B&B restricted medium")
    # algorithm_list.append("B&B restricted long")
    # algorithm_list.append("SRR path-combination")
    # algorithm_list.append("SRR path-combination no penalization")
    # algorithm_list.append("SRR path-combination timestep")
    # algorithm_list.append("SRR path-combination commodity")
    # algorithm_list.append("SRR path-combination restricted")

    algorithm_list.append("SRR")
    # algorithm_list.append("SRR unsorted")
    # algorithm_list.append("SRR congestion")
    algorithm_list.append("RR")
    # algorithm_list.append("RR sorted")
    # algorithm_list.append("RR congestion")
    algorithm_list.append("CSRR")
    algorithm_list.append("SA")
    # algorithm_list.append("SA 2")
    # algorithm_list.append("VNS")
    # algorithm_list.append("VNS 2")
    # algorithm_list.append("Ant colony")
    # algorithm_list.append("MILP solver")

    plot_dataset(global_path, dataset_name, algorithm_list, x_label="Nb nodes", legend_position="upper left")
