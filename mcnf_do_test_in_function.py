import random
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats
import pickle

from instance_mcnf import generate_instance, mutate_instance
from mcnf import *
from recuit_simule import recuit_unsplittable_flows
from read_telesat_data import read_data
from mcnf_dynamic import is_correct_path
from recuit_simule_LP import recuit_LP
from recuit_simule_LP2 import recuit_LP2
from VNS_masri import VNS_masri
from ant_colony import ant_colony_optimiser


def f():

    # Here you choose the setting of the instances and of the solvers

    # Size of the graph
    size_list = [15]*10
    # size_list = [3, 4, 5, 6, 7, 9, 10, 12, 13, 15]
    # size_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
    # size_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
    # size_list = [12, 13, 14, 15, 17, 20, 23, 26, 30, 35, 40]
    size_list = np.array(size_list)
    # size_list = size_list**2

    # Capacity of the arcs of the graph
    # capacity_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    capacity_list = [10000] * len(size_list)

    # Threshold of actualisation of the heuristic
    actulisation_threshold_list = None
    # actulisation_threshold_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Upper bound on the size of the commodities
    # max_demand_list = [10, 20 , 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    max_demand_list = [1500] * len(size_list)
    # max_demand_list = [capa / 5 for capa in capacity_list]
    # max_demand_list = [int(np.sqrt(capa)) for capa in capacity_list]

    test_list = []
    for size, capacity in zip(size_list, capacity_list):
        test_list += [("grid", (size, size, size, 2*size, capacity, capacity))]
    # for size, capacity in zip(size_list, capacity_list):
        # test_list += [("random_connected", (size, 5/size, 0.1, capacity))]

    # Choice of the tested algorithms
    tested_algorithms = []
    # tested_algorithms.append("heuristic")
    tested_algorithms.append("heuristic_arc_path")
    # tested_algorithms.append("heuristic_congestion")
    # tested_algorithms.append("approximation")
    # tested_algorithms.append("approximation_congestion")
    # tested_algorithms.append("heuristic_unsorted")
    # tested_algorithms.append("new_approx")
    # tested_algorithms.append("recuit_simule")
    # tested_algorithms.append("recuit_simule_LP")
    # tested_algorithms.append("MILP_solver")
    # tested_algorithms.append("VNS_masri")
    # tested_algorithms.append("ant_colony")

    results_dict = {algorithm_name : ([],[]) for algorithm_name in tested_algorithms}

    i = -1
    nb_commodity_list = []
    nb_node_list = []
    # graph, commodity_list, initial_solution = generate_instance(*test_list[0], max_demand=max_demand_list[0])

    for graph_type, graph_generator_inputs in test_list:
        i += 1
        print("##############################  ", i,"/",len(test_list))

        # Instance generation
        graph, commodity_list, initial_solution = generate_instance(graph_type, graph_generator_inputs, max_demand=max_demand_list[i])
        # instance_list = read_data(1)
        # graph, commodity_list = instance_list[-1]
        # instance_file = open("/home/pc-francois/Bureau/MCNF_solver/instance_files/commodity_scaling_dataset/grid_240_1000_31_0.p", 'rb')
        # graph, commodity_list = pickle.load(instance_file)
        # instance_file.close()

        total_demand = sum([c[2] for c in commodity_list])
        nb_nodes = len(graph)
        nb_commodities = len(commodity_list)
        print("total_demand is : ", total_demand)
        print("nb_commodities = ", nb_commodities)
        nb_commodity_list.append(len(commodity_list))
        nb_node_list.append(nb_nodes)

        #Setting default Threshold for the heuristic
        if actulisation_threshold_list is None:
            actulisation_threshold = len(commodity_list)/100
            # actulisation_threshold = 10
        else:
            actulisation_threshold = actulisation_threshold_list[i]


        # Applying the algorithm present in tested_algorithms
        for algorithm_name in tested_algorithms:
            print("Running {}".format(algorithm_name))
            temp = time.time()

            if algorithm_name == "heuristic" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, verbose=1)
            if algorithm_name == "heuristic_arc_path" : a = randomized_rounding_heuristic_arc_path(graph, commodity_list, actulisation_threshold=actulisation_threshold, verbose=1)
            if algorithm_name == "heuristic_congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, linear_objectif="congestion", verbose=1)
            if algorithm_name == "heuristic_unsorted" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, sorted_commodities=False)
            if algorithm_name == "new_approx" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, proof_constaint=True)
            if algorithm_name == "approximation" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False)
            if algorithm_name == "approximation_congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, linear_objectif="congestion")
            if algorithm_name == "recuit_simule" : a = recuit_unsplittable_flows(graph, commodity_list, nb_iterations= len(commodity_list)*200 + 3000, verbose=1)
            if algorithm_name == "recuit_simule_LP" : a = recuit_LP2(graph, commodity_list, nb_iterations= 200000)
            if algorithm_name == "MILP_solver" : a = gurobi_unsplittable_flows(graph, commodity_list, verbose=1, time_limit=1000)
            if algorithm_name == "VNS_masri" : a = VNS_masri(graph, commodity_list)
            if algorithm_name == "ant_colony" : a = ant_colony_optimiser(graph, commodity_list, 100)

            commodity_path_list, total_overload = a
            computing_time = time.time() - temp
            results_dict[algorithm_name][0].append(total_overload / total_demand)
            results_dict[algorithm_name][1].append(computing_time)
            print("Performance = ", total_overload / total_demand)
            print("Computing_time = ", computing_time)

            use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(len(graph))]
            for commodity_index, path in enumerate(commodity_path_list):
                update_graph_capacity(use_graph, path, -commodity_list[commodity_index][2])

            overload_graph = [{neighbor : max(0, use_graph[node][neighbor] - graph[node][neighbor]) for neighbor in graph[node]} for node in range(len(graph))]
            overload = sum([sum(dct.values()) for dct in overload_graph])
            print("Overload = ", overload)


            # use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]
            # for commodity_index, path in enumerate(commodity_path_list):
            #     update_graph_capacity(use_graph, path, -commodity_list[commodity_index][2])
            #
            # served_demand = total_demand
            # unserved_users = 0
            # demand_list = [c[2] for c in commodity_list]
            # sorted_commodity_indices = sorted(list(range(nb_commodities)), key=lambda x : demand_list[x])
            # for commodity_index in sorted_commodity_indices:
            #     path = commodity_path_list[commodity_index]
            #
            #     for node_index in range(len(path)-1):
            #         node, neighbor = path[node_index], path[node_index+1]
            #         if use_graph[node][neighbor] > graph[node][neighbor]:
            #             served_demand -= commodity_list[commodity_index][2]
            #             update_graph_capacity(use_graph, path, commodity_list[commodity_index][2])
            #             unserved_users += 1
            #             break
            #
            # print("xxxxxxxxx", unserved_users, total_demand - served_demand)

            # overload_graph = [{neighbor : max(0, use_graph[node][neighbor] - graph[node][neighbor]) for neighbor in graph[node]} for node in range(len(graph))]
            # total_overload = sum([sum(dct.values()) for dct in overload_graph])



    # Curves drawing

    # abscisse = nb_commodity_list
    # abscisse = nb_node_list
    # abscisse = actulisation_threshold_list
    abscisse = size_list

    for algorithm_name in tested_algorithms:
        res = np.array(results_dict[algorithm_name][0])
        mean = sum(res)/len(res)
        print("Mean "+ algorithm_name+ " = ", mean)
        print("Standard deviation "+ algorithm_name+ " = ", np.sqrt(sum((res-mean)**2)/len(res)))

    colors = {"heuristic" : '#1f77b4', "approximation" : '#ff7f0e', "heuristic_congestion" : '#1f77b4',
                "approximation_congestion" : '#ff7f0e', "recuit_simule" : '#2ca02c', "MILP_solver" : '#d62728',
                "new_approx" : '#9467bd', "iterated_heuristic_unsorted" : "#000000"}

    fig = plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    ax = fig.gca()
    for algorithm_name in tested_algorithms:
        plt.plot(abscisse, results_dict[algorithm_name][0], label=algorithm_name+"_results", color=colors[algorithm_name])
    ax.legend()
    plt.show()

    fig = plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    ax = fig.gca()
    for algorithm_name in tested_algorithms:
        plt.plot(abscisse, results_dict[algorithm_name][1], label=algorithm_name+"_c_time", color=colors[algorithm_name])
    ax.legend()
    plt.show()

if __name__ == "__main__":
    f()
