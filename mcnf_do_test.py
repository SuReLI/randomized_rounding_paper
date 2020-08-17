import random
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats
import pickle

from instance_mcnf import generate_instance, mutate_instance
from mcnf import *
from simulated_annealing import annealing_unsplittable_flows


# In this file, it is possible create custom instances and lauch several algorithms on them and plot the results

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
max_demand_list = [1500] * len(size_list)
# max_demand_list = [int(np.sqrt(capa)) for capa in capacity_list]

test_list = []
for size, capacity in zip(size_list, capacity_list):
    test_list += [("grid", (size, size, size, 2*size, capacity, capacity))]
# for size, capacity in zip(size_list, capacity_list):
    # test_list += [("random_connected", (size, 5/size, 0.1, capacity))]

# Choice of the tested algorithms
tested_algorithms = []
tested_algorithms.append("RR")
# tested_algorithms.append("RR congestion")
tested_algorithms.append("SRR")
# tested_algorithms.append("SRR congestion")
# tested_algorithms.append("SRR unsorted")
# tested_algorithms.append("CSRR")
tested_algorithms.append("Simulated annaeling")
# tested_algorithms.append("MILP solver")

results_dict = {algorithm_name : ([],[]) for algorithm_name in tested_algorithms}

i = -1
nb_commodity_list = []
nb_node_list = []

for graph_type, graph_generator_inputs in test_list:
    i += 1
    print("##############################  ", i,"/",len(test_list))

    # Instance generation
    graph, commodity_list, initial_solution = generate_instance(graph_type, graph_generator_inputs, max_demand=max_demand_list[i])


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

        if algorithm_name == "RR" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False)
        if algorithm_name == "RR congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, linear_objectif="congestion")
        if algorithm_name == "SRR" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold)
        if algorithm_name == "SRR unsorted" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, sorted_commodities=False)
        if algorithm_name == "SRR congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, linear_objectif="congestion")
        if algorithm_name == "CSRR" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, proof_constaint=True)
        if algorithm_name == "Simulated annaeling" : a = annealing_unsplittable_flows(graph, commodity_list, nb_iterations= len(commodity_list)*30)
        if algorithm_name == "MILP solver" : a = gurobi_unsplittable_flows(graph, commodity_list, time_limit=1000)

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
