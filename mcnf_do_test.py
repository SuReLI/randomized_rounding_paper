import random
import numpy as np
import time

from instance_mcnf import generate_instance
from mcnf import *
from simulated_annealing import simulated_annealing_unsplittable_flows
from VNS_masri import VNS_masri
from ant_colony import ant_colony_optimiser

# Here you choose the setting of the instances and of the solvers

# Size of the graph
# size_list = [10]*10
# size_list = [3, 4, 5, 6, 7, 9, 10, 12, 13, 15]
size_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
# size_list = [30, 50, 70, 100, 130, 160, 200, 250, 300, 400]
size_list = np.array(size_list)
# size_list = size_list**2

# Capacity of the arcs of the graph
capacity_list = [10000] * len(size_list)
# capacity_list = [3] * len(size_list)
# capacity_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Threshold of actualisation of the heuristic
actulisation_threshold_list = None
# actulisation_threshold_list = 2 ** (np.arange(10) + 4)

# Upper bound on the size of the commodities
max_demand_list = [1500] * len(size_list)
# max_demand_list = [2] * len(size_list)
# max_demand_list = [int(np.sqrt(capa)) for capa in capacity_list]

test_list = []
for size, capacity, max_demand in zip(size_list, capacity_list, max_demand_list):
    test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]
    # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : True})]
    # test_list += [("random_connected", (size, 5/size, int(size * 0.1), capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]

# Choice of the tested algorithms
tested_algorithms = []

tested_algorithms.append("RR")
# tested_algorithms.append("RR sorted")
# tested_algorithms.append("RR congestion")
tested_algorithms.append("SRR")
# tested_algorithms.append("SRR unsorted")
# tested_algorithms.append("SRR congestion")
tested_algorithms.append("CSRR")
tested_algorithms.append("SA")
# tested_algorithms.append("SA 2")
# tested_algorithms.append("MILP solver")
# tested_algorithms.append("VNS")
# tested_algorithms.append("VNS 2")
# tested_algorithms.append("Ant colony")

results_dict = {algorithm_name : ([],[]) for algorithm_name in tested_algorithms}

i = -1
nb_commodity_list = []
nb_node_list = []

for graph_type, graph_generator_inputs, demand_generator_inputs in test_list:
    i += 1
    print("##############################  ", i,"/",len(test_list))

    # Instance generation
    graph, commodity_list, initial_solution, origin_list = generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs)

    total_demand = sum([c[2] for c in commodity_list])
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    print("total_demand is : ", total_demand)
    print("nb_commodities = ", nb_commodities)
    nb_commodity_list.append(len(commodity_list))
    nb_node_list.append(nb_nodes)

    #Setting default Threshold for the heuristic
    if actulisation_threshold_list is None:
        actualisation_threshold = None
    else:
        actualisation_threshold = actulisation_threshold_list[i]

    # Applying the algorithm present in tested_algorithms
    for algorithm_name in tested_algorithms:
        print("Running {}".format(algorithm_name))
        temp = time.time()


        if algorithm_name == "RR" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, sorted_commodities=False)
        if algorithm_name == "RR sorted" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False)
        if algorithm_name == "RR congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, linear_objectif="congestion")
        if algorithm_name == "SRR" : a = randomized_rounding_heuristic(graph, commodity_list)
        if algorithm_name == "SRR unsorted" : a = randomized_rounding_heuristic(graph, commodity_list, sorted_commodities=False)
        if algorithm_name == "SRR congestion" : a = randomized_rounding_heuristic(graph, commodity_list, linear_objectif="congestion")
        if algorithm_name == "CSRR" : a = randomized_rounding_heuristic(graph, commodity_list, proof_constaint=True)
        if algorithm_name == "SA" : a = simulated_annealing_unsplittable_flows(graph, commodity_list, nb_iterations= int(len(commodity_list)**1.5 * 2))
        if algorithm_name == "SA 2" : a = simulated_annealing_unsplittable_flows(graph, commodity_list, nb_iterations= int(len(commodity_list)**1.5 * 6))
        if algorithm_name == "MILP solver" : a = gurobi_unsplittable_flows(graph, commodity_list, time_limit=1000)
        if algorithm_name == "VNS" : a = VNS_masri(graph, commodity_list, nb_iterations= 100)
        if algorithm_name == "VNS 2" : a = VNS_masri(graph, commodity_list, nb_iterations= int(len(commodity_list)**1.5), amelioration=True)
        if algorithm_name == "Ant colony" : a = ant_colony_optimiser(graph, commodity_list, nb_iterations=50)

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
        print()
