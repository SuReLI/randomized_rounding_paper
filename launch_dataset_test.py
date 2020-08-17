import random
import numpy as np
import time
import pickle
import multiprocessing as mp

from instance_mcnf import generate_instance, mutate_instance
from mcnf import *
from simulated_annealing import annealing_unsplittable_flows

def launch_dataset(global_path, dataset_name, tested_algorithms, nb_repetitions):
    # Launches all the algorithms to test on the instance present in the dataset directory
    # The number of time algorithms are lauched is decided with nb_repetitions

    # Open the file containing the name of the instances
    instance_name_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/instance_name_file.p", "rb" )
    instance_name_list = pickle.load(instance_name_file)
    instance_name_file.close()

    log_file = open(global_path + "/MCNF_solver/log_file.txt", 'w')
    log_file.write("Start\n")
    log_file.close()

    pool = mp.Pool(3)

    result_dict = {algorithm_name : {instance_name : [] for instance_name in instance_name_list} for algorithm_name in tested_algorithms}

    for repetition_index in range(nb_repetitions):
        for instance_index, instance_name in enumerate(instance_name_list):
            for algorithm_index, algorithm_name in enumerate(tested_algorithms):
                print_string = "repetition : {0}/{1}, instance : {2}/{3}, algorithm : {4}".format(repetition_index, nb_repetitions, instance_index, len(instance_name_list), algorithm_name)

                instance_file_path = global_path + "/MCNF_solver/instance_files/" + dataset_name + "/" + instance_name + ".p"

                # Launch an algorithm on an instance
                result_dict[algorithm_name][instance_name].append(pool.apply_async(launch_solver_on_instance, args=(instance_file_path, algorithm_name, print_string)))

    # Collect the data from the multiprocessing results
    for repetition_index in range(nb_repetitions):
        for instance_index, instance_name in enumerate(instance_name_list):
            for algorithm_index, algorithm_name in enumerate(tested_algorithms):
                result_dict[algorithm_name][instance_name][repetition_index] = result_dict[algorithm_name][instance_name][repetition_index].get()

    # Write the results in a file
    result_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/result_file.p", "wb" )
    pickle.dump(result_dict, result_file)
    result_file.close()


def launch_solver_on_instance(instance_file_path, algorithm_name, print_string):
    # Lauch the algorithm named algortihm_name on the instance store in the file at instance_file_path

    print(print_string)

    # Read the instance in the instance file
    instance_file = open(instance_file_path, "rb" )
    graph, commodity_list = pickle.load(instance_file)
    instance_file.close()

    total_demand = sum([commodity[2] for commodity in commodity_list])

    actulisation_threshold = len(commodity_list)/100

    temp = time.time()

    # Launch the chosen algorithm
    if algorithm_name == "RR" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False)
    if algorithm_name == "RR congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, linear_objectif="congestion")
    if algorithm_name == "SRR" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold)
    if algorithm_name == "SRR unsorted" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, sorted_commodities=False)
    if algorithm_name == "SRR congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, linear_objectif="congestion")
    if algorithm_name == "CSRR" : a = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=actulisation_threshold, proof_constaint=True)
    if algorithm_name == "Simulated annaeling" : a = annealing_unsplittable_flows(graph, commodity_list, nb_iterations= len(commodity_list)*30)
    if algorithm_name == "MILP solver" : a = gurobi_unsplittable_flows(graph, commodity_list, time_limit=1000)

    computing_time = time.time() - temp

    commodity_path_list, total_overload = a

    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(len(graph))]
    for commodity_index, path in enumerate(commodity_path_list):
        update_graph_capacity(use_graph, path, -commodity_list[commodity_index][2])

    overload_graph = [{neighbor : max(0, use_graph[node][neighbor] - graph[node][neighbor]) for neighbor in graph[node]} for node in range(len(graph))]
    total_overload = sum([sum(dct.values()) for dct in overload_graph])

    performance = total_overload / total_demand

    congestion_graph = [{neighbor : use_graph[node][neighbor] / graph[node][neighbor] for neighbor in graph[node]} for node in range(len(graph))]
    congestion = max([max(dct.values()) for dct in overload_graph])

    # log_file = open("/home/disc/f.lamothe/MCNF_solver/log_file.txt", 'a')
    # log_file.write("Finished : " + instance_file_path + ", " + print_string + "\n")
    # log_file.close()

    return performance, total_overload, congestion, computing_time


if __name__ == "__main__":
    # Set the path to the global directory
    # global_path = "/home/francois/Desktop/"
    global_path = None
    assert global_path, "Complete global_path with the path to the main directory"

    # Set the number of repetition
    nb_repetitions = 1

    settings_list = []
    settings_list.append((global_path, "graph_scaling_dataset", ["RR", "SRR", 'SRR unsorted', "CSRR", "Simulated annaeling"], nb_repetitions))
    # settings_list.append((global_path, "graph_scaling_dataset_random", ["RR", "SRR", 'SRR unsorted', "CSRR", "Simulated annaeling"], nb_repetitions))
    # settings_list.append((global_path, "graph_scaling_dataset_small_commodities", ["RR", "SRR", "Simulated annaeling"], nb_repetitions))
    # settings_list.append((global_path, "commodity_scaling_dataset", ["RR", "SRR", "Simulated annaeling"], nb_repetitions))
    # settings_list.append((global_path, "small_instance_dataset", ["RR", "SRR", "MILP solver"], 1))

    for settings in settings_list:
        launch_dataset(*settings)
