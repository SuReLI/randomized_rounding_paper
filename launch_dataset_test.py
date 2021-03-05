import random
import time
import pickle
from multiprocessing import Process, Manager

from instance_mcnf import generate_instance, mutate_instance
from mcnf import *
from VNS_masri import VNS_masri
from ant_colony import ant_colony_optimiser
from simulated_annealing import simulated_annealing_unsplittable_flows

def launch_dataset(global_path, dataset_name, algorithm_list, nb_repetitions, nb_workers, duration_before_timeout):
    # Launches all the algorithms to test on the instance present in the dataset directory
    # The number of time algorithms are lauched is decided with nb_repetitions

    # Open the file containing the name of the instances
    instance_name_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/instance_name_file.p", "rb" )
    instance_name_list = pickle.load(instance_name_file)
    instance_name_file.close()

    log_file = open(global_path + "/MCNF_solver/log_file.txt", 'w')
    log_file.write("Start\n")
    log_file.close()

    manager = Manager()
    result_dict = {algorithm_name : {instance_name : [None]*nb_repetitions for instance_name in instance_name_list} for algorithm_name in algorithm_list}
    worker_list = [] # Will contain the process running in parallel. Each process runs one algorithm on one instance
    computation_list = [(repetition_index, instance_index, instance_name, algorithm_name) for repetition_index in range(nb_repetitions)
                                                                                            for instance_index, instance_name in enumerate(instance_name_list)
                                                                                                for algorithm_name in algorithm_list]

    while len(computation_list) + len(worker_list) > 0:

        remaining_worker_list = []
        for process, start_time, return_list, computation_info in worker_list:
            repetition_index, instance_index, instance_name, algorithm_name = computation_info

            if not process.is_alive():
                # If the process terminated with errors store the results
                result_dict[algorithm_name][instance_name][repetition_index] = return_list[0]

            elif time.time() > start_time + duration_before_timeout:
                # If the process used more than the maximum computing time, kill it and store an error result
                process.terminate()
                result_dict[algorithm_name][instance_name][repetition_index] = (None, None, None, None, duration_before_timeout)

            else:
                # Let the worker continue
                remaining_worker_list.append((process, start_time, return_list, computation_info))

        worker_list = remaining_worker_list

        if len(worker_list) < nb_workers and len(computation_list) > 0: # If all the workers are not working and there is still some experiments to launch
            computation_info = computation_list.pop(0)
            repetition_index, instance_index, instance_name, algorithm_name = computation_info

            print_string = "repetition : {0}/{1}, instance : {2}/{3}, algorithm : {4}".format(repetition_index, nb_repetitions, instance_index, len(instance_name_list), algorithm_name)
            instance_file_path = global_path + "/MCNF_solver/instance_files/" + dataset_name + "/" + instance_name + ".p"
            return_list = manager.list()

            # Create a worker that launchs an algorithm through the function launch_solver_on_instance
            process = Process(target=launch_solver_on_instance, args=(instance_file_path, algorithm_name, print_string, global_path, return_list))
            start_time = time.time()
            process.start()
            worker_list.append((process, start_time, return_list, computation_info))

    # Write the results in a file
    result_file = open(global_path + "/MCNF_solver/instance_files/" + dataset_name + "/result_file.p", "wb" )
    pickle.dump(result_dict, result_file)
    result_file.close()


def launch_solver_on_instance(instance_file_path, algorithm_name, print_string, global_path, return_list):
    # Launch the algorithm named algortihm_name on the instance stored in the file at instance_file_path

    print(print_string)

    # Read the instance in the instance file
    instance_file = open(instance_file_path, "rb" )
    graph, commodity_list = pickle.load(instance_file)
    instance_file.close()

    total_demand = sum([commodity[2] for commodity in commodity_list])
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)


    temp = time.time()

    # Launch the chosen algorithm
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

    computing_time = time.time() - temp

    commodity_path_list, total_overload = a
    performance = total_overload / total_demand

    log_file = open(global_path + "/MCNF_solver/log_file.txt", 'a')
    log_file.write("Finished : " + instance_file_path + ", " + print_string + "\n")
    log_file.close()

    return_list.append((performance, total_overload, computing_time))


if __name__ == "__main__":
    # Set the path to the global directory
    global_path = "/home/disc/f.lamothe"
    # global_path = "/home/francois/Desktop"
    # assert False, "Complete global_path with the path to the main directory"

    # Set the number of repetition
    nb_workers = 15
    duration_before_timeout = 2*60*60

    settings_list = []
    settings_list.append(("graph_scaling_dataset", ["RR", 'RR sorted', "SRR", 'SRR unsorted', "CSRR", "SA", "SA 2", "VNS", "VNS 2", "Ant colony"], 1))
    settings_list.append(("graph_scaling_dataset_small_commodities", ["RR", 'RR sorted', "SRR", 'SRR unsorted', "CSRR", "SA", "SA 2", "VNS", "VNS 2", "Ant colony"], 1))
    settings_list.append(("graph_scaling_dataset_random", ["RR", 'RR sorted', "SRR", 'SRR unsorted', "CSRR", "SA", "SA 2", "VNS", "VNS 2", "Ant colony"], 1))
    settings_list.append(("commodity_scaling_dataset", ["RR", 'RR sorted', "SRR", 'SRR unsorted', "CSRR", "SA", "SA 2", "VNS", "VNS 2", "Ant colony"], 1))
    settings_list.append(("small_instance_dataset", ["RR", "SRR", "MILP solver"], 1))

    for dataset_name, algorithm_list, nb_repetitions in settings_list:
        launch_dataset(global_path, dataset_name, algorithm_list, nb_repetitions, nb_workers, duration_before_timeout)
