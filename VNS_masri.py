import heapq as hp
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from k_shortest_path import k_shortest_path_algorithm, k_shortest_path_all_destination
from simulated_annealing import compute_all_distances

def VNS_masri(graph, commodity_list, nb_iterations, amelioration=False, verbose=0):
    # Setting hyper-parameters
    nb_modifications = 1
    nb_modification_max = 3
    nb_k_shortest_paths = 10

    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    total_demand = sum([c[2] for c in commodity_list])

    # Compute the k-shortest paths for each commodity and store them in possible_paths_per_commodity
    k_shortest_path_structure = {}
    possible_paths_per_commodity = []
    for commodity_index, commodity in enumerate(commodity_list):
        origin, destination, demand = commodity

        if origin not in k_shortest_path_structure:
            k_shortest_path_structure[origin] = k_shortest_path_all_destination(graph, origin, nb_k_shortest_paths)

        path_and_cost_list = k_shortest_path_structure[origin][destination]
        possible_paths_per_commodity.append([path for path, path_cost in path_and_cost_list])
    if verbose : print("possible_paths_per_commodity computed")

    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]
    all_distances = compute_all_distances(graph)

    # Create an initial solution and compute its solution value
    solution = []
    solution_value = 0
    for commodity_index, commodity in enumerate(commodity_list):
        path_index = np.random.choice(len(possible_paths_per_commodity[commodity_index]))
        path = possible_paths_per_commodity[commodity_index][path_index]
        solution.append(path)
        solution_value += update_fitness_and_use_graph(use_graph, graph, [], path, commodity[2])

    # Main loop
    for iter_index in range(nb_iterations):
        if iter_index % 1 == 0 and verbose:
            print(iter_index, solution_value, nb_modifications, end='     \r')

        # Make several modifications on the current solution and evaluate the new solution
        new_solution_value, modification_list = make_modifications(graph, commodity_list, solution, use_graph, possible_paths_per_commodity, nb_modifications, solution_value)

        # Keep the new solution if it has a smaller salution value
        if new_solution_value < solution_value:
            nb_modifications = 1
            solution_value = new_solution_value
            continue

        # Apply local search on the new solution
        for local_search_index in range((not amelioration) * nb_commodities // 2):
            modified_commodity_index = np.random.choice(nb_commodities)

            # Create a new path for a commodity
            new_path = create_new_path(graph, use_graph, commodity_list[modified_commodity_index], all_distances)
            old_path = solution[modified_commodity_index]
            solution[modified_commodity_index] = new_path
            new_solution_value += update_fitness_and_use_graph(use_graph, graph, old_path, new_path, commodity_list[modified_commodity_index][2])

            if new_solution_value < solution_value:
                nb_modifications = 1
                solution_value = new_solution_value
                break

            solution[modified_commodity_index] = old_path
            new_solution_value += update_fitness_and_use_graph(use_graph, graph, new_path, old_path, commodity_list[modified_commodity_index][2])

        else:
            # If the local search ends without finding an improving solution, return to the old solution and change the size of the neighborhood
            nb_modifications = min(nb_modification_max, nb_modifications + 1)
            for commodity_index, old_path, new_path in modification_list:
                solution[commodity_index] = old_path
                update_fitness_and_use_graph(use_graph, graph, new_path, old_path, commodity_list[commodity_index][2])

    return solution, solution_value


def make_modifications(graph, commodity_list, solution, use_graph, possible_paths_per_commodity, nb_modifications, solution_value):
    # Make several modifications on the current solution and evaluate the solution obtained
    new_solution_value = solution_value
    modification_list = []
    modified_commodity_list = np.random.choice(len(commodity_list), size=nb_modifications, replace=False)

    for commodity_index in modified_commodity_list:
        possible_paths = possible_paths_per_commodity[commodity_index]
        old_path = solution[commodity_index]
        new_path = possible_paths[random.randint(0, len(possible_paths)-1)]
        solution[commodity_index] = new_path
        new_solution_value += update_fitness_and_use_graph(use_graph, graph, old_path, new_path, commodity_list[commodity_index][2])
        modification_list.append((commodity_index, old_path, new_path))

    return new_solution_value, modification_list



def update_fitness_and_use_graph(use_graph, graph, old_path, new_path, commodity_demand):
    # This function makes the updates necessary to reflect the fact that a commodity uses new_path instead of old_path
    # To do so, it updates use_graph (total flow going through eahc arc) and computes in delta_fitness
    delta_fitness = 0

    for i in range(len(old_path) - 1):
        node1 = old_path[i]
        node2 = old_path[i+1]
        old_overload = max(use_graph[node1][node2] - graph[node1][node2], 0)
        use_graph[node1][node2] -= commodity_demand
        delta_fitness += max(use_graph[node1][node2] - graph[node1][node2], 0) - old_overload

    for i in range(len(new_path) - 1):
        node1 = new_path[i]
        node2 = new_path[i+1]
        old_overload = max(use_graph[node1][node2] - graph[node1][node2], 0)
        use_graph[node1][node2] += commodity_demand
        delta_fitness += max(use_graph[node1][node2] - graph[node1][node2], 0) - old_overload

    return delta_fitness


def create_new_path(graph, use_graph, commodity, all_distances, q=0.5, remove_cycles=False, better_heuristic=False):
    # Create a new path for a commodity using a guided random walk
    origin, destination, demand = commodity
    current_node = origin
    path_with_cycles = [current_node]

    while current_node != destination:
        heuristic_information_list = []
        neighbor_list = list(graph[current_node].keys())

        # Computing the heuristic information for each possible neighbor
        for neighbor in graph[current_node]:
            x = graph[current_node][neighbor] - use_graph[current_node][neighbor]

            if better_heuristic:
                heuristic_information_list.append(1 / (1 + all_distances[neighbor][destination]) + 0.5 * (1 + x / (1 + abs(x)) ))
            else:
                heuristic_information_list.append(1 + 0.5 * (1 + x / (1 + abs(x)) ))

        heuristic_information_list = np.array(heuristic_information_list)

        # Choosing the next node of the path
        if random.random() < q:
            neighbor_index = np.argmax(heuristic_information_list)
        else:
            proba_list = heuristic_information_list/np.sum(heuristic_information_list)
            neighbor_index = np.random.choice(len(neighbor_list), p=proba_list)

        current_node = neighbor_list[neighbor_index]
        path_with_cycles.append(current_node)

    if remove_cycles:
        # Cycle deletion
        path = []
        in_path = [False] * len(graph)
        for node in path_with_cycles:
            if in_path[node]:
                while path[-1] != node:
                    poped_node = path.pop()
                    in_path[poped_node] = False
            else:
                path.append(node)
        return path

    else:
        return path_with_cycles
