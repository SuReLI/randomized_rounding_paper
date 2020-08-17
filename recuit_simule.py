import random
import numpy as np
import heapq as hp
import matplotlib.pyplot as plt
import time

from k_shortest_path import k_shortest_path_algorithm
from mcnf import randomized_rounding_heuristic


def recuit_unsplittable_flows(graph, commodity_list, commodity_path_list=None, nb_iterations=10**5, verbose=0):
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    all_distances = compute_all_distances(graph)
    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]


    T = T_init = 200
    T_final = 1
    dT = (T_final/T_init)**(1/nb_iterations)

    solution = []
    fitness_log = []
    proba_log = []
    fitness = 0
    # commodity_path_list, _ = randomized_rounding_heuristic(graph, commodity_list, actulisation_threshold=10**10, verbose=1)
    for commodity_index, commodity in enumerate(commodity_list):
        new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=3)
        if commodity_path_list is not None :
            new_path = commodity_path_list[commodity_index]
        solution.append(new_path)
        fitness += update_fitness_and_use_graph(use_graph, graph, [], new_path, commodity[2])

    for iter_index in range(nb_iterations):
        if verbose and iter_index %1000 == 0:
            print(iter_index, fitness, end='   \r')


        commodity_index = random.randint(0, nb_commodities - 1)
        commodity = commodity_list[commodity_index]

        if commodity_path_list is not None and random.random() < 0.:
            new_path = commodity_path_list[commodity_index]
        elif random.random() < 0.:
            new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=0.5)
        elif random.random() < 0:
            new_path = get_new_path_LP(graph, use_graph, commodity, solution[commodity_index])
        else:
            new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=3)

        new_fitness = fitness + update_fitness_and_use_graph(use_graph, graph, solution[commodity_index], new_path, commodity[2])
        old_path = solution[commodity_index]
        solution[commodity_index] = new_path

        proba = 1 * np.exp((fitness - new_fitness) / T)# / T_d)# / commodity[2])

        if new_fitness <= fitness or random.random() < proba:
            fitness = new_fitness
        else:
            solution[commodity_index] = old_path
            update_fitness_and_use_graph(use_graph, graph, new_path, old_path, commodity[2])

        fitness_log.append(fitness)
        if proba >= 1 :
            proba_log.append(0)
        else:
            proba_log.append(proba*1000)
        T = T * dT

    plt.plot(fitness_log)
    plt.show()

    return solution, fitness



def recuit_unsplittable_flows2(graph, commodity_list, commodity_path_list=None, nb_iterations=10**5, verbose=0):
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    all_distances = compute_all_distances(graph)
    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]

    T = T_init = 1
    T_final = 1
    dT = (T_final/T_init)**(1/nb_iterations)

    solution = []
    fitness_log = []
    proba_log = []
    t = [0]*4
    fitness = 0
    for commodity_index, commodity in enumerate(commodity_list):
        new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=3)
        if commodity_path_list is not None :
            new_path = commodity_path_list[commodity_index]
        solution.append(new_path)
        fitness += update_fitness_and_use_graph(use_graph, graph, [], new_path, commodity[2])

    possible_paths = [[path] for path in solution]

    for iter_index in range(nb_iterations):
        if verbose and iter_index %1000 == 0:
            print(iter_index, fitness, t, end='   \r')

        temp = time.time()
        commodity_index = random.randint(0, nb_commodities - 1)
        commodity = commodity_list[commodity_index]

        if random.random() < 0.01 or len(possible_paths[commodity_index]) < 10:
            if commodity_path_list is not None and random.random() < 0.:
                new_path = commodity_path_list[commodity_index]
            elif random.random() < 0.:
                new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=0.5)
            elif random.random() < 0:
                new_path = get_new_path_LP(graph, use_graph, commodity, solution[commodity_index])
            else:
                new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=3)

            if len(possible_paths[commodity_index]) == 10:
                possible_paths[commodity_index].pop(random.randint(0, len(possible_paths[commodity_index]) - 1))
            possible_paths[commodity_index].append(new_path)

        new_path = possible_paths[commodity_index][random.randint(0, len(possible_paths[commodity_index]) - 1)]
        t[0] += time.time() - temp
        temp = time.time()

        new_fitness = fitness + update_fitness_and_use_graph(use_graph, graph, solution[commodity_index], new_path, commodity[2])
        old_path = solution[commodity_index]
        solution[commodity_index] = new_path

        proba = 1 * np.exp((fitness - new_fitness) / T)# / T_d)# / commodity[2])

        if new_fitness <= fitness or random.random() < proba:
            fitness = new_fitness
        else:
            solution[commodity_index] = old_path
            update_fitness_and_use_graph(use_graph, graph, new_path, old_path, commodity[2])

        fitness_log.append(fitness)
        if proba >= 1 :
            proba_log.append(0)
        else:
            proba_log.append(proba*1000)
        T = T * dT
        t[1] += time.time() - temp

    # plt.plot(fitness_log)
    # plt.show()

    return solution, fitness



def update_fitness_and_use_graph(use_graph, graph, old_path, new_path, commodity_demand):
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


def get_new_path(graph, commodity, all_distances, T=1):
    origin, destination, demand = commodity
    current_node = origin
    new_path = [current_node]

    while current_node != destination:
        neighbor_list = list(graph[current_node].keys())
        distances_to_destination = []
        l = []

        for neighbor in neighbor_list:
            if True or neighbor in graph[current_node] and graph[current_node][neighbor] > 0:
                distances_to_destination.append(all_distances[neighbor][destination] + 1)
                l.append(neighbor)

        distances_to_destination = np.array(distances_to_destination)
        neighbor_list = l

        proba = np.exp(- distances_to_destination * T)
        proba = proba/sum(proba)

        current_node = np.random.choice(neighbor_list, p=proba)
        new_path.append(current_node)

    return new_path


def get_new_path_2(graph, use_graph, commodity, all_distances, T=1, overload_coeff=None):
    origin, destination, demand = commodity
    current_node = origin
    new_path = [current_node]
    if overload_coeff is None:
        overload_coeff = 10**-3

    i = 0
    while current_node != destination:
        i+=1
        if i%20==0:
            overload_coeff /= 10

        neighbor_list = list(graph[current_node].keys())
        arc_efficiency_list = []
        l = []

        for neighbor in neighbor_list:
            if True or neighbor in graph[current_node] and graph[current_node][neighbor] > 0:
                arc_efficiency_list.append(all_distances[neighbor][destination] + 1 + overload_coeff * max(0, demand + use_graph[current_node][neighbor] - graph[current_node][neighbor]))
                l.append(neighbor)

        arc_efficiency_list = np.array(arc_efficiency_list)
        neighbor_list = l

        proba = np.exp(- arc_efficiency_list * T)
        proba = proba/sum(proba)

        current_node = np.random.choice(neighbor_list, p=proba)
        new_path.append(current_node)

    return new_path


def get_new_path_LP(graph, use_graph, commodity, old_path):
    nb_nodes = len(graph)
    origin, destination, demand = commodity
    remaining_capacity_graph = [{neighbor : graph[node][neighbor] - use_graph[node][neighbor] for neighbor in graph[node]} for node in range(nb_nodes)]
    update_graph_capacity(remaining_capacity_graph, old_path, -demand)

    remaining_demand = demand
    path_list = []
    used_capacity_list = []
    while remaining_demand > 10**-6:
        path, path_capacity = dijkstra_for_LP(remaining_capacity_graph, origin, destination)

        used_capacity = min(path_capacity, remaining_demand)
        path_list.append(path)
        used_capacity_list.append(used_capacity)
        remaining_demand -= used_capacity
        update_graph_capacity(remaining_capacity_graph, path, used_capacity)

    # Choose a path for the commodity
    proba_list = np.array(used_capacity_list) / sum(used_capacity_list)
    new_path_index = np.random.choice(len(path_list), p=proba_list)
    new_path = path_list[new_path_index]

    return new_path


def dijkstra_for_LP(graph, initial_node, destination_node=None):
    priority_q = [(0, initial_node, None)]
    parent_list = [None] * len(graph)
    distances = [None] * len(graph)

    while priority_q:
        value, current_node, parent_node = hp.heappop(priority_q)
        if distances[current_node] is None:
            parent_list[current_node] = parent_node
            distances[current_node] = value

            if current_node == destination_node:
                break

            for neighbor in graph[current_node]:
                if distances[neighbor] is None:
                    hp.heappush(priority_q, (value + int(graph[current_node][neighbor] <= 10**-5) + 0*10**-3, neighbor, current_node))

    path = [destination_node]
    current_node = destination_node
    path_capacity = 10**10
    while current_node != initial_node:
        if graph[parent_list[current_node]][current_node] > 10**-5:
            path_capacity = min(path_capacity, graph[parent_list[current_node]][current_node])
        current_node = parent_list[current_node]
        path.append(current_node)
    path.reverse()

    return path, path_capacity




def update_graph_capacity(graph, path, demand, reverse_graph=False):
    # deecrease the capacities in the graph taken by a commodity size "demand" and allocate to the path "path"
    # also computes the overload created
    # negative demands are possible to increase the capacity instead of decreasing it

    new_overload = 0

    for node_index in range(len(path)-1):
        node, neighbor = path[node_index], path[node_index+1]

        if reverse_graph:
            node, neighbor = neighbor, node

        old_overload = - min(0, graph[node][neighbor])
        graph[node][neighbor] -= demand
        new_overload += - min(0, graph[node][neighbor]) - old_overload

    return new_overload

def compute_all_distances(graph):
    nb_nodes = len(graph)
    all_distances = []
    unitary_graph = [{neigh : 1 for neigh in graph[node]} for node in range(nb_nodes)]

    for initial_node in range(nb_nodes):
        parent_list, distances = dijkstra(unitary_graph, initial_node)
        for i in range(len(distances)):
            if distances[i] is None:
                distances[i] = 10.**10
        all_distances.append(distances)

    return all_distances


def dijkstra(graph, initial_node, destination_node=None):
    priority_q = [(0, initial_node, None)]
    parent_list = [None] * len(graph)
    distances = [None] * len(graph)

    while priority_q:
        value, current_node, parent_node = hp.heappop(priority_q)
        if distances[current_node] is None:
            parent_list[current_node] = parent_node
            distances[current_node] = value

            if current_node == destination_node:
                return parent_list, distances

            for neighbor in graph[current_node]:
                if distances[neighbor] is None:
                    hp.heappush(priority_q, (value + graph[current_node][neighbor], neighbor, current_node))

    return parent_list, distances



if __name__ == "__main__":
    pass
