import random
import numpy as np
import heapq as hp
import matplotlib.pyplot as plt
import time



def annealing_unsplittable_flows(graph, commodity_list, commodity_path_list=None, nb_iterations=10**5, verbose=0):
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    all_distances = compute_all_distances(graph)
    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]

    T = T_init = 200
    T_final = 1
    dT = (T_final/T_init)**(1/nb_iterations)

    # Create a first solution
    solution = []
    fitness = 0
    for commodity_index, commodity in enumerate(commodity_list):
        new_path = get_new_path_2(graph, use_graph, commodity, all_distances, T=3)
        if commodity_path_list is not None :
            new_path = commodity_path_list[commodity_index]
        solution.append(new_path)
        fitness += update_fitness_and_use_graph(use_graph, graph, [], new_path, commodity[2])

    fitness_log = []
    proba_log = []
    for iter_index in range(nb_iterations):
        if verbose and iter_index %1000 == 0:
            print(iter_index, fitness, end='   \r')

        # Make a modification on the current solution
        commodity_index = random.randint(0, nb_commodities - 1)
        commodity = commodity_list[commodity_index]
        if random.random() < 0.2:
            new_path = get_new_path(graph, use_graph, commodity, all_distances, T=0.5)
        else:
            new_path = get_new_path(graph, use_graph, commodity, all_distances, T=3)
        old_path = solution[commodity_index]
        solution[commodity_index] = new_path

        # Evaluate the new solution
        new_fitness = fitness + update_fitness_and_use_graph(use_graph, graph, old_path, new_path, commodity[2])

        # Decide wether or not to keep the new solution
        proba = 1 * np.exp((fitness - new_fitness) / T)
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

    # plt.plot(fitness_log)
    # plt.plot(proba_log)
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


def get_new_path(graph, use_graph, commodity, all_distances, T=1, overload_coeff=None):
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
