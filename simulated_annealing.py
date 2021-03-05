import random
import numpy as np
import heapq as hp
import time

from k_shortest_path import k_shortest_path_algorithm, k_shortest_path_all_destination


def simulated_annealing_unsplittable_flows(graph, commodity_list, nb_iterations=10**5, nb_k_shortest_paths=10, verbose=0):
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    # Set the initial/final temperature and the temperature decrement
    T = T_init = 200
    T_final = 1
    dT = (T_final/T_init)**(1/nb_iterations)

    all_distances = compute_all_distances(graph)
    log_values = - np.log(1 - np.arange(0, 1, 0.001))
    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)] # Will contain the total flow used on each arc

    # Compute the k-shortest paths for each commodity and store them in possible_paths_per_commodity
    shortest_paths_per_origin = {}
    possible_paths_per_commodity = []
    for commodity_index, commodity in enumerate(commodity_list):
        origin, destination, demand = commodity

        if origin not in shortest_paths_per_origin:
            shortest_paths_per_origin[origin] = k_shortest_path_all_destination(graph, origin, nb_k_shortest_paths)

        path_and_cost_list = shortest_paths_per_origin[origin][destination]
        possible_paths_per_commodity.append([remove_cycle_from_path(path) for path, cost in path_and_cost_list])

    solution = []
    fitness = 0

    # Create an intial solution
    for commodity_index, commodity in enumerate(commodity_list):
        new_path_index = np.random.choice(len(possible_paths_per_commodity[commodity_index]))
        new_path = possible_paths_per_commodity[commodity_index][new_path_index]
        solution.append(new_path)
        fitness += update_fitness_and_use_graph(use_graph, graph, [], new_path, commodity[2])

    # Main loop
    for iter_index in range(nb_iterations):
        if verbose and iter_index %1000 == 0:
            print(iter_index, fitness, end='   \r')

        commodity_index = random.randint(0, nb_commodities - 1) # Choose the commodity which will change its path during the iteration
        commodity = commodity_list[commodity_index]
        nb_possible_paths = len(possible_paths_per_commodity[commodity_index])


        if nb_possible_paths < 5 or random.random() < 0.:
            if nb_possible_paths >= 10:
                index_to_remove = np.random.choice(len(possible_paths_per_commodity[commodity_index]))
                possible_paths_per_commodity[commodity_index].pop(index_to_remove)

            # Create a new possible path for the commodity : this procedure considers the current overflow of the arcs
            new_path = get_new_path(graph, use_graph, commodity, log_values, all_distances, T=3)
            possible_paths_per_commodity[commodity_index].append(new_path)

        else:
            # Choose a random new_path for the commodity
            new_path_index = np.random.choice(len(possible_paths_per_commodity[commodity_index]))
            new_path = possible_paths_per_commodity[commodity_index][new_path_index]

        # Change the path used by the commodity and modify the fitness accordingly
        new_fitness = fitness + update_fitness_and_use_graph(use_graph, graph, solution[commodity_index], new_path, commodity[2])
        old_path = solution[commodity_index]
        solution[commodity_index] = new_path

        proba = np.exp((fitness - new_fitness) / T)
        T = T * dT

        # Keep the new solution according to the simulated annealing rule or return to the old solution
        if new_fitness <= fitness or random.random() < proba:
            fitness = new_fitness
        else:
            solution[commodity_index] = old_path
            update_fitness_and_use_graph(use_graph, graph, new_path, old_path, commodity[2]) # Modify use_graph

    return solution, fitness


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


def get_new_path(graph, use_graph, commodity, log_values, all_distances, T=1, overload_coeff=10**-2):
    # This functions create a random path for a commodity biased toward short path and arcs with a low overflow
    # To do so, an A* algorithm is applied with random arc cost and a penalisation of the overflow

    origin, destination, demand = commodity
    priority_q = [(0, origin, None)] # Priority queue storing the nodes to explore : encode with a binary tree
    parent_list = [None] * len(graph)
    distances = [None] * len(graph)

    while priority_q:
        value, current_node, parent_node = hp.heappop(priority_q)
        if distances[current_node] is None:
            parent_list[current_node] = parent_node
            distances[current_node] = value

            if current_node == destination:
                break

            for neighbor in graph[current_node]:
                if distances[neighbor] is None:
                    arc_cost = 1 # Each arc has length 1 to guide toward shorter paths
                    arc_cost += overload_coeff * max(0, demand + use_graph[current_node][neighbor] - graph[current_node][neighbor]) # Penalize the arc with high overflow
                    arc_cost += log_values[int(random.random() * 1000)] # Add some randomness to the arc cost
                    arc_cost += all_distances[neighbor][destination] # A* heuristic : distance to the target node
                    hp.heappush(priority_q, (value + arc_cost, neighbor, current_node))

    # Compute a reverse path according to parent_list
    path = [destination]
    current_node = destination
    while current_node != origin:
        current_node = parent_list[current_node]
        path.append(current_node)

    path.reverse()
    return path


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


def compute_all_distances(graph):
    nb_nodes = len(graph)
    all_distances = []
    unitary_graph = [{neighbor : 1 for neighbor in graph[node]} for node in range(nb_nodes)]

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


def remove_cycle_from_path(path):
    is_in_path = set()
    new_path = []

    for node in path:
        if node in is_in_path:
            while new_path[-1] != node:
                node_to_remove = new_path.pop()
                is_in_path.remove(node_to_remove)

        else:
            is_in_path.add(node)
            new_path.append(node)

    return new_path



if __name__ == "__main__":
    pass
