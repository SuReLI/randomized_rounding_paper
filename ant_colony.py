import heapq as hp
import random
import numpy as np
import time

def ant_colony_optimiser(graph, commodity_list, nb_iterations, verbose=0):
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    nb_edges = sum(len(neighbor_dict) for neighbor_dict in graph)

    # Setting hyper-parameters
    evaporation_factor = 0.9
    reset_counter = 0
    reset_threshold = 50

    solution, solution_value = None, 10**15
    best_solution, best_solution_value = None, 10**15

    pheromone_trails_per_commodity = np.ones((nb_commodities, nb_edges)) * 0.5 # Initialize pheromone trails
    all_distances = compute_all_distances(graph)
    edge_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]
    edge_index_dict = {edge : edge_index for edge_index, edge in edge_list}
    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]
    commodities_order = list(range(nb_commodities))

    # Main loop
    for iter_index in range(nb_iterations):
        if iter_index %1 == 0 and verbose:
            print(iter_index, solution_value, t)

        temp = time.time()

        old_solution = solution
        solution = [None]*nb_commodities
        solution_value = 0
        use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]

        # Create a new solution with the ant colony method
        for commodity_index in commodities_order:
            commodity = commodity_list[commodity_index]
            path = compute_path(graph, use_graph, all_distances, pheromone_trails_per_commodity[commodity_index], edge_index_dict, commodity)
            solution[commodity_index] = path
            solution_value += update_fitness_and_use_graph(use_graph, graph, [], path, commodity[2])

        # Apply local search
        solution_value = local_search(solution, use_graph, graph, solution_value, commodity_list)

        # Keep the solution if it is the best so far
        if best_solution_value >= solution_value:
            best_solution = solution
            best_solution_value = solution_value
            reset_counter = 0

        else:
            reset_counter += 1

        # Update the pheromones
        pheromone_trails_per_commodity *= evaporation_factor
        update_pheromones(pheromone_trails_per_commodity, edge_index_dict, best_solution, evaporation_factor)

        # Reset the pheromone if no improvement is made for a large number of iteration
        if reset_counter >= reset_threshold:
            pheromone_trails_per_commodity = np.ones((nb_commodities, nb_edges)) * 0.5
            solution = None
            reset_counter = 0

    return best_solution, best_solution_value


def update_pheromones(pheromone_trails_per_commodity, edge_index_dict, best_solution, evaporation_factor):
    # This function update the pheromones of an ant colony optimiser in the hyper-cube framework

    for commodity_index, path in enumerate(best_solution):
        pheromone_trails = pheromone_trails_per_commodity[commodity_index]

        for node_index in range(len(path) - 1):
            node, neighbor = path[node_index], path[node_index + 1]
            edge_index = edge_index_dict[node, neighbor]
            pheromone_trails[edge_index] += (1 - evaporation_factor)
            pheromone_trails[edge_index]  = max(0.001, min(0.999, pheromone_trails[edge_index]))


def compute_path(graph, use_graph, all_distances, pheromone_trails, edge_index_dict, commodity, beta=2, gamma=3, p0=0.4, reset_lenght=25):
    # This function creates a new path for a commodity with the ant colony method : have an ant do a guided random walked on the graph

    nb_nodes = len(graph)
    origin, destination, demand = commodity
    current_node = origin
    visited = [False] * nb_nodes
    visited[current_node] = True
    path_with_cycles = [current_node]

    while current_node != destination:
        neighbor_list = [neighbor for neighbor in graph[current_node] if not visited[neighbor]]

        if neighbor_list == []:
            neighbor_list = [neighbor for neighbor in graph[current_node]]

        # Compute all heuristic information necessary to choose the next arc
        pheromone_array = np.array([pheromone_trails[edge_index_dict[current_node, neighbor]] for neighbor in neighbor_list])
        pheromone_array = np.maximum(0.001, pheromone_array)
        use_array = np.array([use_graph[current_node][neighbor] for neighbor in neighbor_list])
        if np.sum(use_array) == 0:
            use_heuristic = use_array + 1
        else:
            use_heuristic = np.maximum(0.001, 1 - use_array / np.sum(use_array))
        distance_heuristic = np.array([all_distances[neighbor][destination] for neighbor in neighbor_list])
        distance_heuristic = 1 / np.maximum(1, distance_heuristic)

        proba_list = use_heuristic**beta * distance_heuristic**gamma * pheromone_array

        # Choose the next arc
        if random.random() < p0:
            neighbor_index = np.argmax(proba_list)
        else:
            proba_list = proba_list / sum(proba_list)
            neighbor_index = np.random.choice(len(neighbor_list), p=proba_list)

        neighbor = neighbor_list[neighbor_index]
        visited[neighbor] = True
        path_with_cycles.append(neighbor)
        current_node = neighbor

        # Reset the path if it becomes too long
        if len(path_with_cycles) > reset_lenght:
            current_node = origin
            visited = [False] * nb_nodes
            visited[current_node] = True
            path_with_cycles = [current_node]

    # Cycle deletion
    path = []
    in_path = [False] * nb_nodes
    for node in path_with_cycles:
        if in_path[node]:
            while path[-1] != node:
                poped_node = path.pop()
                in_path[poped_node] = False
        else:
            path.append(node)

    return path


def local_search(solution, use_graph, graph, solution_value, commodity_list):
    counter = 0

    while counter < 3:
        arc = find_random_most_congested_arc(use_graph, graph)
        commodities_on_congested_arc = find_commodities_on_arc(solution, arc)
        best_congestion_reduction = 0
        best_reduction_move = None
        node1, node2 = arc
        current_congestion = use_graph[node1][node2] / graph[node1][node2]

        for commodity_index, node1_index in commodities_on_congested_arc:
            origin, destination, demand = commodity_list[commodity_index]
            path = solution[commodity_index]

            if node1_index > 0 and node2 in graph[path[node1_index - 1]]:
                # Try skip first node
                node1_predecessor = path[node1_index - 1]
                congestion_reduction = current_congestion - (use_graph[node1_predecessor][node2] + demand) / graph[node1_predecessor][node2]

                if congestion_reduction > best_congestion_reduction :
                    best_congestion_reduction = congestion_reduction
                    best_reduction_move = (commodity_index, path[ : node1_index] + path[node1_index + 1 :])

            if node1_index < len(path)-2 and path[node1_index + 2] in graph[node1]:
                # Try skip second node
                node2_successor = path[node1_index + 2]
                congestion_reduction = current_congestion - (use_graph[node1][node2_successor] + demand) / graph[node1][node2_successor]

                if congestion_reduction > best_congestion_reduction :
                    best_congestion_reduction = congestion_reduction
                    best_reduction_move = (commodity_index, path[ : node1_index + 1] + path[node1_index + 2 :])

            for added_node in [neighbor for neighbor in graph[node1] if node2 in graph(neighbor)]:
                # Try add intermediate node
                new_congestion1 = (use_graph[node1][added_node] + demand) / graph[node1][added_node]
                new_congestion2 = (use_graph[added_node][node2] + demand) / graph[added_node][node2]
                congestion_reduction = current_congestion - max(new_congestion1, new_congestion2)

                if congestion_reduction > best_congestion_reduction :
                    best_congestion_reduction = congestion_reduction
                    best_reduction_move = (commodity_index, path[ : node1_index + 1]+ [added_node] + path[node1_index + 1 :])

        if best_congestion_reduction < 0:
            changed_commodity_index, new_path = best_reduction_move
            solution_value += update_fitness_and_use_graph(use_graph, graph, solution[changed_commodity_index], new_path, commodity_list[commodity_index][2])
            solution[changed_commodity_index] = new_path

        else:
            counter += 1

    return solution_value


def find_random_most_congested_arc(use_graph, graph):
    nb_nodes = len(graph)
    largest_congestion = 0
    most_congested_arc_list = []

    for node in range(nb_nodes):
        for neighbor in graph[node]:
            congestion = use_graph[node][neighbor] / graph[node][neighbor]

            if congestion > largest_congestion:
                largest_congestion = congestion
                most_congested_arc_list = []

            if congestion >= largest_congestion:
                most_congested_arc_list.append((node, neighbor))

    chosen_arc_index = np.random.choice(len(most_congested_arc_list))

    return most_congested_arc_list[chosen_arc_index]


def find_commodities_on_arc(solution, arc):
    commodities_on_arc = []

    for commodity_index, path in enumerate(solution):
        for node_index in range(len(path) - 1):
            if path == (path[node_index], path[node_index + 1]):
                commodities_on_arc.append((commodity_index, node_index))
                break

    return commodities_on_arc


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
