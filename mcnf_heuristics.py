import random
import numpy as np
import heapq as hp

def single_source_mcnf_preprocessing(reverse_graph, commodity_list):
    commodity_path_list = [[] for c in commodity_list]
    process_graph = [{neighbor : reverse_graph[node][neighbor] for neighbor in reverse_graph[node]} for node in range(len(reverse_graph))]
    commodity_indices = sorted(list(range(len(commodity_list))), key = lambda x : commodity_list[x][2], reverse=True)

    for commodity_index in commodity_indices:
        origin, destination, demand = commodity_list[commodity_index]
        path = commodity_path_list[commodity_index]

        current_node  = destination

        while current_node != origin:
            for neighbor in process_graph[current_node]:
                if process_graph[current_node][neighbor] >= demand:
                    process_graph[current_node][neighbor] -= demand
                    path.append(current_node)
                    current_node = neighbor
                    break
            else:
                while path != []:
                    previous_node = path.pop()
                    process_graph[previous_node][current_node] += demand
                    current_node = previous_node
                break

        if current_node == origin:
            path.append(origin)

        path.reverse()

    return process_graph, commodity_path_list


def allocate_biggest_commodities(reverse_graph, commodity_list, nb_max_failure=0):
    commodity_path_list = [[] for c in commodity_list]
    process_graph = [{neighbor : reverse_graph[neighbor][node] for neighbor in range(len(reverse_graph)) if node in reverse_graph[neighbor]} for node in range(len(reverse_graph))]
    commodity_indices = sorted(list(range(len(commodity_list))), key = lambda x : commodity_list[x][2], reverse=True)
    nb_failure = 0

    for commodity_index in commodity_indices:
        origin, destination, demand = commodity_list[commodity_index]
        path = find_most_capacited_path(process_graph, origin, destination, demand)
        overload = update_graph_capacity(graph, path, demand)
        commodity_path_list[commodity_index] = path

        if overload > 0:
            nb_failure += 1
            if nb_failure > nb_max_failure:
                return process_graph, commodity_path_list

    return process_graph, commodity_path_list


def find_fitting_most_capacited_path(graph1, graph2, origin, destination, minimum_capacity):
    # this is a dijkstra like algorithm that computes the most capacited path from the origin to the destination
    # the best capacity is according to graph1 but only edges with capacities >= minimum_capacity in graph2 are taken into account
    priority_q = [(-10.**10, origin, None)]

    parent_list = [None] * len(graph1)
    visited = [False]*len(graph1)
    best_capacity = [None] * len(graph1)

    while priority_q != []:
        c, current_node, parent_node = hp.heappop(priority_q)
        capa_of_current_node = -c

        if not visited[current_node]:
            visited[current_node] = True
            parent_list[current_node] = parent_node
            best_capacity[current_node] = capa_of_current_node

            if current_node == destination:
                break

            for neighbor in graph1[current_node]:
                if not visited[neighbor] and graph2[current_node][neighbor] >= minimum_capacity:
                    hp.heappush(priority_q, (-min(capa_of_current_node, graph1[current_node][neighbor]), neighbor, current_node))

    if parent_list[destination] is None:
        return None, None

    path = [destination]
    current_node = destination
    while current_node != origin:
        current_node = parent_list[current_node]
        path.append(current_node)
    path.reverse()

    return path, best_capacity[destination]


def find_shortest_path(graph, origin, destination, minimum_capacity):
    # this is a breadth first search algorithm
    pile = [(origin, None)]
    parent_list = [None] * len(graph)
    visited = [0]*len(graph)
    visited[origin] = 1

    while pile != []:
        current_node, parent_node = pile.pop(0)
        parent_list[current_node] = parent_node

        if current_node == destination:
            break

        neighbor_list = list(graph[current_node].keys())
        random.shuffle(neighbor_list)

        for neighbor in neighbor_list:
            if not visited[neighbor] and graph[current_node][neighbor] >= minimum_capacity:
                pile.append((neighbor, current_node))
                visited[neighbor] = 1

    if parent_list[destination] is None:
        return

    path = [destination]
    current_node = destination
    while current_node != origin:
        current_node = parent_list[current_node]
        path.append(current_node)

    path.reverse()

    return path

def find_shortest_path2(graph, origin, destination):
    # this is a breadth first search algorithm
    pile = [(origin, None)]

    parent_list = [None] * len(graph)
    visited = [0]*len(graph)
    visited[origin] = 1
    path_capacity = [None]*len(graph)


    while pile != []:
        current_node, parent_node = pile.pop(0)
        parent_list[current_node] = parent_node

        if current_node != origin:
            path_capacity[current_node] = min(path_capacity[parent_node], graph[parent_node][current_node])
        else:
            path_capacity[origin] = 10**10

        if current_node == destination:
            break

        neighbor_list = list(graph[current_node].keys())
        random.shuffle(neighbor_list)

        for neighbor in neighbor_list:
            if not visited[neighbor] and graph[current_node][neighbor] > 0:
                pile.append((neighbor, current_node))
                visited[neighbor] = 1

    if parent_list[destination] is None:
        return None, None

    path = [destination]
    current_node = destination
    while current_node != origin:
        current_node = parent_list[current_node]
        path.append(current_node)

    path.reverse()

    return path, path_capacity[destination]


def find_shortest_path_double_graph(graph1, graph2, origin, destination, minimum_capacity1, minimum_capacity2):
    # this is a breadth first search algorithm where edges must verify capacity condition in the 2 graphs
    pile = [(origin, None)]

    parent_list = [None] * len(graph1)
    visited = [0]*len(graph1)
    visited[origin] = 1


    while pile != []:
        current_node, parent_node = pile.pop(0)
        parent_list[current_node] = parent_node

        if current_node == destination:
            break

        neighbor_list = list(graph1[current_node].keys())
        random.shuffle(neighbor_list)

        for neighbor in neighbor_list:
            if not visited[neighbor] and graph1[current_node][neighbor] >= minimum_capacity1 and graph2[current_node][neighbor] >= minimum_capacity2:
                pile.append((neighbor, current_node))
                visited[neighbor] = 1

    if parent_list[destination] is None:
        return

    path = [destination]
    current_node = destination
    while current_node != origin:
        current_node = parent_list[current_node]
        path.append(current_node)

    path.reverse()

    return path

def compute_all_shortest_path(graph, origin_list):
    nb_nodes = len(graph)
    all_shortest_path = {}

    for origin in origin_list:
        parent_list, distances = dijkstra(graph, origin)
        shortest_path_list = [None for node in range(nb_nodes)]
        shortest_path_list[origin] = [origin]

        for node in range(nb_nodes):
            if shortest_path_list[node] is None and parent_list[node] is not None:
                compute_shortest_path(shortest_path_list, parent_list, node)

        all_shortest_path[origin] = [(shortest_path_list[node], distances[node]) for node in range(nb_nodes)]

    return all_shortest_path


def compute_shortest_path(shortest_path_list, parent_list, node):
    parent = parent_list[node]
    if shortest_path_list[parent] is None:
        compute_shortest_path(shortest_path_list, parent_list, parent)

    shortest_path_list[node] = shortest_path_list[parent] + [node]


def dijkstra(graph, intial_node, destination_node=None):
    priority_q = [(0, intial_node, None)]
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


def update_graph_capacity(graph, path, demand, reverse_graph=False):
    overload = 0

    for i in range(len(path)-1):
        node = path[i]
        neighbor = path[i+1]

        if reverse_graph:
            node, neighbor = neighbor, node

        graph[node][neighbor] -= demand
        overload += max(0, min(-graph[node][neighbor], demand))

    return overload


def is_strongly_connected(graph):
    nb_nodes = len(graph)

    for initial_node in range(nb_nodes):
        reachable = [False]*nb_nodes
        reachable[initial_node] = True
        pile = [initial_node]
        nb_reachable = 1

        while pile:
            current_node = pile.pop()
            for neighbor in graph[current_node]:
                if not reachable[neighbor]:
                    reachable[neighbor] = True
                    pile.append(neighbor)
                    nb_reachable += 1

        if nb_reachable < nb_nodes:
            print(nb_reachable, nb_nodes)
            return False

    return True
