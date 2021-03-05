import heapq as hp
import random
import numpy as np
import time
import gurobipy

from mcnf_continuous import gurobi_congestion_solver, gurobi_overload_sum_solver
from mcnf_heuristics import find_fitting_most_capacited_path


def gurobi_unsplittable_flows(graph, commodity_list, verbose=0, time_limit=None):
    # MILP model that solves the unsplittable flow problem with an arc-node formulation

    nb_nodes = len(graph)
    nb_edges = sum([len(d) for d in graph])
    nb_commodities = len(commodity_list)


    arcs = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]
    capacities = [graph[node][neighbor] for node, neighbor in arcs]
    commodities = range(nb_commodities)

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 2
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Create variables
    flow = model.addVars(commodities, arcs, vtype=gurobipy.GRB.BINARY, name="flow") # flow variables
    overload_var = model.addVars(arcs, obj=1, name="lambda") # overload variables : we want to minimize their sum

    # Arc capacity constraints :
    model.addConstrs((sum([flow[commodity_index, node, neighbor] * commodity[2] for commodity_index, commodity in enumerate(commodity_list)]) - overload_var[node, neighbor] <= graph[node][neighbor] for node,neighbor in arcs), "cap")

    # Flow conservation constraints
    for commodity_index, commodity in enumerate(commodity_list):
        origin, destination, demand = commodity
        # print(origin, destination)
        for node in range(nb_nodes):

            if node == destination:
                rhs = -1
            elif node == origin:
                rhs = 1
            else:
                rhs = 0

            model.addConstr((flow.sum(commodity_index,node,'*') - flow.sum(commodity_index,'*',node) == rhs), "node{}_{}".format(node, origin))

    # Launching the model
    model.update()
    model.optimize()

    # Getting the results from the solver : the allocation of each super commodity and the total necessary capacity
    if model.status == gurobipy.GRB.Status.OPTIMAL or model.status == gurobipy.GRB.Status.TIME_LIMIT:
        remaining_capacity_graph = [{neighbor : graph[node][neighbor] for neighbor in graph[node]} for node in range(nb_nodes)]
        solution = model.getAttr('x', overload_var)
        for node,neighbor in arcs:
            remaining_capacity_graph[node][neighbor] += solution[node,neighbor]

        commodity_path_list = []
        solution = model.getAttr('x', flow)
        for commodity_index in range(nb_commodities):
            origin, destination, demand = commodity_list[commodity_index]

            allocation_graph = [{} for node in range(nb_nodes)]
            for node,neighbor in arcs:
                allocation_graph[node][neighbor] = solution[commodity_index,node,neighbor]

            path, path_capacity = find_fitting_most_capacited_path(allocation_graph, allocation_graph, origin, destination, 1)
            commodity_path_list.append(path)

        return commodity_path_list, model.objVal
    else:
        return None, model.objVal


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


def is_correct_path(graph, commodity, path):
    # function that checks if a path is valid for a commodity in an instance

    origin, destination, demand = commodity
    is_correct =  path[0] == origin and path[-1] == destination

    for node_index in range(len(path)-1):
        node, neighbor = path[node_index], path[node_index+1]
        is_correct = is_correct and neighbor in graph[node]
        if not is_correct:
            break

    return is_correct


def randomized_rounding_heuristic(graph, commodity_list, actualisation=True, actualisation_threshold=None, proof_constaint=False, sorted_commodities=True,
                                    linear_objectif="overload_sum", verbose=0):
    # randomized round heuristic :
    # - it uses information retreived from the continuous solution to create an unsplittable solution
    # - if actualisation = False and actualisation_threshold = None no actualisation is done (as in Raghavan and Thompson's algorithm)
    # - actualisation_threshold : when that many commodities have their path fixed while htey used several paths in the linear relaxation, the linear relaxation is acualised
    # - proof_constaint : tell the linear solver to add the constraint that makes this algorithm an approximation algorithm
    # - sorted_commodities : decides if the commodites are allocated in the graph in decreasing demand order
    # - linear_objectif : decide if we minimize the sum of overload or the maximum overload

    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    # Create the default actualisation_threshold
    if actualisation_threshold is None:
        if actualisation:
            actualisation_threshold = nb_nodes * 0.25
        else:
            actualisation_threshold = nb_commodities + 1

    counter = actualisation_threshold + 1

    use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)] # record the capacity used by the assigned commodities
    flow_upper_bound_graph = [{neighbor : graph[node][neighbor] for neighbor in graph[node]} for node in range(nb_nodes)]
    commodity_path_list = [[] for i in range(nb_commodities)] # store the path of the commodities : the solution of the problem

    commodities_order = list(range(nb_commodities))
    random.shuffle(commodities_order)
    if sorted_commodities:
         # Commodities are sorted by decreasing demand so that the biggest are assigned first and the smallest fills the gaps
        commodities_order.sort(key= lambda commodity_index : commodity_list[commodity_index][2])

    t = [0]*2
    # main loop : 1 commodity is assigned in each iteration
    while commodities_order != []:

        # at the begginning or when the solution deviates to much from the the previously computed continuous solution :
        # compute a new continuous solution with the non assigned commodities
        if counter > actualisation_threshold:
            temp = time.time()
            if verbose : print()
            counter = 0
            remaining_commodities = [commodity_list[index] for index in commodities_order]

            if linear_objectif == "overload_sum":
                allocation_graph_per_origin, remaining_capacity_graph = gurobi_overload_sum_solver(graph, remaining_commodities, use_graph=use_graph, flow_upper_bound_graph=flow_upper_bound_graph, proof_constaint=proof_constaint, verbose=max(0,verbose-1))
            elif linear_objectif == "congestion":
                allocation_graph_per_origin, remaining_capacity_graph = gurobi_congestion_solver(graph, remaining_commodities, use_graph=use_graph, flow_upper_bound_graph=flow_upper_bound_graph, proof_constaint=proof_constaint, verbose=max(0,verbose-1))
            else:
                assert False, "Objectif not implemented, please check your spelling or contribute"
            t[0] += time.time() - temp


        temp = time.time()
        # Select a commodity
        commodity_index = commodities_order.pop()
        origin, destination, demand = commodity_list[commodity_index]
        allocation_graph = allocation_graph_per_origin[origin]

        # Extract the paths of the commodity from the linear relaxation results
        remaining_demand = demand
        path_list = []
        used_capacity_list = []
        while remaining_demand > 10**-6:
            path, path_capacity = find_fitting_most_capacited_path(allocation_graph, remaining_capacity_graph, origin, destination, demand)
            if path is None or path_capacity <= 10**-5:
                path, path_capacity = find_fitting_most_capacited_path(allocation_graph, remaining_capacity_graph, origin, destination, -10**10)

            used_capacity = min(path_capacity, remaining_demand)
            path_list.append(path)
            used_capacity_list.append(used_capacity)
            remaining_demand -= used_capacity
            update_graph_capacity(allocation_graph, path, used_capacity)
            update_graph_capacity(flow_upper_bound_graph, path, used_capacity)

        if len(path_list) > 1:
            counter += 1

        # Choose a path for the commodity
        proba_list = np.array(used_capacity_list) / sum(used_capacity_list)
        chosen_path_index = np.random.choice(len(path_list), p=proba_list)
        path = path_list[chosen_path_index]

        # allcoate the commodity and update the capacities in different graphs
        update_graph_capacity(use_graph, path, -demand)
        update_graph_capacity(remaining_capacity_graph, path, demand)
        commodity_path_list[commodity_index] = path
        t[1] += time.time() - temp


        # a bit of printing
        if len(commodities_order) % 100 == 0 or counter > actualisation_threshold:
            if verbose : print(len(commodities_order), sum([commodity_list[index][2] for index in commodities_order]), end="         \r")

    # compute metrics for the overload
    overload_graph = [{neighbor : max(0, use_graph[node][neighbor] - graph[node][neighbor]) for neighbor in graph[node]} for node in range(len(graph))]
    congestion_graph = [{neighbor : use_graph[node][neighbor] / graph[node][neighbor] for neighbor in graph[node] if graph[node][neighbor] > 0} for node in range(len(graph))]
    total_overload = sum([sum(dct.values()) for dct in overload_graph])
    if verbose :
        print("total_overload is ", total_overload)
        print("Congestion is ", max([max(list(dct.values())+[0]) for dct in congestion_graph]))
        print(t)

    return commodity_path_list, total_overload


def update_graph_capacity(graph, path, demand, reverse_graph=False):
    # deecrease the capacities in the graph taken by a commodity size "demand" and allocate to the path "path"
    # also computes the overload created
    # negative demands are possible to increase the capacity instead of decreasing it

    new_overload = 0

    for i in range(len(path)-1):
        node = path[i]
        neighbor = path[i+1]

        if reverse_graph:
            node, neighbor = neighbor, node

        old_overload = - min(0, graph[node][neighbor])
        graph[node][neighbor] -= demand
        new_overload += - min(0, graph[node][neighbor]) - old_overload

    return new_overload
