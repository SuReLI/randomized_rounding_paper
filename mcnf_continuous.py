import random
import numpy as np
import gurobipy


def gurobi_overload_sum_solver(graph, commodity_list, use_graph=None, flow_upper_bound_graph=None, verbose=0, proof_constaint=False, return_model=False):
    # LP program that solves the multicommodity flow problem with the following objective function : minimize the sum of the arcs overload

    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    if use_graph is None:
        use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]

    # we aggregate the commodities by origin : this create a super commodity
    # this process does not change the results of the continuous solver
    super_commodity_dict = {}
    for origin, destination, demand in commodity_list:

        if origin not in super_commodity_dict:
            super_commodity_dict[origin] = {}

        if destination not in super_commodity_dict[origin]:
            super_commodity_dict[origin][destination] = 0

        super_commodity_dict[origin][destination] += demand

    arcs = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]
    capacities = [graph[node][neighbor] for node, neighbor in arcs]
    commodities = super_commodity_dict.keys()

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose

    # Create variables
    flow_var = model.addVars(commodities, arcs, obj=0*10**-4, name="flow_var") # flow variables
    overload_var = model.addVars(arcs, obj=1, name="overload_var") # overload variables : we want to minimize their sum

    # Arc capacity constraints :
    model.addConstrs((flow_var.sum('*',node,neighbor) + use_graph[node][neighbor] <= graph[node][neighbor] + overload_var[(node, neighbor)] for node,neighbor in arcs), "cap")
    if proof_constaint:
        model.addConstrs(flow_var.sum('*',node,neighbor) <= flow_upper_bound_graph[node][neighbor] for node, neighbor in arcs)

    # Flow conservation constraints
    for origin in super_commodity_dict:
        for node in range(nb_nodes):

            if node in super_commodity_dict[origin]:
                rhs = -super_commodity_dict[origin][node]
            elif node == origin:
                rhs = sum(super_commodity_dict[origin].values())
            else:
                rhs = 0

            model.addConstr((flow_var.sum(origin,node,'*') - flow_var.sum(origin,'*',node) == rhs), "node{}_{}".format(node, origin))

    model.update()

    if return_model:
        return model, overload_var, flow_var, super_commodity_dict

    # Launching the model
    model.optimize()

    # Getting the results from the solver : the allocation of each super commodity and the total necessary capacity
    if model.status == gurobipy.GRB.Status.OPTIMAL:
        overload_values = model.getAttr('x', overload_var)
        if verbose : print("Result = ", sum(overload_values.values()))
        if proof_constaint:
            remaining_capacity_graph = [{neighbor : min(flow_upper_bound_graph[node][neighbor], graph[node][neighbor] + overload_values[(node, neighbor)] - use_graph[node][neighbor]) for neighbor in graph[node]} for node in range(nb_nodes)]
        else:
            remaining_capacity_graph = [{neighbor : graph[node][neighbor] + overload_values[(node, neighbor)] - use_graph[node][neighbor] for neighbor in graph[node]} for node in range(nb_nodes)]

        allocation_graph_per_origin = {}
        flow_values = model.getAttr('x', flow_var)
        for origin in super_commodity_dict:
            allocation_graph = [{} for node in range(nb_nodes)]
            for node,neighbor in arcs:
                allocation_graph[node][neighbor] = flow_values[origin,node,neighbor]
            allocation_graph_per_origin[origin] = allocation_graph

        return allocation_graph_per_origin, remaining_capacity_graph

    else:
        print("Solver exit status : ", model.status)


def gurobi_congestion_solver(graph, commodity_list, use_graph=None, flow_upper_bound_graph=None, verbose=0, proof_constaint=False, return_model=False):
    # LP program that solves the multicommodity flow problem with the following objective function : minimize the maximum arc overload (i.e. the congestion)

    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)

    if use_graph is None:
        use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]

    # we aggregate the commodities by origin : this create a super commodity
    # this process does not change the results of the continuous solver
    super_commodity_dict = {}
    for origin, destination, demand in commodity_list:

        if origin not in super_commodity_dict:
            super_commodity_dict[origin] = {}

        if destination not in super_commodity_dict[origin]:
            super_commodity_dict[origin][destination] = 0

        super_commodity_dict[origin][destination] += demand

    arcs = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]
    capacities = [graph[node][neighbor] for node, neighbor in arcs]
    commodities = super_commodity_dict.keys()

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose

    # Create variables
    flow_var = model.addVars(commodities, arcs, obj=0, name="flow_var") # flow variables
    overload_var = model.addVar(obj=1, name="overload_var") # overload variable
    overload_var_sum = model.addVars(arcs, obj=0, name="overload_var") # overload variables : we want to minimize their sum

    model.setObjectiveN(sum(overload_var_sum.values()),1)

    # Arc capacity constraints :
    model.addConstrs((flow_var.sum('*',node,neighbor) + use_graph[node][neighbor] <= graph[node][neighbor]* (1 + overload_var_sum[(node, neighbor)]) for node,neighbor in arcs), "cap")
    model.addConstrs((flow_var.sum('*',node,neighbor) + use_graph[node][neighbor] <= graph[node][neighbor] * overload_var for node,neighbor in arcs), "cap")
    if proof_constaint:
        model.addConstrs(flow_var.sum('*',node,neighbor) <= flow_upper_bound_graph[node][neighbor] for node, neighbor in arcs)

    # Flow conservation constraints
    for origin in super_commodity_dict:
        for node in range(nb_nodes):

            if node in super_commodity_dict[origin]:
                rhs = -super_commodity_dict[origin][node]
            elif node == origin:
                rhs = sum(super_commodity_dict[origin].values())
            else:
                rhs = 0

            model.addConstr((flow_var.sum(origin,node,'*') - flow_var.sum(origin,'*',node) == rhs), "node{}_{}".format(node, origin))

    model.update()

    if return_model:
        return model, overload_var, flow_var, super_commodity_dict

    # Launching the model
    model.optimize()

    # Getting the results from the solver : the allocation of each super commodity and the total necessary capacity
    if model.status == gurobipy.GRB.Status.OPTIMAL:
        overload_values = overload_var.X
        if verbose : print("Result = ", overload_values)
        if proof_constaint:
            remaining_capacity_graph = [{neighbor : min(flow_upper_bound_graph[node][neighbor], graph[node][neighbor] * overload_values - use_graph[node][neighbor]) for neighbor in graph[node]} for node in range(nb_nodes)]
        else:
            remaining_capacity_graph = [{neighbor : graph[node][neighbor] * overload_values - use_graph[node][neighbor] for neighbor in graph[node]} for node in range(nb_nodes)]

        allocation_graph_per_origin = {}
        flow_values = model.getAttr('x', flow_var)
        for origin in super_commodity_dict:
            allocation_graph = [{} for node in range(nb_nodes)]
            for node,neighbor in arcs:
                allocation_graph[node][neighbor] = flow_values[origin,node,neighbor]
            allocation_graph_per_origin[origin] = allocation_graph

        return allocation_graph_per_origin, remaining_capacity_graph

    else:
        print("Solver exit status : ", model.status)


if __name__ == "__main__":
    pass
