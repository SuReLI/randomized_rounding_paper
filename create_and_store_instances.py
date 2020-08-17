import random
import numpy as np
import time
import pickle


from instance_mcnf import generate_instance


nb_repetitions = 100
nb_unique_exp = 10

# Size of the graph : controls the number of nodes and arcs
# size_list = [15]*nb_unique_exp
size_list = [5, 7, 8, 10, 12, 13, 14, 16, 18, 20]
# size_list = [3, 4, 5, 6, 7, 9, 10, 12, 13, 15]
# size_list = [30, 50, 70, 100, 130, 160, 200, 250, 300, 400]
size_list = np.array(size_list)

# Capacity of the arcs of the graph
# capacity_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
capacity_list = [10000] * nb_unique_exp
# capacity_list = [3] * nb_unique_exp

# Upper bound on the size of the commodities
max_demand_list = [1500] * nb_unique_exp
# max_demand_list = [2] * nb_unique_exp
# max_demand_list = [int(np.sqrt(capa)) for capa in capacity_list]

# Choose here the type of graph to be created: note that the size parameter does not have the same meaning for both types
instance_parameter_list = []
for size, capacity in zip(size_list, capacity_list):
    instance_parameter_list.append(("grid", (size, size, size, 2*size, capacity, capacity)))
# for size, capacity in zip(size_list, capacity_list):
    # instance_parameter_list += [("random_connected", (size, 5/size, 0.1, capacity))]

# Complete with the path to the main directory
global_path = "/home/francois/Desktop/"
global_path = None
assert global_path, "Complete global_path with the path to the main directory"

# Complete name of the directory that will contain the instances
experience_string = "graph_scaling_dataset/"

i = -1
instance_name_list = []
for graph_type, graph_generator_inputs in instance_parameter_list:
    i += 1
    for j in range(nb_repetitions):

        # Generate the graph and the commodity list
        graph, commodity_list, initial_solution = generate_instance(graph_type, graph_generator_inputs, max_demand=max_demand_list[i])

        graph_type = instance_parameter_list[i][0]
        if graph_type == "grid":
            size = size_list[i]
            nb_nodes = size ** 2 + size
        elif graph_type == "random_connected":
            nb_nodes = size_list[i]

        instance_name = graph_type + "_" + str(nb_nodes) + "_" + str(capacity_list[i]) + "_" + str(max_demand_list[i]) + "_" + str(j)

        # Store the created instance
        instance_file = open(global_path + "randomized_rounding_paper_code/instance_files/" + experience_string + instance_name + ".p", "wb" )
        pickle.dump((graph, commodity_list), instance_file)
        instance_file.close()

        instance_name_list.append(instance_name)

# Create a file containing the name of all the instances
instance_name_file = open(global_path + "randomized_rounding_paper_code/instance_files/" + experience_string + "instance_name_file.p", "wb" )
pickle.dump(instance_name_list, instance_name_file)
instance_name_file.close()
