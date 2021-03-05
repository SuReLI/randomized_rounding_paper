This read_me describes how to read the instances of the unsplittable flow problem contained in this directory.
The instances were created using Python and stored using the module pickle.

To read an instance contained in a file whose path is noted path_to_file, use the following Python instructions :
import pickle
file = open(path_to_file, "rb" )
graph, commodity_list = pickle.load(file)
file.close()

This will create two Python object that you can now use:
 - graph : this object is a list containing one dictionnary per node, the keys of the dictionnary at index i are indices of the neighbors of the node i. The value corresponding to the neighbor j is a number describing the capacity of the arc (i, j)
 - commodity_list : this object is a list containing one tuple per commodity, each tuple contains (origin, destination, demand) for a commodity.


The name of the files can be interpreted as follow. A file name "string_n_x_y_i" means:
 - string is the type of graph of the instance
 - n is the number of node of the graph
 - x is the capacity of every arc in the graph
 - y is a upper-bound on the demand of the commodities
 - i is an index used to differentiate instances created with the same parameters


For a more complete description of the procedure used to create these instances see our article : Randomized rounding algorithms for large scale unsplittable flow problems
