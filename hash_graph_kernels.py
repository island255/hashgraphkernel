# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl

import os
import pickle
import numpy as np


def main():
    # Load ENZYMES data set
    
    if not os.path.exists("graph_db"):
        print("reading graphs from data folder ...")
        graph_db  = dp.read_txt("Matrix")
        graph_db_file = open("graph_db","w")
        pickle.dump(graph_db,graph_db_file)
    else:
        print("loading graphs from pickle ...")
        graph_db_file = open("graph_db","r")
        graph_db = pickle.load(graph_db_file)

    # Parameters used: 
    # Compute gram matrix: False, 
    # Normalize gram matrix: False
    # Use discrete labels: False
    # kernel_parameters_sp = [False, False, 0]

    # Parameters used: 
    # Compute gram matrix: False, 
    # Normalize gram matrix: False
    # Use discrete labels: False
    # Number of iterations for WL: 3
    kernel_parameters_wl = [3, False, False, 0]

    sub_graph_db = []

    feature_vectors = []


    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    sigma=1.0

    hash_v_set = []
    hash_b_set = []

    for i in range(20):
            
        # Compute random projection vector
        hash_v = np.random.randn(dim_attributes, 1) * sigma  # / np.random.randn(d, 1)
        hash_v_set.append(hash_v)
        # Compute random offset
        hash_b = 1 * np.random.rand() * sigma
        hash_b_set.append(hash_b)



    for index, graph in enumerate(graph_db):
        sub_graph_db.append(graph)

        
        if (index+1) % 600 == 0 or index == len(graph_db) - 1:
            print("getting feature vectors for set of graphs: "+str(index))
            feature_vectors = rbk.hash_graph_kernel(hash_v_set,hash_b_set,sub_graph_db, wl.weisfeiler_lehman_subtree_kernel,
                                                     kernel_parameters_wl, 20,
                                                     scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
            sub_graph_db = []
            dp.save_feature_vectors(feature_vectors, "feature_vectors/sub_vectors_"+str(index))


    # Compute gram matrix for HGK-SP

    # Normalize gram matrix
    # gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Write out LIBSVM matrix
    


if __name__ == "__main__":
    main()
