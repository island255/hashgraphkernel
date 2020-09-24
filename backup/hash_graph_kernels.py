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


    rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel, kernel_parameters_wl, 20, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
          
    # Compute gram matrix for HGK-SP

    # Normalize gram matrix
    # gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Write out LIBSVM matrix
    


if __name__ == "__main__":
    main()
