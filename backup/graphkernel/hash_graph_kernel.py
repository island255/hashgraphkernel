# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
import scipy.sparse as sparse
from sklearn import preprocessing as pre
from auxiliarymethods import dataset_parsers as dp
from auxiliarymethods import auxiliary_methods as aux


def hash_graph_kernel(graph_db, base_kernel, kernel_parameters, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()
    # n = len(graph_db)

    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    colors_0 = np.zeros([num_vertices, dim_attributes])
    offset = 0

    # get seed of locally_sensitive_hashing
    print("get seed of locally_sensitive_hashing")
    sigma = 1.0

    hash_v_set = []
    hash_b_set = []

    for i in range(20):
        # Compute random projection vector
        hash_v = np.random.randn(dim_attributes, 1) * sigma  # / np.random.randn(d, 1)
        hash_v_set.append(hash_v)
        # Compute random offset
        hash_b = 1 * np.random.rand() * sigma
        hash_b_set.append(hash_b)
    # gram_matrix = np.zeros([n, n])
    hash_v = np.random.randn(dim_attributes, 1) * sigma
    hash_b = 1 * np.random.rand() * sigma

    # Get attributes from all graph instances
    graph_indices = []
    for g in graph_db:
        for i, v in enumerate(g.vertices()):
            colors_0[i + offset] = g.vp.na[v]

        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    # Normalize attributes: center to the mean and component wise scale to unit variance
    if scale_attributes:
        # axis=0 means normalize each column which would be different for different groups
        colors_0 = pre.scale(colors_0, axis=0)


    print("getting unified max_1 for further process...")
    colors_hashed = aux.locally_sensitive_hashing(hash_v, hash_b, colors_0, dim_attributes, lsh_bin_width, sigma=sigma)
    max_1 = int(np.amax(colors_hashed) + 1)
    
    print("largest max:" + str(max_1))
    max_1 = [50,1000,5000,10000]

    sub_graph_db = []


    for index, graph in enumerate(graph_db):
        sub_graph_db.append(graph)

        if (index + 1) % 600 == 0 or index == len(graph_db) - 1:
            print("getting feature vectors for set of graphs: " + str(index))
            start_index = index+1-600
            end_index = index
            if index == len(graph_db) - 1:
                start_index = index - index % 600
            start_graph_node_offset = graph_indices[start_index][0]
            end_graph_node_offset = graph_indices[end_index][1]

            print(start_graph_node_offset,end_graph_node_offset+1)

            sub_colors_0 = colors_0[start_graph_node_offset : end_graph_node_offset+1]
            for it in xrange(0, iterations):
                colors_hashed = aux.locally_sensitive_hashing(hash_v_set[it], hash_b_set[it], sub_colors_0,
                                                              dim_attributes, lsh_bin_width, sigma=sigma)

                tmp = base_kernel(max_1, sub_graph_db, colors_hashed, *kernel_parameters)

                if it == 0 and not use_gram_matrices:
                    feature_vectors = tmp
                else:
                    feature_vectors = sparse.hstack((feature_vectors, tmp))

            feature_vectors = feature_vectors.tocsr()

            sub_graph_db = []
            print("writting results")
            dp.save_feature_vectors(feature_vectors, "feature_vectors/sub_vectors_" + str(index))
