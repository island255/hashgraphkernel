# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
from numba import jit, cuda 


@jit
def normalize_gram_matrix(graph_vectors_A, graph_vectors_B, gram_matrix):
    n = gram_matrix.shape[0]
    p = gram_matrix.shape[1]
    gram_matrix_norm = np.zeros([n, p], dtype=np.float64)
    gram_matrix_A = np.zeros(n, dtype=np.float64)
    gram_matrix_B = np.zeros(p, dtype=np.float64)
    graph_vectors_A = graph_vectors_A.tocsr()
    graph_vectors_B = graph_vectors_B.tocsr()

    # print(gram_matrix_norm._shape)


    for i in xrange(0 ,n):
        
        # print(graph_vectors_A[i]._shape)
        # print((graph_vectors_A[i].dot(graph_vectors_A[i].T))._shape)
        # print((graph_vectors_A[i].dot(graph_vectors_A[i].T)))
        # print((graph_vectors_A[i].dot(graph_vectors_A[i].T)).toarray()[0][0])

        gram_matrix_A[i] = (graph_vectors_A[i].dot(graph_vectors_A[i].T)).toarray()[0][0]

    for j in xrange(0, p):
        gram_matrix_B[j] = (graph_vectors_B[j].dot(graph_vectors_B[j].T)).toarray()[0][0]

    for i in xrange(0, n):
        for j in xrange(0, p):
            if not (gram_matrix_A[i] == 0.0 or gram_matrix_B[j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix_A[i] * gram_matrix_B[j])
                gram_matrix_norm[i][j] = g
                # gram_matrix_norm[j][i] = g

    return gram_matrix_norm


@jit
def locally_sensitive_hashing(v,b,m, d, w, sigma=1.0):
    # # Compute random projection vector
    # v = np.random.randn(d, 1) * sigma  # / np.random.randn(d, 1)

    # # Compute random offset
    # b = w * np.random.rand() * sigma

    # Compute hashes
    labels = np.floor((np.dot(m, v) + b) / w)

    # Compute label
    _, indices = np.unique(labels, return_inverse=True)

    return indices
