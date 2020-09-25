import os
import pickle
import scipy.sparse as sparse
from auxiliarymethods import auxiliary_methods as aux
import math as m


def get_graph_file_path(dir_path):
    """
    get paths for feature vectors of several graph groups
    :param dir_path:
    :return:
    """
    paths = []
    for sub_path in os.listdir(dir_path):
        paths.append(os.path.join(dir_path, sub_path))

    for index_x in range(len(paths)):
        for index_y in range(index_x, len(paths)):
            num_x = int(paths[index_x].replace("feature_vectors/sub_vectors_", ""))
            num_y = int(paths[index_y].replace("feature_vectors/sub_vectors_", ""))
            if num_x > num_y:
                temp = paths[index_x]
                paths[index_x] = paths[index_y]
                paths[index_y] = temp

    return paths


def get_pickle_content(paths):
    """
    read pickle files to get feature vectors for all graphs
    :param paths:
    :return:
    """
    feature_vectors = []
    for path in paths:
        pickle_file = open(path, "r")
        feature_vector = pickle.load(pickle_file)
        pickle_file.close()
        feature_vectors = sparse.vstack((feature_vectors, feature_vector))
    return feature_vectors


def write_pickle(feature_vectors, file_path):
    with open(file_path, "w") as f:
        pickle.dump(feature_vectors, f)
    f.close()


def read_pickle(file_path):
    with open(file_path, "r") as f:
        graph_vectors = pickle.load(f)
    f.close()
    return graph_vectors


def read_graph_names(file_name):
    graph_names = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            graph_names.append(line.strip("/").split("/"))
    return graph_names


def main():
    print("getting gram_matrix ...")
    if os.path.exists("graph_vectors"):
        print("load graph_vectors from pickle")
        graph_vectors = read_pickle("graph_vectors")
    else:
        print("read graph vectors and aggregate to one pickle")
        feature_vector_paths = get_graph_file_path("feature_vectors")
        graph_vectors = get_pickle_content(feature_vector_paths)
        print(type(graph_vectors))
        write_pickle(graph_vectors, "graph_vectors")

    print("calculate gram matrix for graph vectors")
    iterations = 20
    feature_vectors = m.sqrt(1.0 / iterations) * graph_vectors
    # gram_matrix = feature_vectors.dot(feature_vectors.T)
    # gram_matrix = gram_matrix.toarray()
    # normalize_gram_matrix = True
    # if normalize_gram_matrix:
    #     gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    print("save gram matrix as pickle")
    # write_pickle(gram_matrix, "gram_matrix")


if __name__ == '__main__':
    main()
