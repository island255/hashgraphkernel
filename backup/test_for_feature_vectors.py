import os
import pickle
import scipy.sparse as sparse
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
        # feature_vector = sparse.csr_matrix(feature_vector)
        pickle_file.close()
        print(type(feature_vector))
        print(feature_vector[0])
        print(feature_vector._shape)
        # print(len(pickle_file))
        feature_vectors = sparse.vstack((feature_vectors, feature_vector),format="csr")
        print(type(feature_vectors))
    return feature_vectors


def main():
    print("read graph vectors and aggregate to one pickle")
    feature_vector_paths = get_graph_file_path("feature_vectors")
    graph_vectors = get_pickle_content(feature_vector_paths)
    print(type(graph_vectors),graph_vectors._shape)



if __name__ == '__main__':
    main()
