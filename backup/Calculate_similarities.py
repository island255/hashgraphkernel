import os
import pickle
import scipy.sparse as sparse


def get_graph_file_path(dir_path):
    """
    get paths for feature vectors of several graph groups
    :param dir_path:
    :return:
    """
    paths = []
    for sub_path in os.listdir(dir_path):
        paths.append(os.path.join(dir_path, sub_path))
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
        feature_vectors = sparse.vstack(feature_vectors, feature_vector)
    return feature_vectors


def write_pickle_of_all_graphs(feature_vectors, file_path):
    with open(file_path, "w") as f:
        pickle.dump(feature_vectors, f)
    f.close()


def read_pickle_of_all_graphs(file_path):
    with open(file_path, "r") as f:
        graph_vectors = pickle.load(f)
    f.close()
    return graph_vectors


def main():
    if os.path.exists("graph_vectors"):
        graph_vectors = read_pickle_of_all_graphs("graph_vectors")
    else:
        print("read graph vectors and aggregate to one pickle")
        feature_vector_paths = get_graph_file_path("feature_vectors")
        graph_vectors = get_pickle_content(feature_vector_paths)
        print(len(graph_vectors), len(graph_vectors[0]))
        write_pickle_of_all_graphs(graph_vectors, "graph_vectors")


if __name__ == '__main__':
    main()
