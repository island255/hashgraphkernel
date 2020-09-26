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
        # print(type(pickle_file))
        feature_vectors = sparse.vstack((feature_vectors, feature_vector), format="csr")
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
            graph_names.append(line.split(",")[-1].strip("/").strip("\n").split("/"))
    return graph_names


def separate_vectors_by_names(graph_names, graph_vectors):
    """
    separate graph names and vectors according their projects
    :param graph_names:
    :param graph_vectors:
    :return:
    """

    print("start separating ...")
    graph_names_separate = []
    graph_vectors_separate = []

    print(len(graph_names), graph_vectors._shape)
    for i in range(len(graph_names)):
        start_index = 0
        end_index = 0
        if i != 0 and (graph_names[i][0] != graph_names[i - 1][0] or graph_names[i][1] != graph_names[i - 1][1]):
            print("separating num: " + str(i))
            print ("separating name: " + graph_names[i][0] + "-" + graph_names[i][1])
            end_index = i
            graph_names_separate.append(graph_names[start_index: end_index])
            graph_vectors_separate.append(graph_vectors[start_index: end_index])
            start_index = i

        if i == len(graph_names) - 1:
            end_index = len(graph_names)
            graph_names_separate.append(graph_names[start_index: end_index])
            graph_vectors_separate.append(graph_vectors[start_index: end_index])

        if i % 1000 == 0:
            print(i)

    return graph_names_separate, graph_vectors_separate


def write_similarity_results(gram_matrix, graph_name_pair):
    """
    writing results of similarities between groups
    :param gram_matrix:
    :param graph_name_pair:
    :return:
    """
    filename = graph_name_pair[0][0][0] + "_" + graph_name_pair[0][0][1] + "-" + graph_name_pair[1][0][0] + "_" + \
               graph_name_pair[1][0][1]
    dir_name = "similarities"
    if os.path.exists(dir_name) is False:
        os.mkdir(dir_name)
    with open(dir_name + "/" + filename, "w") as f:
        graph_name_pair_0 = ""
        for i in range(len(graph_name_pair[0])):
            graph_name_pair_0 = graph_name_pair_0 + "_".join(graph_name_pair[0][i]) + ","
        graph_name_pair_0.strip(",")
        graph_name_pair_1 = ""
        for i in range(len(graph_name_pair[1])):
            graph_name_pair_1 = graph_name_pair_1 + "_".join(graph_name_pair[1][i]) + ","
        graph_name_pair_1.strip(",")

        f.write(graph_name_pair_0 + "\n")
        f.write(graph_name_pair_1 + "\n")
        for line in gram_matrix:
            f.write(",".join(line) + "\n")


def calculate_similarities(graph_names_separate, graph_vectors_separate):
    """
    calculate similarities between groups
    :param graph_names_separate:
    :param graph_vectors_separate:
    :return:
    """
    for i in range(len(graph_names_separate)):
        for j in range(len(graph_names_separate)):
            graph_name_pair = (graph_names_separate[i], graph_names_separate[j])
            print ("process groups of " + graph_name_pair[0][0][0] + "_" + graph_name_pair[0][0][1] + "-" +
                   graph_name_pair[1][0][0] + "_" + graph_name_pair[1][0][1])
            gram_matrix = graph_vectors_separate[i].dot(graph_vectors_separate[j].T)
            gram_matrix = gram_matrix.toarray()
            gram_matrix = aux.normalize_gram_matrix(gram_matrix)

            write_similarity_results(gram_matrix, graph_name_pair)


def main():
    print("getting gram_matrix ...")
    if os.path.exists("graph_vectors"):
        print("load graph_vectors from pickle")
        graph_vectors = read_pickle("graph_vectors")
    else:
        print("read graph vectors and aggregate to one pickle")
        feature_vector_paths = get_graph_file_path("feature_vectors")
        graph_vectors = get_pickle_content(feature_vector_paths)
        iterations = 20
        graph_vectors = m.sqrt(1.0 / iterations) * graph_vectors
        # print("convert coo_matrix to csr_matrix")
        # graph_vectors = graph_vectors.tocsr()
        write_pickle(graph_vectors, "graph_vectors")

    print(type(graph_vectors))
    print("calculate similarity matrix for graph vectors")

    graph_names = read_graph_names("LLVM_IR_to_Graph.txt")
    print(graph_names[0])

    print("separate vectors according their groups")
    graph_names_separate, graph_vectors_separate = separate_vectors_by_names(graph_names, graph_vectors)

    print("calculate similarities between groups")
    calculate_similarities(graph_names_separate, graph_vectors_separate)


if __name__ == '__main__':
    main()
