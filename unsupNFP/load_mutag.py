from scipy.io import loadmat
import numpy as np


MAX_EDGE_TYPE = 12
MAX_NUMBER_ATOM = 120


def load_mutag_file(filename):
    """

    :param filename:
    :return: list of graph, each consists of graphID,
             list of atoms, list of edges

    """
    inputdata = loadmat("MUTAG.mat")
    data = inputdata["MUTAG"]
    labels = inputdata["lmutag"].tolist()
    atom_lists = data["nl"][0]
    edge_lists = data["el"][0]

    size = len(atom_lists)
    graphs = []
    for i in range(size):
        graphId = i
        tmp = atom_lists[i][0][0][0]
        atom_list = [a[0] for a in tmp]
        tmp = edge_lists[i][0][0][0]
        edge_list = [e.tolist() for e in tmp]
        label = labels[i][0]
        if label == 1:
            label = 1
        else:
            label = 0
        graphs.append((graphId, atom_list, edge_list, label))
    return graphs


def convert_graph(graphs):
    n_graphs = len(graphs)
    edge2id = {'empty': 0}
    for graph in graphs:
        edge_list = graph[2]
        for edge in edge_list:
            edge_id = str(edge[2])
            if edge_id not in edge2id:
                edge2id[edge_id] = len(edge2id)

    for i in range(n_graphs):
        for j in range(len(graphs[i][2])):
            edge_id = graphs[i][2][j][2]
            graphs[i][2][j][2] = edge2id[str(edge_id)]
    print("MAX NUMBER of EDGES", len(edge2id))
    return graphs, edge2id


def construct_edge_matrix(graph):
    edge_list = graph[2]

    size = MAX_NUMBER_ATOM
    adjs = np.zeros((MAX_EDGE_TYPE, size, size), dtype=np.float32)

    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        edge_type = edge[2]
        adjs[edge_type, node1, node2] = 1.0
        adjs[edge_type, node2, node1] = 1.0
    return adjs


def load_whole_data(filename):
    results = []
    graphs = load_mutag_file(filename)
    graphs, edge2id = convert_graph(graphs)
    for graph in graphs:
        graphid = graph[0]
        atom_array = np.zeros((MAX_NUMBER_ATOM,), dtype=np.int32)
        atom_list = graph[1]
        natoms = len(atom_list)
        atom_array[:natoms] = np.array(atom_list)
        label = graph[3]
        adjs = construct_edge_matrix(graph)
        results.append((graphid, atom_array, adjs, label))
    return results


def check_graph(graph):
    print("graphID:", graph[0])
    print("atom list:", graph[1])
    print("adjs:", graph[2])
    print("label:", graph[3])
