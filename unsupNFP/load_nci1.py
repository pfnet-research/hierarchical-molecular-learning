import numpy as np
from rdkit.Chem import MolFromSmiles
from scipy.io import loadmat


MAX_EDGE_TYPE = 4
MAX_NUMBER_ATOM = 120


def construct_edge_matrix_from(mol):
    if mol is None:
        return None
    N = mol.GetNumAtoms()
    size = MAX_NUMBER_ATOM
    adjs = np.zeros((4, size, size), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            bond = mol.GetBondBetweenAtoms(i, j)  # type: Chem.Bond
            if bond is not None:
                bondType = str(bond.GetBondType())
                if bondType == 'SINGLE':
                    adjs[0, i, j] = 1.0
                elif bondType == 'DOUBLE':
                    adjs[1, i, j] = 1.0
                elif bondType == 'TRIPLE':
                    adjs[2, i, j] = 1.0
                elif bondType == 'AROMATIC':
                    adjs[3, i, j] = 1.0
                else:
                    print("[ERROR] Unknown bond type", bondType)
                    assert False  # Should not come here
    return adjs


def getAtom2id(graphs):
    max_atom = 0
    for graph in graphs:
        atom_list = graph[1]
        max_atom = max(max_atom, len(atom_list))
    assert max_atom <= MAX_NUMBER_ATOM
    atom2id = {'empty': 0}
    atoms = [graph[1] for graph in graphs]
    atoms = sum(atoms, [])
    for a in atoms:
        if a not in atom2id:
            atom2id[a] = len(atom2id)
    print(atom2id)
    return atom2id


def load_nci1_file(filename):
    """

    :param filename:
    :return: list of graph, each consists of graphID,
             list of atoms, list of edges

    """
    inputdata = loadmat(filename)
    data = inputdata['NCI1']
    labels = inputdata['lnci1'].tolist()
    atom_lists = data["nl"][0]
    edge_lists = data["el"][0]
    adj_lists = data["al"][0]

    size = len(atom_lists)
    graphs = []
    for i in range(size):
        graphId = i
        tmp = atom_lists[graphId][0][0][0]
        atom_list = [a[0] for a in tmp]
        tmp = edge_lists[graphId][0][0][0]
        edge_list = []
        for e in tmp:
            e_l = e[0].tolist()
            edge_list.append(e_l)

        tmp = adj_lists[graphId]
        adj_list = [a[0].tolist() for a in tmp]
        label = labels[graphId][0]
        graphs.append((graphId, atom_list, adj_list, edge_list, label))
    return graphs


def convert_graph(graphs):
    n_graphs = len(graphs)
    edge2id = {'empty': 0}
    all_edges = []
    for graph in graphs:
        edge_list = graph[2]
        for edge in edge_list:
            all_edges.append(edge)
    all_edges = sum(all_edges, [])
    print(all_edges)

    for i in range(n_graphs):
        for j in range(len(graphs[i][2])):
            edge_id = graphs[i][2][j][2]
            graphs[i][2][j][2] = edge2id[str(edge_id)]
    print("MAX NUMBER of EDGES", len(edge2id))
    return graphs, edge2id


def construct_edge_matrix(graph):
    atom_list = graph[1]
    adj_list = graph[2]
    edge_list = graph[3]

    N = len(atom_list)  # number of atoms in the molecule
    size = MAX_NUMBER_ATOM
    adjs = np.zeros((MAX_EDGE_TYPE, size, size), dtype=np.float32)

    for i in range(N):
        node1 = i
        adj_atoms = adj_list[i]  # [4,5,6]
        if len(adj_atoms) == 0:
            continue
        edge_labels = edge_list[i]  # [1,1,2]
        n_adj = len(adj_atoms)
        for j in range(n_adj):
            node2 = adj_atoms[0][j]
            edge_type = edge_labels[0][j]
            adjs[edge_type - 1, node1, node2] = 1.0
    return adjs


def load_whole_data(filename):
    results = []
    graphs = load_nci1_file(filename)
    for graph in graphs:
        graphid = graph[0]
        atom_array = np.zeros((MAX_NUMBER_ATOM,), dtype=np.int32)
        atom_list = graph[1]
        natoms = len(atom_list)
        atom_array[:natoms] = np.array(atom_list)
        label = graph[4]
        adjs = construct_edge_matrix(graph)
        results.append((graphid, atom_array, adjs, label))
    return results


def check_graph(graph):
    print("graphID:", graph[0])
    print("atom list:", graph[1])
    print("adj :", graph[2][1][1])
    print("label:", graph[3])
    print('size of atom list:', len(graph[1]))


def convert_graph_1(graphs, atom2id):
    ret = []
    for graph in graphs:
        (id, atom_list, adj, label) = graph
        atom_list = [atom2id[a] for a in atom_list]
        n_atom = len(atom_list)
        atom_array = np.zeros((MAX_NUMBER_ATOM,), dtype=np.int32)
        atom_array[:n_atom] = np.array(atom_list)

        ret.append((id, atom_array, adj, label))
    return ret


def load_ptc(smile_file, result_file):
    filtered = []
    valid_list = ['MR=P', 'MR=CE', 'MR=SE', 'MR=NE', 'MR=N']
    f_smile = open(smile_file, 'r')
    f_result = open(result_file, 'r')
    smiles = []
    labels = []
    for line in f_smile:
        smile = line.split()[1]
        smiles.append(smile)
    for line in f_result:
        words = line.split(',')
        data = words[0]
        label = data.split()[1]
        labels.append(label)

    for i in range(len(smiles)):
        smile = smiles[i]
        label = labels[i]
        if label not in valid_list:
            continue
        if label in ['MR=P', 'MR=CE', 'MR=SE']:
            label = 1
        else:
            label = 0
        filtered.append((smile, label))

    graphs = []
    id = 0
    for data in filtered:
        smile = data[0]
        label = data[1]
        mol = MolFromSmiles(str(smile))
        if mol is None:
            continue
        adj = construct_edge_matrix_from(mol)
        atom_list = [a.GetSymbol() for a in mol.GetAtoms()]
        graphs.append((id, atom_list, adj, label))
        id += 1
    atom2id = getAtom2id(graphs)
    graphs = convert_graph_1(graphs, atom2id)
    return graphs
