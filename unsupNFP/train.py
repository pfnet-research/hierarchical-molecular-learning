import argparse

from chainer import optimizers
from chainer import serializers
import numpy as np

import model
import load_mutag
import load_nci1
import classification


n_epoch = 200
n_parts = 5

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=('mutag', 'ptc'))
args = parser.parse_args()

if args.dataset == 'mutag':
    mutag_file_name = "MUTAG.mat"
    graphs = load_mutag.load_whole_data('MUTAG.mat')
    MAX_EDGE_TYPE = load_mutag.MAX_EDGE_TYPE
    MAX_NUMBER_ATOM = load_mutag.MAX_NUMBER_ATOM
elif args.dataset == 'ptc':
    smile_filename = 'corrected_smiles.txt'
    result_filename = 'corrected_results.txt'
    graphs = load_nci1.load_ptc(smile_filename, result_filename)
    MAX_EDGE_TYPE = load_nci1.MAX_EDGE_TYPE
    MAX_NUMBER_ATOM = load_nci1.MAX_NUMBER_ATOM
else:
    raise ValueError('Invalid dataset type: {}'.format(args.dataset))

model.MAX_EDGE_TYPE = MAX_EDGE_TYPE
model.MAX_NUMBER_ATOM = MAX_NUMBER_ATOM

indexs_test = np.random.permutation(len(graphs))
n_graphs = len(graphs)
print("num of graphs:", n_graphs)


rep_dim = 101
max_degree = 5
num_levels = 6
neg_size = 10
batchsize = 100

hid_dim = 100
out_dim = 2

softmax = model.SoftmaxCrossEntropy(rep_dim, MAX_NUMBER_ATOM)
print("[CONFIG: representation dim =", rep_dim, "]")
atom2vec = model.Atom2vec(MAX_NUMBER_ATOM, rep_dim, max_degree, softmax)
model = model.Mol2Vec(len(graphs), rep_dim, max_degree,
                      num_levels, neg_size, atom2vec)

optimizer = optimizers.Adam()
optimizer.setup(model)
print("start training")
for epoch in range(1, n_epoch + 1):
    print("epoch:", epoch)
    indexes = np.random.permutation(len(graphs))
    sum_loss = 0

    for i in range(0, n_graphs, batchsize):
        maxid = min(i + batchsize, n_graphs)
        ids = indexes[i:maxid]

        graphids = []
        adjs = []
        atom_arrays = []
        for id in indexes[i:maxid]:
            graphids.append(graphs[id][0])
            # index 1 and 2 need to be changed for MUTAG or NCI1 datasets
            atom_arrays.append(graphs[id][1])
            adjs.append(graphs[id][2])

        graphids = np.asarray(graphids)
        adjs = np.asarray(adjs, dtype=np.float32)
        atom_arrays = np.asarray(atom_arrays, dtype=np.int32)
        optimizer.update(model, graphids, adjs, atom_arrays)

        sum_loss += float(model.loss.data) * len(graphids)
        print("-----", float(model.loss.data) * len(graphids))
    print("loss: ", sum_loss / n_graphs)
    serializers.save_npz(str(rep_dim) + "_model_ptc.npz", model)

    # after each epcoh, check result
    if epoch % 10 == 0:
        classification.MLPClassifier(model, graphs, indexs_test,
                                     rep_dim, batchsize)
