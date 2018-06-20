import model
import chainer
from chainer import serializers
import numpy as np


def MLPClassifier(unsmodel, graphs, indexes, rep_dim, batchsize):
    print('start classification')
    split = int(len(indexes) * 0.9)
    graph_train = []
    graph_test = []

    for i in indexes[0:split]:
        graph_train.append(graphs[i])

    for i in indexes[split:len(indexes)]:
        graph_test.append(graphs[i])

    serializers.load_npz(str(rep_dim) + "_model_ptc.npz", unsmodel)

    hid_dim = 150
    out_dim = 2
    mlp = model.MLP(rep_dim, hid_dim, out_dim)
    classifier = model.SoftmaxClassifier(mlp)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classifier)
    n_epochs = 30

    # training phase
    best = 0.00
    for epoch in range(n_epochs):
        print("epoch:", epoch)
        perm = np.random.permutation(len(graph_train))
        N_train = len(graph_train)
        sum_loss = 0
        sum_accuracy = 0
        for i in range(0, N_train, batchsize):
            maxid = min(i + batchsize, N_train)
            graphids = []
            adjs = []
            atom_arrays = []
            labels = []
            for id in perm[i:maxid]:
                graphids.append(graph_train[id][0])
                adjs.append(graph_train[id][2])
                atom_arrays.append(graph_train[id][1])
                labels.append(graph_train[id][3])
            graphids = np.asarray(graphids)
            adjs = np.asarray(adjs, dtype=np.float32)
            atom_arrays = np.asarray(atom_arrays, dtype=np.int32)
            labels = np.asarray(labels, dtype=np.int32)
            rep_list, counts = unsmodel.extract_fp(graphids, adjs, atom_arrays)
            y = chainer.Variable(labels)
            x = rep_list
            optimizer.update(classifier, x, counts, y)
            sum_loss += float(classifier.loss.data) * len(y.data)
            sum_accuracy += float(classifier.accuracy.data) * len(y.data)
        print("train acc:", sum_accuracy / N_train,
              "train loss:", sum_loss / N_train)
        if best < sum_accuracy:
            serializers.save_npz(str(rep_dim) + "_nn_ptc.npz", classifier)
            best = sum_accuracy

    # test
    graphids = []
    adjs = []
    atom_arrays = []
    labels = []
    serializers.load_npz(str(rep_dim) + "_nn_ptc.npz", classifier)
    for id in range(len(graph_test)):
        graphids.append(graph_test[id][0])
        adjs.append(graph_test[id][2])
        atom_arrays.append(graph_test[id][1])
        labels.append(graph_test[id][3])
    graphids = np.asarray(graphids)
    adjs = np.asarray(adjs, dtype=np.float32)
    atom_arrays = np.asarray(atom_arrays, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    rep_list, counts = unsmodel.extract_fp(graphids, adjs, atom_arrays)

    x = rep_list
    y = chainer.Variable(labels)
    print("test acc:", classifier.accuracy.data)
