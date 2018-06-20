import chainer
import chainer.functions as F
import chainer.links as L
from chainer import ChainList
import numpy as np
import six


global MAX_EDGE_TYPE
global MAX_NUMBER_ATOM
MAX_EDGE_TYPE = None
MAX_NUMBER_ATOM = None
# max atom type for NCI1 must be large
MAX_ATOM_TYPE = 50


class SoftmaxCrossEntropy(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropy, self).__init__()
        with self.init_scope():
            self.out = L.Linear(n_in, n_out, initialW=0)

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


class Atom2vec(chainer.Chain):
    def __init__(self, n_atoms, rep_dim, max_degree, loss_func):
        super(Atom2vec, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.atom_embed = L.EmbedID(n_atoms, rep_dim)
            self.hidden_weights = chainer.ChainList(
                *[L.Linear(rep_dim, rep_dim)
                  for _ in range(num_degree_type)]
            )
            self.edge_layer = L.Linear(rep_dim, MAX_EDGE_TYPE * rep_dim)
            self.loss_func = loss_func
        self.max_degree = num_degree_type
        self.rep_dim = rep_dim

    def __call__(self, adj, atom_array):
        counts = []
        for list_atom in atom_array:
            list_atom = np.array(list_atom)
            count = np.count_nonzero(list_atom)
            counts.append(count)
        x = self.atom_embed(atom_array)
        degree_mat = F.sum(adj, axis=1)
        degree_mat = F.sum(degree_mat, axis=1)

        s0, s1, s2 = x.shape
        m = F.reshape(self.edge_layer(F.reshape(x, (s0 * s1, s2))),
                      (s0, s1, s2, MAX_EDGE_TYPE))
        m = F.transpose(m, (0, 3, 1, 2))
        adj = F.reshape(adj, (s0 * MAX_EDGE_TYPE, s1, s1))
        m = F.reshape(m, (s0 * MAX_EDGE_TYPE, s1, s2))
        m = F.batch_matmul(adj, m)
        m = F.reshape(m, (s0, MAX_EDGE_TYPE, s1, s2))
        m = F.sum(m, axis=1)
        s0, s1, s2 = m.shape
        m = F.reshape(m, (s0 * s1, s2))

        atom_array = np.asarray(atom_array).reshape(-1)
        for s_index in range(s0):
            atom_array[counts[s_index] + s_index * s1:(s_index + 1) * s1] = -1

        t = chainer.Variable(atom_array)
        self.loss = self.loss_func(m, t)
        return self.loss


class MLP(chainer.Chain):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.softmax = L.Linear(None, n_hid)
            self.l1 = L.Linear(None, n_hid)
            self.l2 = L.Linear(None, n_hid)
            self.l3 = L.Linear(None, n_out)
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out

    def __call__(self, x_list, counts):
        h = 0
        for x in x_list:
            s0, s1, s2 = x.shape
            x = F.reshape(x, (s0 * s1, s2))
            dh = self.softmax(x)
            dh = F.softmax(dh)
            for s_index in range(s0):
                _from = counts[s_index] + s_index * s1
                _to = (s_index + 1) * s1
                dh.data[_from:_to, :] = 0.0
            dh = F.sum(F.reshape(dh, (s0, s1, self.n_hid)), axis=1)
            h += dh
        # got h
        h = F.relu(self.l1(h))
        y = self.l3(h)
        return y


class SoftmaxClassifier(chainer.Chain):
    def __init__(self, predictor):
        super(SoftmaxClassifier, self).__init__(predictor=predictor)

    def __call__(self, x, counts, t, test=False):
        y = self.predictor(x, counts)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss


class Mol2Vec2(chainer.Chain):
    def __init__(self, num_molecules, rep_dim, max_degree, num_levels):
        super(Mol2Vec2, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.mol_embed_layer = L.EmbedID(num_molecules, rep_dim)
            self.atom_embed_layer = L.EmbedID(MAX_NUMBER_ATOM, rep_dim)
            self.edge_layer = L.Linear(rep_dim, rep_dim * MAX_EDGE_TYPE)
            self.out = L.Linear(rep_dim, MAX_ATOM_TYPE)
            self.H = ChainList(*[ChainList(
                *[L.Linear(rep_dim, rep_dim)
                  for i in six.moves.range(num_degree_type)])
                                 for j in six.moves.range(num_levels)])
        # representation dim of molecules, substructures and atoms
        self.rep_dim = rep_dim
        self.max_degree_type = num_degree_type
        self.num_mol = num_molecules
        self.n_levels = num_levels

    def one_step(self, mol_reps, sub_reps, adj, atom_array, counts):
        s0, s1, s2 = sub_reps.shape
        tmp = self.edge_layer(F.reshape(sub_reps, (s0 * s1, s2)))
        m = F.reshape(tmp, (s0, s1, s2, MAX_EDGE_TYPE))
        m = F.transpose(m, (0, 3, 1, 2))
        m = F.reshape(m, (s0 * MAX_EDGE_TYPE, s1, s2))
        adj = F.reshape(adj, (s0 * MAX_EDGE_TYPE, s1, s1))
        m = F.batch_matmul(adj, m)
        m = F.reshape(m, (s0, MAX_EDGE_TYPE, s1, s2))
        m = F.sum(m, axis=1)
        s0, s1, s2 = m.shape
        m = F.reshape(m, (s0 * s1, s2))
        mol_reps = F.tile(mol_reps, (1, s1))
        mol_reps = F.reshape(mol_reps, (s0 * s1, s2))
        reps = mol_reps + m
        atom_array = atom_array.flatten()
        for s_index in range(s0):
            _from = counts[s_index] + s_index * s1
            _to = (s_index + 1) * s1
            reps.data[_from:_to, :] = 0.0
            atom_array[_from:_to] = -1

        t = chainer.Variable(atom_array)
        loss = F.softmax_cross_entropy(self.out(reps), t)
        return loss

    def message_and_update(self, cur, adj, deg_conds, counts, level):

        s0, s1, s2 = cur.shape
        tmp = self.edge_layer(F.reshape(cur, (s0 * s1, s2)))
        m = F.reshape(tmp, (s0, s1, s2, MAX_EDGE_TYPE))
        m = F.transpose(m, (0, 3, 1, 2))
        m = F.reshape(m, (s0 * MAX_EDGE_TYPE, s1, s2))
        adj = F.reshape(adj, (s0 * MAX_EDGE_TYPE, s1, s1))
        m = F.batch_matmul(adj, m)

        m = F.reshape(m, (s0, MAX_EDGE_TYPE, s1, s2))
        m = F.sum(m, axis=1)
        m = m + cur
        s0, s1, s2 = m.shape
        zero_array = np.zeros(m.shape, dtype=np.float32)
        ms = [F.reshape(F.where(cond, m, zero_array), (s0 * s1, s2))
              for cond in deg_conds]
        out_x = 0
        for hidden_weight, m in zip(self.H[level], ms):
            out_x = out_x + hidden_weight(m)
        out_x = F.sigmoid(out_x)
        for s_index in range(s0):
            _from = counts[s_index] + s_index * s1
            _to = (s_index + 1) * s1
            out_x.data[_from:_to, :] = 0.0

        out_x = F.reshape(out_x, (s0, s1, s2))
        return out_x

    def __call__(self, ids, adj, atom_array):
        ids = np.array(ids, dtype=np.int32)
        counts = []
        for list_atom in atom_array:
            list_atom = np.array(list_atom)
            count = np.count_nonzero(list_atom)
            counts.append(count)

        mol_rep = self.mol_embed_layer(ids)
        sub_rep = self.atom_embed_layer(atom_array)
        degree_mat = F.sum(adj, axis=1)
        degree_mat = F.sum(degree_mat, axis=1)
        deg_conds = [np.broadcast_to(
            ((degree_mat - degree).data == 0)[:, :, None],
            sub_rep.shape)
                     for degree in range(1, self.max_degree_type + 1)]

        self.loss = 0
        for level in range(self.n_levels):
            loss = self.one_step(mol_rep, sub_rep, adj, atom_array, counts)
            sub_rep = self.message_and_update(
                sub_rep, adj, deg_conds, counts, level)
            self.loss += loss
        return self.loss


class Mol2Vec(chainer.Chain):
    def __init__(self, num_molecules, rep_dim, max_degree,
                 num_levels, neg_size, atom2vec):
        super(Mol2Vec, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.gate_weight = L.Linear(rep_dim, 1)
            self.mol_embed_layer = L.EmbedID(num_molecules, rep_dim)
            self.atom_embed_layer = L.EmbedID(MAX_ATOM_TYPE, rep_dim)
            self.edge_layer = L.Linear(rep_dim, rep_dim * MAX_EDGE_TYPE)
            self.H = ChainList(*[ChainList(
                *[L.Linear(rep_dim, rep_dim)
                  for i in six.moves.range(num_degree_type)])
                                 for j in six.moves.range(num_levels)])
        # representation dim of molecules, substructures and atoms
        self.rep_dim = rep_dim
        self.max_degree_type = num_degree_type
        self.num_mol = num_molecules
        self.neg_size = neg_size
        self.n_levels = num_levels
        self.atom2vec = atom2vec

    def extract_fp(self, ids, adj, atom_array):
        counts = []
        for list_atom in atom_array:
            list_atom = np.array(list_atom)
            count = np.count_nonzero(list_atom)
            counts.append(count)

        out = []
        sub_rep = self.atom_embed_layer(atom_array)
        out.append(sub_rep)

        degree_mat = F.sum(adj, axis=1)
        degree_mat = F.sum(degree_mat, axis=1)
        deg_conds = [np.broadcast_to(
            ((degree_mat - degree).data == 0)[:, :, None],
            sub_rep.shape)
                     for degree in range(1, self.max_degree_type + 1)]
        for level in range(self.n_levels):
            sub_rep = self.message_and_update(
                sub_rep, adj, deg_conds, counts, level)
            out.append(sub_rep)
        return out, counts

    def message_and_update(self, cur, adj, deg_conds, counts, level):

        s0, s1, s2 = cur.shape
        tmp = self.edge_layer(F.reshape(cur, (s0 * s1, s2)))

        m = F.reshape(tmp, (s0, s1, s2, MAX_EDGE_TYPE))
        m = F.transpose(m, (0, 3, 1, 2))
        m = F.reshape(m, (s0 * MAX_EDGE_TYPE, s1, s2))

        adj = F.reshape(adj, (s0 * MAX_EDGE_TYPE, s1, s1))

        m = F.batch_matmul(adj, m)
        m = F.reshape(m, (s0, MAX_EDGE_TYPE, s1, s2))
        m = F.sum(m, axis=1)
        m = m + cur
        s0, s1, s2 = m.shape
        zero_array = np.zeros(m.shape, dtype=np.float32)
        ms = [F.reshape(F.where(cond, m, zero_array), (s0 * s1, s2))
              for cond in deg_conds]
        out_x = 0
        for hidden_weight, m in zip(self.H[level], ms):
            out_x = out_x + hidden_weight(m)
        out_x = F.sigmoid(out_x)
        for s_index in range(s0):
            _from = counts[s_index] + s_index * s1
            _to = (s_index + 1) * s1
            out_x.data[_from:_to:, :] = 0.0

        out_x = F.reshape(out_x, (s0, s1, s2))
        return out_x

    def __call__(self, ids, adj, atom_array):
        ids = np.array(ids, dtype=np.int32)
        counts = []
        for list_atom in atom_array:
            list_atom = np.array(list_atom)
            count = np.count_nonzero(list_atom)
            counts.append(count)

        mol_rep = self.mol_embed_layer(ids)
        sub_rep = self.atom_embed_layer(atom_array)

        degree_mat = F.sum(adj, axis=1)
        degree_mat = F.sum(degree_mat, axis=1)
        deg_conds = [np.broadcast_to(
            ((degree_mat - degree).data == 0)[:, :, None],
            sub_rep.shape)
                     for degree in range(1, self.max_degree_type + 1)]

        self.loss = 0
        self.pos = 0
        self.neg = 0
        for level in range(self.n_levels):
            sub_rep = self.message_and_update(
                sub_rep, adj, deg_conds, counts, level)
            neg_rep = self.sampler(sub_rep, self.neg_size, counts)
            loss = self.loss_func(mol_rep, sub_rep, neg_rep, counts, level)

            self.loss += loss
        return self.loss

    def loss_func(self, mol_rep, pos, neg, counts, level):
        s0, s1, s2 = pos.shape
        assert s0 == mol_rep.shape[0]
        assert s2 == mol_rep.shape[1]

        # molecules
        mol_rep = F.tile(mol_rep, (1, s1))
        mol_rep = F.reshape(mol_rep, (s0 * s1, s2))

        # poss part
        pos = F.reshape(pos, (s0 * s1, s2))
        gate = F.exp((self.gate_weight(pos)))
        norm = gate
        norm = F.reshape(norm, (s0, s1))
        norm = F.sum(norm, axis=1)
        norm_rep = F.reshape(F.tile(norm, (1, s1)), (s0 * s1, 1))
        gate = gate / norm_rep

        pos_loss = F.sum(mol_rep * pos, axis=1)
        pos_loss = F.reshape(pos_loss, (s0 * s1, 1))

        mol_rep_re = F.tile(mol_rep, (1, self.neg_size))
        mol_rep_re = F.reshape(mol_rep_re, (s0 * s1 * self.neg_size, s2))
        neg = F.reshape(neg, (s0 * s1 * self.neg_size, s2))
        neg_loss = F.sum(-mol_rep_re * neg, axis=1)
        neg_loss = F.reshape(neg_loss, (s0 * s1, self.neg_size))

        pos_t = [1 for _ in range(s0 * s1)]
        for s_index in range(s0):
            _from = counts[s_index] + s_index * s1
            _to = (s_index + 1) * s1
            pos_t[_from:_to] = [-1 for aa in pos_t[_from:_to]]
        pos_t = chainer.Variable(np.array(pos_t, dtype=np.int32))
        pos_t = F.reshape(pos_t, (s0 * s1, 1))
        pos = F.sigmoid_cross_entropy(pos_loss, pos_t, reduce="no")
        pos = F.sum(pos)

        neg = F.sigmoid_cross_entropy(
            neg_loss, F.tile(pos_t, (1, self.neg_size)), reduce="no")
        neg = F.sum(neg)
        self.pos += pos
        self.neg += neg
        return pos + neg / self.neg_size

    def check_present(self, sub_list, sub):
        for _sub in sub_list:
            if np.sum(sub - _sub) == 0:
                return True
        return False

    def sampler(self, sub_rep, negSampSize, counts):
        s0, s1, s2 = sub_rep.shape
        sub_rep = F.reshape(sub_rep, (s0 * s1, s2))

        sub_rep_arr = sub_rep.data
        negatives = []

        for sam_index in range(s0):
            for atom_index in range(counts[sam_index]):
                n_neg_sample = 0
                neg_samps = None

                while n_neg_sample < negSampSize:
                    rand_sam = np.random.random_integers(0, s0 - 1)
                    rand_atom = np.random.random_integers(
                        0, counts[rand_sam] - 1)
                    row_index = s1 * rand_sam + rand_atom
                    neg_samp = sub_rep_arr[row_index, :]

                    _from = sam_index * s1
                    _to = sam_index * s1 + counts[sam_index]
                    if self.check_present(sub_rep_arr[_from:_to, :], neg_samp):
                        continue
                    if n_neg_sample == 0:
                        neg_samps = neg_samp
                    else:
                        neg_samps = np.concatenate((neg_samps, neg_samp))
                    n_neg_sample += 1
                negatives.append(neg_samps)
            for atom_index in range(counts[sam_index], s1):
                neg_samples = np.zeros(s2 * negSampSize, dtype=np.float32)
                negatives.append(neg_samples)
        return chainer.Variable(np.asarray(negatives))
