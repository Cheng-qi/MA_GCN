import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid

def randSplit(y, train_pro, val_pro, random_seed = 0):
    nodes_nums = y.shape[0]
    train_val_nodes_nums = y.shape[0]-1000
    val_train_labels = y[:train_val_nodes_nums]

    train_num = int(nodes_nums * train_pro)
    val_num = int(nodes_nums * val_pro)
    # test_num = nodes_nums - train_num - val_num

    val_train_nodes = np.array(list(range(train_val_nodes_nums)))
    # val_train_nodes, test_nodes, val_train_labels, test_labels = \
    #     train_test_split(nodes_arr, data.y.numpy(), train_size=train_num + val_num, test_size=test_num,
    #                      stratify=data.y.numpy(), random_state=random_seed)
    train_nodes, val_nodes, train_labels, val_labels = \
        train_test_split(val_train_nodes, val_train_labels, train_size=train_num, test_size=val_num,
                         stratify=val_train_labels, random_state=random_seed)

    # train_nodes, val_nodes, test_nodes = randSplit(data, 0.1, 0.1)
    # data.train_mask[:] = False
    # data.train_mask[train_nodes] = True
    # data.val_mask[:] = False
    # data.val_mask[val_nodes] = True
    # data.test_mask[:] = False
    # data.test_mask[test_nodes] = True

    return train_nodes, val_nodes

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data_new(dataset_str, is_rand_split=False, train_set_pro=0):
    '''
    Rewrite loading data to make it can be used rand split.

    dataset_str: str, name of dataset
    is_rand_split: bool, if using rang split, default Fasle(standard split)
    train_set_pro: float, proportion of train set when using rand split(only usable when is_rand_split=True).

    '''

    dataset = Planetoid(root='../../data/' + dataset_str, name=dataset_str)
    data = dataset[0]
    adj = nx.adjacency_matrix(nx.from_edgelist(data.edge_index.numpy().T), nodelist=list(range(data.num_nodes)))
    features = sp.lil_matrix(dataset.data.x)
    labels = np.zeros((data.num_nodes, dataset.num_classes))
    labels[list(range(data.num_nodes)), data.y] = 1


    train_mask, val_mask, test_mask = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[data.train_mask, :] = labels[data.train_mask, :]
    y_val[data.val_mask, :] = labels[data.val_mask, :]
    y_test[data.test_mask, :] = labels[data.test_mask, :]

    if(is_rand_split):
        idx_train, idx_val = randSplit(labels,train_set_pro, 0.2)
        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  #每一行求和，得到列向量
    r_inv = np.power(rowsum, -1).flatten() #对rowsum的每一个元素求倒数，然后拉平为一维数组（flatten()）
    r_inv[np.isinf(r_inv)] = 0. #将无效值变为0
    r_mat_inv = sp.diags(r_inv) #转为对角矩阵
    features = r_mat_inv.dot(features) #点乘
    return sparse_to_tuple(features) #表示为元组


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)#将稀疏表示组合为矩阵表示A^~
    rowsum = np.array(adj.sum(1))#度矩阵D^~
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()#(A^~*D^~^-0.5)^T*(D^~^-0.5)=D^~^-0.5*A^~*D^~^-0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)#D^-0.5*A*D^-0.5
    laplacian = sp.eye(adj.shape[0]) - adj_normalized#L
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])#L^~

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))#T_0
    t_k.append(scaled_laplacian)#T_1

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
