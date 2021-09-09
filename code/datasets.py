import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import os
import torch
import sys
import argparse
import numpy as np
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from get_adj import get_undirected_adj, get_directed_adj, get_pr_directed_adj, get_appr_directed_adj, get_second_directed_adj

# citation and Amazon co-porchase datasets
class Datasets(InMemoryDataset):
    r"""
    For citation datasets, nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    For Amazon co-purchase, nodes represent goods, edges indicate that two goods are 
    frequently bought together, node features are bag-of-words encoded product reviews, 
    and class labels are given by the product category.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"cora_ml"`,
            :obj:`"citeseer"`, :obj:`"amazon_computer", :obj:`"amazon_photo").
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, adj_type=None, transform=None, pre_transform=None):
        self.name = name
        self.adj_type = adj_type
        super(Datasets, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return 'data.pt'

    # def download(self):
    #     return

    def process(self):
        data = process_datasets(self.raw_dir, self.name, self.adj_type)
        # data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def process_datasets(path="./data/citeseer/raw/", dataset='citeseer', adj_type='di'):
    se = 1020
    if dataset == 'cora_ml' or 'citeseer':
        se = 177
    os.makedirs(path, exist_ok=True)
    dataset_path = os.path.join(path, '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    adj, features, labels = g['A'], g['X'], g['z']
    
    # Set new random splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * the rest for testing

    mask = train_test_split(labels, seed=se, train_examples_per_class=20, val_size=500, test_size=None)

    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    coo = adj.tocoo()
    # incoming edges and outgoing edge
    indices = np.vstack((coo.row, coo.col))
    indices2 = np.vstack((coo.col, coo.row))

    indices = torch.from_numpy(indices).long()
    indices2 = torch.from_numpy(indices2).long()
    features = torch.from_numpy(features.todense()).float()
    labels = torch.from_numpy(labels).long()

    if adj_type == 'un':
        print("Processing to undirected adj matrix")
        indices = to_undirected(indices)
        # normlize the symmetric adjacency matrix
        edge_index, edge_weight = get_undirected_adj(indices, features.shape[0], features.dtype)
        data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    elif adj_type == 'di':
        print("Processing to directed adj matrix")
        # normlize the asymmetric adjacency matrix
        edge_index1, edge_weight1 = get_directed_adj(indices, features.shape[0], features.dtype)
        data = Data(x=features, edge_index=edge_index1, edge_weight=edge_weight1, y=labels)
        edge_index2, edge_weight2 = get_directed_adj(indices2, features.shape[0], features.dtype)
        data.edge_index2 = edge_index2
        data.edge_weight2 = edge_weight2
    elif adj_type == 'or':
        print("Processing to original directed adj")
        data = Data(x=features, edge_index=indices, edge_weight=None, y=labels)
    else:
        print("Unsupported adj type.")
        sys.exit()
    
    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']

    return data

def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        # edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask

####################################### NA
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

def load_ENAS_data(dataset_path, n_types=6, batch_size=64, adj_type='di', with_y=True, burn_in=1000):
    # load ENAS format NNs to pyg_graphs
    g_list = []
    max_n = 0  # maximum number of nodes
    with open(dataset_path, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            # decode data to pyggraph
            g = decode_ENAS_to_pygraph(row, y, adj_type)
            max_n = max(max_n, g.num_nodes)
            g_list.append(g)
    graph_args.num_vertex_type = n_types + 2
    graph_args.max_n = max_n  # maximum number of nodes
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    # split train/test data
    train_data = g_list[:int(ng*0.9)]
    test_data = g_list[int(ng*0.9):]
    # construct batch
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader

def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x

def decode_ENAS_to_pygraph(row, y, adj_type, n_types=6):
    n_types += 2  # add start_type 0, end_type 1

    adj = np.zeros((n_types, n_types))
    x = []
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    x += [one_hot(0, n_types)]
    # ignore start vertex
    node_types2 = []
    for i, node in enumerate(row):
        node_type = node[0] + 2  # convert 0, 1, 2... to 2, 3, 4...
        node_types2 += [node_type]
        x += [one_hot(node_type, n_types)]
        adj[i, i+1] = 1
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                adj[j, i + 1] = 1

    # output node
    node_type = 1
    x += [one_hot(node_type, n_types)]
    adj[n, n + 1] = 1

    nx_graph = nx.DiGraph(adj)
    x = torch.cat(x, dim=0).float()

    ro, col = torch.tensor(list(nx_graph.edges)).t().contiguous()
    
    indices = np.vstack((ro, col))
    indices2 = np.vstack((col, ro))

    indices = torch.from_numpy(indices).long()
    indices2 = torch.from_numpy(indices2).long()
    if adj_type == 'di':
        edge_index1, edge_weight1 = get_directed_adj(indices, x.shape[0], x.dtype)
        graph = Data(x=x, edge_index=edge_index1, edge_weight=edge_weight1, y=y)
        edge_index2, edge_weight2 = get_directed_adj(indices2, x.shape[0], x.dtype)
        graph.edge_index2 = edge_index2
        graph.edge_weight2 = edge_weight2
    elif adj_type == 'ib':
        edge_index1, edge_weight1 = get_appr_directed_adj(0.1, indices, x.shape[0], x.dtype) 
        graph = Data(x=x, edge_index=edge_index1, edge_weight=edge_weight1, y=y)
        edge_index2, edge_weight2 = get_second_directed_adj(indices, x.shape[0], x.dtype)
        graph.edge_index2 = edge_index2
        graph.edge_weight2 = edge_weight2

    return graph

########################################## BN
def load_BN_data(dataset_path, n_types=6, batch_size=64, adj_type='di', with_y=True):
    # load raw Bayesian network strings to pyg_graphs
    g_list = []
    max_n = 0  # maximum number of nodes
    with open(dataset_path, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            g = decode_BN_to_pygraph(row, y, adj_type)
            max_n = max(max_n, g.num_nodes)
            assert(max_n == g.num_nodes)  # all BNs should have the same node number
            g_list.append(g)
    graph_args.num_vertex_type = n_types + 2
    graph_args.max_n = max_n  # maximum number of nodes
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    # random.Random(rand_seed).shuffle(g_list)
    train_data = g_list[:int(ng*0.9)]
    test_data = g_list[int(ng*0.9):]
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader

def decode_BN_to_pygraph(row, y, adj_type, n_types=8):
    n_types += 2  # add start_type 0, end_type 1

    adj = np.zeros((n_types, n_types))
    x = []
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    end_vertices = [True] * n
    x += [one_hot(0, n_types)]
    # ignore start vertex
    node_types2 = []
    for i, node in enumerate(row):
        node_type = node[0] + 2  # convert 0, 1, 2... to 2, 3, 4...
        node_types2 += [node_type]
        x += [one_hot(node_type, n_types)]
        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            adj[0, i + 1] = 1
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    adj[j + 1, i + 1] = 1
                    end_vertices[j] = False
    # output node
    node_type = 1
    x += [one_hot(node_type, n_types)]
    for j, flag in enumerate(end_vertices):  # connect all loose-end vertices to the output node
        if flag == True:
            adj[j + 1, n + 1] = 1

    nx_graph = nx.DiGraph(adj)
    x = torch.cat(x, dim=0).float()

    ro, col = torch.tensor(list(nx_graph.edges)).t().contiguous()
    
    indices = np.vstack((ro, col))
    indices2 = np.vstack((col, ro))

    indices = torch.from_numpy(indices).long()
    indices2 = torch.from_numpy(indices2).long()
    if adj_type == 'di':
        edge_index1, edge_weight1 = get_directed_adj(indices, x.shape[0], x.dtype)
        graph = Data(x=x, edge_index=edge_index1, edge_weight=edge_weight1, y=y)
        edge_index2, edge_weight2 = get_directed_adj(indices2, x.shape[0], x.dtype)
        graph.edge_index2 = edge_index2
        graph.edge_weight2 = edge_weight2
    elif adj_type == 'ib':
        edge_index1, edge_weight1 = get_appr_directed_adj(0.1, indices, x.shape[0], x.dtype) 
        graph = Data(x=x, edge_index=edge_index1, edge_weight=edge_weight1, y=y)
        edge_index2, edge_weight2 = get_second_directed_adj(indices, x.shape[0], x.dtype)
        graph.edge_index2 = edge_index2
        graph.edge_weight2 = edge_weight2

    return graph

if __name__ == "__main__":
    ############################# node classification task: cora_ml, citeseer, am_computer, am_photo
    dataset = Datasets('./code/data/', 'citeseer', adj_type='di')


    ############################# graph regression task: NA
    train_data, test_data, graph_args = load_ENAS_data("./code/data/na/raw/final_structures6.txt", n_types=6)

    for batch in train_data:
        print(batch)
        edge_index = batch.edge_index
        adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0,:], edge_index[1,:])),
                            shape=(batch.num_nodes, batch.num_nodes),
                            dtype=np.float32)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        print("adj0: ", adj.to_dense()[0][:16])
        print("adj1: ", adj.to_dense()[1][:16])
        print("adj8: ", adj.to_dense()[8][:16])
        print("adj9: ", adj.to_dense()[9][:16])
        break