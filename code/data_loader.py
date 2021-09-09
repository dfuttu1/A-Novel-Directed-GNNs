import os.path as osp
import os
import shutil
import torch_geometric.transforms as T
from datasets import Datasets

# load citation and Amazon co-porchase datasets
def get_dataset(name, recache=False, normalize_features=False, adj_type=None, transform=None):

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    file_path = osp.join(path, name, 'processed')
    if recache == True:
        print("Delete old processed data cache...")
        if osp.exists(file_path):
            shutil.rmtree(file_path)
        os.mkdir(file_path)
        print('Finish cleaning.')
    
    dataset = Datasets(path, name, adj_type=adj_type)
    
    print('Finish dataset preprocessing.')
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

if __name__ == "__main__":
    dataset = 'cora'
    alpha = 0.1
    recache = True
    normalize_features = True
    adj_type = 'di'
    dataset = get_dataset(dataset, alpha, recache, normalize_features, adj_type)
    data = dataset[0]
    print("Num of nodes ", data.num_nodes)
    print("Num of edges ", data.num_edges)
