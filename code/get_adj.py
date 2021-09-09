import torch
import scipy
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add


############################# AGNN
def get_directed_adj(edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    else:
        edge_weight = torch.FloatTensor(edge_weight).to(edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    # out degree
    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # in degree
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    # avg_out_deg = sum(out_deg) / num_nodes
    # avg_in_deg = sum(in_deg) / num_nodes
    # out_deg = out_deg + avg_out_deg
    # in_deg = in_deg + avg_in_deg

    out_deg_inv_sqrt = out_deg.pow(-0.5)
    in_deg_inv_sqrt = in_deg.pow(-0.5)
    # deg_inv = out_deg.pow(-1)
    # deg_inv[deg_inv == float('inf')] = 0
    out_deg_inv_sqrt[out_deg_inv_sqrt == float('inf')] = 0
    in_deg_inv_sqrt[in_deg_inv_sqrt == float('inf')] = 0

    # print("deg :", deg.shape)
    # print('edge_weight:\n', edge_weight.shape)
    # print('deg[row]: \n', deg[row])
    # print('deg[col]: \n', deg[col])
    # zz

    return edge_index, out_deg_inv_sqrt[row] * edge_weight * in_deg_inv_sqrt[col]


'''
The following code is followed by DiGCN.
'''

def get_undirected_adj(edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_pr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight = None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    else:
        edge_weight = torch.FloatTensor(edge_weight).to(edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    # pagerank p
    p_pr = (1.0-alpha) * p_dense + alpha / num_nodes * torch.ones((num_nodes,num_nodes), dtype=dtype, device=p.device)


    eig_value, left_vector = scipy.linalg.eig(p_pr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)
    
    # assert val[0] == 1.0

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_pr
    L = (torch.mm(torch.mm(pi_sqrt, p_pr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_pr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # # let little possbility connection to 0, make L sparse
    # L[ L < (1/num_nodes)] = 0
    # L[ L < 5e-4] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
###########
    L_indices_2 = torch.cat([L_indices[1], L_indices[0]], dim=0).reshape(2, -1)
    L_values_2 = L[L_indices[1], L_indices[0]]
    edge_index_2 = L_indices_2
###########

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, edge_index_2, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[row]

def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight ==None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index, num_nodes, dtype):

    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
    
    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]