import torch
import torch.nn as nn
import torch.nn.functional as F
from utiles import DIGCNConv
from torch_geometric.nn import GCNConv, SGConv, APPNP, SAGEConv, GATConv

class OurModel(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum',  readout='sum', cached=True):
        super(OurModel, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.fusion = fusion
        self.out_dim = out_dim
        self.readout = readout

        self.conv1_1 = GCNConv(in_dim, hidden)
        self.conv1_2 = GCNConv(in_dim, hidden)

        self.convs_1 = torch.nn.ModuleList()
        self.convs_2 = torch.nn.ModuleList()
        for layer in range(num_layer - 2):
            self.convs_1.append(GCNConv(hidden, hidden))
            self.convs_2.append(GCNConv(hidden, hidden))
        self.act = nn.ReLU()

        self.conv2_1 = GCNConv(hidden, out_dim)
        self.conv2_2 = GCNConv(hidden, out_dim)
        self.fc_1 = nn.Linear(out_dim * 2, out_dim)
        self.fc_2 = nn.Linear(out_dim * 2, out_dim)
        self.act_1 = nn.Sigmoid()
        self.act_2 = nn.Tanh()

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        for i in range(len(self.convs_1)):
            self.convs_1[i].reset_parameters()
            self.convs_2[i].reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index1, edge_index2, edge_weight1, edge_weight2  = data.x, data.edge_index, data.edge_index2, data.edge_weight, data.edge_weight2

        x_1 = self.conv1_1(x, edge_index1, edge_weight1)
        x_2 = self.conv1_2(x, edge_index2, edge_weight2)
        x_1 = F.dropout(self.act(x_1), p=self.dropout, training=self.training)
        x_2 = F.dropout(self.act(x_2), p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x_1 = self.convs_1[layer](x_1, edge_index1, edge_weight1)
                x_2 = self.convs_2[layer](x_2, edge_index2, edge_weight2)
                # h = self.batch_norms[layer](h)
                # h_2 = self.batch_norms_2[layer](h_2)

                x_1 = F.dropout(self.act(x_1), self.dropout, training=self.training)
                x_2 = F.dropout(self.act(x_2), self.dropout, training=self.training)

        x_11 = self.conv2_1(x_1, edge_index1, edge_weight1).reshape(-1, num_nodes, self.out_dim)
        x_12 = self.conv2_2(x_2, edge_index2, edge_weight2).reshape(-1, num_nodes, self.out_dim)


        x_cat_ = torch.cat([x_11, x_12], dim=2)
        x_cat = self.fc_1(x_cat_)
        x_max = torch.max(x_11, x_12)
        x_sum = x_11 + x_12
        x_mean = (x_11 + x_12) / 2

        # x_1_max = torch.max(x_1, dim=1).values
        # print('x_1_max: ', x_1_max.shape)

        if self.fusion == 'sum':
            out_fusion = x_sum
        elif self.fusion == 'max':
            out_fusion = x_max
        elif self.fusion == 'mean':
            out_fusion = x_mean
        elif self.fusion == 'cat':
            out_fusion = x_cat
        else:
            print("Please type reasonable fusion function: 'sum', 'max', 'mean', or 'cat'.")

        if self.readout == 'sum':
            out = torch.sum(out_fusion, dim=1)
        elif self.readout == 'max':
            out = torch.max(out_fusion, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(out_fusion, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x_1, x_2

class OurModel_share(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum',  readout='sum', cached=True):
        super(OurModel_share, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.fusion = fusion
        self.out_dim = out_dim
        self.readout = readout

        self.conv1_1 = GCNConv(in_dim, hidden)
        self.conv1_2 = GCNConv(in_dim, hidden)

        self.convs_1 = torch.nn.ModuleList()
        self.convs_2 = torch.nn.ModuleList()
        for layer in range(num_layer - 2):
            self.convs_1.append(GCNConv(hidden, hidden))
            self.convs_2.append(GCNConv(hidden, hidden))
        self.act = nn.ReLU()

        self.conv2_1 = GCNConv(hidden, out_dim)
        self.conv2_2 = GCNConv(hidden, out_dim)
        self.fc_1 = nn.Linear(out_dim * 2, out_dim)
        self.fc_2 = nn.Linear(out_dim * 2, out_dim)
        self.act_1 = nn.Sigmoid()
        self.act_2 = nn.Tanh()

    def reset_parameters(self):
        self.conv1_1.reset_parameters()
        self.conv1_2.reset_parameters()
        self.conv2_1.reset_parameters()
        self.conv2_2.reset_parameters()
        for i in range(len(self.convs_1)):
            self.convs_1[i].reset_parameters()
            self.convs_2[i].reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index1, edge_index2, edge_weight1, edge_weight2  = data.x, data.edge_index, data.edge_index2, data.edge_weight, data.edge_weight2

        x_1 = self.conv1_1(x, edge_index1, edge_weight1)
        x_2 = self.conv1_1(x, edge_index2, edge_weight2)
        x_1 = F.dropout(self.act(x_1), p=self.dropout, training=self.training)
        x_2 = F.dropout(self.act(x_2), p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x_1 = self.convs_1[layer](x_1, edge_index1, edge_weight1)
                x_2 = self.convs_1[layer](x_2, edge_index2, edge_weight2)
                # h = self.batch_norms[layer](h)
                # h_2 = self.batch_norms_2[layer](h_2)

                x_1 = F.dropout(self.act(x_1), self.dropout, training=self.training)
                x_2 = F.dropout(self.act(x_2), self.dropout, training=self.training)

        x_11 = self.conv2_1(x_1, edge_index1, edge_weight1).reshape(-1, num_nodes, self.out_dim)
        x_12 = self.conv2_1(x_2, edge_index2, edge_weight2).reshape(-1, num_nodes, self.out_dim)


        x_cat_ = torch.cat([x_11, x_12], dim=2)
        x_cat = self.fc_1(x_cat_)
        x_max = torch.max(x_11, x_12)
        x_sum = x_11 + x_12
        x_mean = (x_11 + x_12) / 2

        # x_1_max = torch.max(x_1, dim=1).values
        # print('x_1_max: ', x_1_max.shape)

        if self.fusion == 'sum':
            out_fusion = x_sum
        elif self.fusion == 'max':
            out_fusion = x_max
        elif self.fusion == 'mean':
            out_fusion = x_mean
        elif self.fusion == 'cat':
            out_fusion = x_cat
        else:
            print("Please type reasonable fusion function: 'sum', 'max', 'mean', or 'cat'.")

        if self.readout == 'sum':
            out = torch.sum(out_fusion, dim=1)
        elif self.readout == 'max':
            out = torch.max(out_fusion, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(out_fusion, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x_1, x_2

########################### GCN
class GCN(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum', readout='max', cached=True):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.out_dim = out_dim
        self.readout = readout

        self.conv1 = GCNConv(in_dim, hidden)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layer - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.act = nn.ReLU()

        self.conv2 = GCNConv(hidden, out_dim)
        self.act_1 = nn.Sigmoid()
        self.act_2 = nn.Tanh()

        # self.readout = torch.sum

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(self.act(x), p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x = self.convs[layer](x, edge_index, edge_weight)
                x = F.dropout(self.act(x), self.dropout, training=self.training)

        x_ = self.conv2(x, edge_index, edge_weight).reshape(-1, num_nodes, self.out_dim)

        if self.readout == 'sum':
            out = torch.sum(x_, dim=1)
        elif self.readout == 'max':
            out = torch.max(x_, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(x_, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x, x

########################### SGC
class SGC(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum', readout='max', cached=True):
        super(SGC, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.out_dim = out_dim
        self.readout = readout

        self.conv1 = SGConv(in_dim, hidden)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layer - 2):
            self.convs.append(SGConv(hidden, hidden))

        self.conv2 = SGConv(hidden, out_dim)

        # self.readout = torch.sum

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x = self.convs[layer](x, edge_index, edge_weight)
                x = F.dropout(x, self.dropout, training=self.training)

        x_ = self.conv2(x, edge_index, edge_weight).reshape(-1, num_nodes, self.out_dim)

        if self.readout == 'sum':
            out = torch.sum(x_, dim=1)
        elif self.readout == 'max':
            out = torch.max(x_, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(x_, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x, x

########################### APPNP
class APPNP_Net(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum', readout='max', cached=True):
        super(APPNP_Net, self).__init__()
        self.dropout = dropout_rate
        self.out_dim = out_dim
        self.readout = readout

        self.fc_1 = nn.Linear(in_dim, hidden)
        self.fc_2 = nn.Linear(hidden, hidden)
        self.fc_3 = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU()
        self.conv = APPNP(num_layer, 0.1)


    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.edge_weight
        x = self.fc_1(x)
        x = F.dropout(self.act(x), p=self.dropout, training=self.training)
        x = self.fc_2(x)
        x = F.dropout(self.act(x), p=self.dropout, training=self.training)

        x = self.conv(x, edge_index, edge_weight)
        x = self.fc_3(x)
        x_ = x.reshape(-1, num_nodes, self.out_dim)

        if self.readout == 'sum':
            out = torch.sum(x_, dim=1)
        elif self.readout == 'max':
            out = torch.max(x_, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(x_, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x, x

########################### GraphSage
class GrapgSage(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum', readout='max', cached=True):
        super(GrapgSage, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.out_dim = out_dim
        self.readout = readout

        self.conv1 = SAGEConv(in_dim, hidden)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layer - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.act = nn.ReLU()

        self.conv2 = SAGEConv(hidden, out_dim)
        self.act_1 = nn.Sigmoid()
        self.act_2 = nn.Tanh()

        # self.readout = torch.sum

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index  = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(self.act(x), p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x = self.convs[layer](x, edge_index)
                x = F.dropout(self.act(x), self.dropout, training=self.training)

        x_ = self.conv2(x, edge_index).reshape(-1, num_nodes, self.out_dim)

        if self.readout == 'sum':
            out = torch.sum(x_, dim=1)
        elif self.readout == 'max':
            out = torch.max(x_, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(x_, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x, x

########################### GAT
class GAT(torch.nn.Module):
    def __init__(self, in_dim=6, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum', readout='max', cached=True):
        super(GAT, self).__init__()
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.out_dim = out_dim
        self.readout = readout

        self.conv1 = GATConv(in_dim, hidden)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layer - 2):
            self.convs.append(GATConv(hidden, hidden))
        self.act = nn.ReLU()

        self.conv2 = GATConv(hidden, out_dim)
        self.act_1 = nn.Sigmoid()
        self.act_2 = nn.Tanh()

        # self.readout = torch.sum

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index  = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(self.act(x), p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x = self.convs[layer](x, edge_index)
                x = F.dropout(self.act(x), self.dropout, training=self.training)

        x_ = self.conv2(x, edge_index).reshape(-1, num_nodes, self.out_dim)

        if self.readout == 'sum':
            out = torch.sum(x_, dim=1)
        elif self.readout == 'max':
            out = torch.max(x_, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(x_, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x, x

########################### DiGCN
class InceptionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = nn.Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index, edge_weight, edge_index2, edge_weight2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2

class DiGCN(torch.nn.Module):
    def __init__(self, in_dim, num_layer=2, hidden=64, out_dim=1, dropout_rate=0.5, fusion='sum', readout='max', cached=True):
        super(DiGCN, self).__init__()
        self.dropout = dropout_rate
        self.out_dim = out_dim
        self.readout = readout

        self.ib1 = InceptionBlock(in_dim, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self.ib3 = InceptionBlock(hidden, out_dim)

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, data, num_nodes):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        edge_index2, edge_weight2 = data.edge_index2, data.edge_weight2
        
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self.dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self.dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x0+x1+x2

        x_ = x.reshape(-1, num_nodes, self.out_dim)

        if self.readout == 'sum':
            out = torch.sum(x_, dim=1)
        elif self.readout == 'max':
            out = torch.max(x_, dim=1).values
        elif self.readout == 'mean':
            out = torch.mean(x_, dim=1)
        else:
            print("Please type reasonable readout function: 'sum', 'max' or 'mean'.")

        return out, x, x
