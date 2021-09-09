import argparse
from genericpath import exists
import torch
import os
import sys
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from data_loader import get_dataset
from run import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer') # cora_ml, citeseer, cora, amazon_computers, amazon_photo
parser.add_argument('--gpu-no', type=int, default=1)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--lam', type=float, help="regularization coefficient", default=0.1)
parser.add_argument('--fusion', type=str, help="fusion the incoming and sending embeddings", default='mean') # sum, cat, max
parser.add_argument('--mask', type=bool, help="just regularize the train examples", default=True)
parser.add_argument('--share_parameters', type=bool, help="the networks of two embeddings share parameters", default=True)
parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)
parser.add_argument('--normalize-features', type=bool, default=True)
parser.add_argument('--adj-type', type=str, help="'di' is the default adj type of AGNN", default='di')

args = parser.parse_args()


class AGNN_share(torch.nn.Module):
    '''
    The AGNN method we proposed, can capture the asymmetric structure of directed graph, 
    by modeling the different roles of receiving and sending information, 
    and obtaining two embeddings of one node. Note that to reduce computation, 
    we let the incoming and outgoing maps share parameters.
    '''
    def __init__(self, dataset, cached=False):
        super(AGNN_share, self).__init__()
        self.conv1_1 = GCNConv(dataset.num_features, args.hidden)
        self.conv1_2 = GCNConv(dataset.num_features, args.hidden)

        self.convs_1 = torch.nn.ModuleList()
        self.convs_2 = torch.nn.ModuleList()
        for layer in range(args.num_layer - 2):
            self.convs_1.append(GCNConv(args.hidden, args.hidden))
            self.convs_2.append(GCNConv(args.hidden, args.hidden))
        self.act = nn.ReLU()

        self.conv2_1 = GCNConv(args.hidden, dataset.num_classes)
        self.conv2_2 = GCNConv(args.hidden, dataset.num_classes)
        self.fc = nn.Linear(dataset.num_classes * 2, dataset.num_classes)
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
        self.fc.reset_parameters()

    def forward(self, data):
        x, edge_index1, edge_index2, edge_weight1, edge_weight2  = data.x, data.edge_index, data.edge_index2, data.edge_weight, data.edge_weight2
        # x_1 aggregates sending information and x_2 aggregates receiving information.
        x_1 = self.conv1_1(x, edge_index1, edge_weight1)
        x_2 = self.conv1_1(x, edge_index2, edge_weight2)
        x_1 = F.dropout(self.act(x_1), p=args.dropout, training=self.training)
        x_2 = F.dropout(self.act(x_2), p=args.dropout, training=self.training)

        # multi-layer
        if args.num_layer > 2:
            for layer in range(args.num_layer - 2):
                x_1 = self.convs_1[layer](x_1, edge_index1, edge_weight1)
                x_2 = self.convs_1[layer](x_2, edge_index2, edge_weight2)
                # h = self.batch_norms[layer](h)
                # h_2 = self.batch_norms_2[layer](h_2)

                x_1 = F.dropout(self.act(x_1), args.dropout, training=self.training)
                x_2 = F.dropout(self.act(x_2), args.dropout, training=self.training)
                
        # obtain the outgoing embedding x_1 and receiving embedding x_2
        x_1 = self.conv2_1(x_1, edge_index1, edge_weight1)
        x_2 = self.conv2_1(x_2, edge_index2, edge_weight2)

        # print("x_1:", x_1.shape)

        x_cat_ = torch.cat([x_1, x_2], dim=1)
        x_cat = self.fc(x_cat_)
        x_max = torch.max(x_1, x_2)
        x_sum = x_1 + x_2
        x_mean = (x_1 + x_2) / 2

        # fusion the two embeddings
        if args.fusion == 'sum':
            out = F.log_softmax(x_sum, dim=1)
        elif args.fusion == 'max':
            out = F.log_softmax(x_max, dim=1)
        elif args.fusion == 'mean':
            out = F.log_softmax(x_mean, dim=1)
        elif args.fusion == 'cat':
            out = F.log_softmax(x_cat, dim=1)
        else:
            print("Please type reasonable fusion function: 'sum', 'max', 'mean', or 'cat'.")

        return out, x_1, x_2

class AGNN(torch.nn.Module):
    def __init__(self, dataset, cached=False):
        super(AGNN, self).__init__()
        self.conv1_1 = GCNConv(dataset.num_features, args.hidden)
        self.conv1_2 = GCNConv(dataset.num_features, args.hidden)

        self.convs_1 = torch.nn.ModuleList()
        self.convs_2 = torch.nn.ModuleList()
        for layer in range(args.num_layer - 2):
            self.convs_1.append(GCNConv(args.hidden, args.hidden))
            self.convs_2.append(GCNConv(args.hidden, args.hidden))
        self.act = nn.ReLU()

        self.conv2_1 = GCNConv(args.hidden, dataset.num_classes)
        self.conv2_2 = GCNConv(args.hidden, dataset.num_classes)
        self.fc = nn.Linear(dataset.num_classes * 2, dataset.num_classes)
        self.fc_1 = nn.Linear(dataset.num_classes * 2, dataset.num_classes)
        self.fc_2 = nn.Linear(dataset.num_classes * 2, dataset.num_classes)
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
        self.fc.reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()


    def forward(self, data):
        x, edge_index1, edge_index2  = data.x, data.edge_index, data.edge_index2
        x_1 = self.conv1_1(x, edge_index1)
        x_2 = self.conv1_2(x, edge_index2)
        x_1 = F.dropout(self.act(x_1), p=args.dropout, training=self.training)
        x_2 = F.dropout(self.act(x_2), p=args.dropout, training=self.training)

        if args.num_layer > 2:
            for layer in range(args.num_layer - 2):
                x_1 = self.convs_1[layer](x_1, edge_index1)
                x_2 = self.convs_2[layer](x_2, edge_index2)
                # h = self.batch_norms[layer](h)
                # h_2 = self.batch_norms_2[layer](h_2)

                x_1 = F.dropout(self.act(x_1), args.dropout, training=self.training)
                x_2 = F.dropout(self.act(x_2), args.dropout, training=self.training)

        x_11 = self.conv2_1(x_1, edge_index1)
        x_12 = self.conv2_2(x_2, edge_index2)

        # print("x_1:", x_1.shape)

        x_cat = torch.cat([x_11, x_12], dim=1)
        x_cat_1 = self.act_1(self.fc_1(x_cat))
        x_cat_2 = self.act_2(self.fc_2(x_cat))
        x_cat = x_cat_1 * x_cat_2

        x_max = torch.max(x_11, x_12)

        x_sum = x_11 + x_12

        x_mean = (x_11 + x_12) / 2

        out = F.log_softmax(x_sum, dim=1)

        return out, x_11, x_12

class GCN(torch.nn.Module):
    def __init__(self, dataset, cached=False):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, args.hidden)

        self.convs = torch.nn.ModuleList()
        for layer in range(args.num_layer - 2):
            self.convs.append(GCNConv(args.hidden, args.hidden))
        self.act = nn.ReLU()

        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.act_1 = nn.Sigmoid()
        self.act_2 = nn.Tanh()

        # self.readout = torch.sum

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for i in range(len(self.convs)):
            self.convs[i].reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(self.act(x), p=self.dropout, training=self.training)

        if self.num_layer > 2:
            for layer in range(self.num_layer - 2):
                x = self.convs[layer](x, edge_index, edge_weight)
                x = F.dropout(self.act(x), self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)

        out = F.log_softmax(x, dim=1)
        return out, x, x


def run_digcn(dataset, name, save_path):
    if args.share_parameters:
        model = AGNN_share(dataset)
    else:
        model = AGNN(dataset)
    val_loss, val_closs, val_regloss, test_acc, test_std, time = run(dataset, name, args.gpu_no, model, args.runs, args.epochs, args.lam, args.mask, args.lr,
                                             args.weight_decay,
                                             args.early_stopping, save_path)
    return val_loss, val_closs, val_regloss, test_acc, test_std, time


if __name__ == '__main__':
    ############ Result Path ###########
    if args.share_parameters:
        out_path = './code/result/share_parameters/' + args.dataset + '/'  + args.adj_type + args.fusion + '/'
    else:
        out_path = './code/result/not_share_parameters/' + args.dataset + '/' + args.adj_type + args.fusion + '/'

    ############ Result Name ###########
    if args.mask:
        outfile_name = 'lam_' + str(args.lam) + '-mask' + '-lr_' + str(args.lr) + '-epochs_' + str(args.epochs) + '-early_' + str(args.early_stopping) + '-layer_' + str(args.num_layer)
    else:
        outfile_name = 'lam_' + str(args.lam) + '-lr_' + str(args.lr) + '-epochs_' + str(args.epochs) + '-early_' + str(args.early_stopping) + '-layer_' + str(args.num_layer)
    os.makedirs(out_path, exist_ok=True)
    sys.stdout = open(out_path + outfile_name + '.txt', 'wt')
    print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    ############ Model Configuration ####
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    ############# Load data #############
    dataset = get_dataset(args.dataset, args.recache, args.normalize_features, args.adj_type)
      
    print("Num of nodes ", dataset[0].num_nodes)
    print("Num of edges ", dataset[0].num_edges)

    ######## Train, Val and test ###########
    val_loss, val_closs, val_regloss, test_acc, test_std, times = run_digcn(dataset, args.dataset, out_path + outfile_name)
    print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))