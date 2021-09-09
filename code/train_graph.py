from __future__ import print_function
import datetime
import argparse
import torch
import os
import sys
import math
import random
import pickle
import numpy as np
import scipy.sparse as sp

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from run import sparse_mx_to_torch_sparse_tensor

from model_graph import OurModel_share, GCN, SGC, APPNP_Net, GrapgSage, GAT, DiGCN
from datasets import load_ENAS_data, load_BN_data
from utiles import load_module_state

m='OurModel_share' # choice = ['OurModel_share', 'GCN', 'SGC', 'APPNP_Net', 'GrapgSage', 'GAT', 'DiGCN']
d ='final_structures6' # choice = ['final_structures6', 'asia_200k']


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='BN' if d == 'asia_200k' else "ENAS",
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default=d, help='graph dataset name')  # default='final_structures6',
parser.add_argument('--nvt', type=int, default=8 if d == 'asia_200k' else 6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='_'+m,
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=50, metavar='N',  # 100
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=True,
                    help='if True, do not remove any old data in the result folder')
# model settings
parser.add_argument('--model', default='OurModel_share', help='model to use: OurModel_share, GCN, SGC, APPNP_Net, GrapgSage, GAT, DiGCN')
parser.add_argument('--adj-type', default='ib' if m == 'DiGCN' else 'di', help='ib is just used to the baseline model DiGCN')
parser.add_argument('--load-latest-model', action='store_true', default=True,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--hidden-size', type=int, default=64, metavar='N',
                    help='hidden size')
parser.add_argument('--output-size', type=int, default=1, metavar='N',
                    help='output size based on datasets')
parser.add_argument('--lam', type=float, help="regularization coefficient", default=0)
parser.add_argument('--fusion-function', type=str, default='mean', choices=['sum','mean','max','cat'],
                    help='fusion function of two embeddings')
parser.add_argument('--readout-function', type=str, default='mean', choices=['sum','mean','max'],
                    help='readout function of all node within a graph')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--dropout-rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=2021, metavar='S',
                    help='random seed (default: 2021)')

parser.add_argument('--clip', default=0, type=float,
                    help='clip grad')
parser.add_argument('--device', type=int, default=1,
                    help='cuda id')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--res-dir', type=str, default="./code/result/",
                    help='the path for storing result')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:"+str(args.device))
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.data_type == 'ENAS':
    args.res_dir = os.path.join(args.res_dir, 'na/') if args.res_dir else os.path.join(args.file_dir, 'result/na/')
    pkl_name = os.path.join(args.file_dir, 'code/data/na/processed/')
elif args.data_type == 'BN':
    args.res_dir = os.path.join(args.res_dir, 'bn/') if args.res_dir else os.path.join(args.file_dir, 'result/bn/')
    pkl_name = os.path.join(args.file_dir, 'code/data/bn/processed/')

if args.model == 'DiGCN':
    pkl_name = os.path.join(pkl_name, args.data_name + '+digcn' + '.pkl')
else:
    pkl_name = os.path.join(pkl_name, args.data_name + '.pkl')

if args.model == 'OurModel_share' or args.model == 'OurModel_share':
    args.res_dir = os.path.join(args.res_dir, '{}{}'.format(args.data_name, args.save_appendix+'+'+args.fusion_function+'+'+args.readout_function))
else:
    args.res_dir = os.path.join(args.res_dir, '{}{}'.format(args.data_name, args.save_appendix+'+'+args.readout_function))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 


# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data = pickle.load(f)
# otherwise process the raw data and save to .pkl
else:
    if args.data_type == 'ENAS':
        train_data, test_data = load_ENAS_data(os.path.join(args.file_dir, 'code/data/na/raw/' + args.data_name + '.txt'), n_types=args.nvt, batch_size=args.batch_size, adj_type=args.adj_type)

    elif args.data_type == 'BN':
        train_data, test_data = load_BN_data(os.path.join(args.file_dir, 'code/data/bn/raw/' + args.data_name + '.txt'), n_types=args.nvt, batch_size=args.batch_size, adj_type=args.adj_type)
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data), f)


# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


'''Prepare the model'''
# model
num_nodes = args.nvt + 2
model = eval(args.model)(
        in_dim=num_nodes,
        num_layer=args.num_layers,
        hidden=args.hidden_size,
        out_dim=args.output_size, 
        dropout_rate=args.dropout_rate,
        fusion=args.fusion_function,
        readout=args.readout_function
        )

model.mseloss = nn.MSELoss()
model.l1loss = nn.L1Loss()

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

if args.load_latest_model:
    ts = []
    for fn in os.listdir(args.res_dir):
        if "model_checkpoint" in fn:
            ts += [int(fn[16:fn.rindex(".")])]
    if ts:
        args.continue_from = max(ts)

if args.continue_from is not None:
    epoch = args.continue_from
    load_module_state(model, os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch)))
    load_module_state(optimizer, os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)))
    load_module_state(scheduler, os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch)))
    print("Loaded module_state epoch", epoch)


'''Define some train/test functions'''
def train(epoch, args):
    model.train()
    train_loss = 0
    train_rmse = 0
    train_l1 = 0
    train_mape = 0
    num_batch = 0
    pbar = tqdm(train_data)
    for i, g in enumerate(pbar):
        y = torch.FloatTensor(g.y).unsqueeze(1).to(device)
        g = g.to(device)
        optimizer.zero_grad()
        y_pred, x_1, x_2 = model(g, num_nodes)
        loss_1 = model.mseloss(y_pred, y)
        loss_l1 = model.l1loss(y_pred, y)
        loss_mape = torch.mean(torch.abs((y_pred-y)/y))
########################################################
        edge_index = g.edge_index
        edge_index = edge_index.cpu()
        # print("edge_index: \n", edge_index.shape)
        # print("edge_index2:\n", edge_index2)
        adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0,:], edge_index[1,:])),
                            shape=(g.num_nodes, g.num_nodes),
                            dtype=np.float32)

        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

        norm_1 = torch.norm(x_1, p=2, dim=1).reshape(-1, 1)
        norm_2 = torch.norm(x_2, p=2, dim=1).reshape(-1, 1)

        mm = torch.mm(x_1/norm_1, (x_2/norm_2).t()).to(device)
        likelihood = - torch.mul(mm, adj.to_dense()) + torch.log(1 + torch.exp(mm))
        loss_2 = torch.mean(likelihood)
        if args.lam > 0:
            loss = loss_1 + args.lam * loss_2
        else:
            loss = loss_1
###############################################################
        batch = y_pred.size(0) / num_nodes
        rmse_batch = torch.sqrt(loss_1)
        pbar.set_description('Epoch: %d, loss: %0.4f, RMSE: %0.4f'\
                % (epoch, loss.item()/batch, rmse_batch))


        loss.backward()
        optimizer.step()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        
        train_loss += loss.data
        train_rmse += rmse_batch.data
        train_l1 += loss_l1.data
        train_mape += loss_mape.data
    num_batch = len(train_data)
    avg_loss = train_loss/num_batch
    avg_rmse = train_rmse/num_batch
    avg_mae = train_l1/num_batch
    avg_mape = train_mape/num_batch
    print('====> Epoch: {} Average loss: {:.4f} Average RMSE: {:.4f} Average MAE: {:.4f} Average MAPE: {:.4f}'.format(
          epoch, avg_loss, avg_rmse, avg_mae, avg_mape))

    return train_loss, avg_loss, avg_rmse, avg_mae, avg_mape


def test():
    # test prediction accuracy
    model.eval()
    pred_loss = 0
    pred_rmse = 0
    pred_mae = 0
    pred_mape = 0
    num_batch = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    for i, g in enumerate(pbar):
        y = torch.FloatTensor(g.y).unsqueeze(1).to(device)
        g = g.to(device)
        y_pred, _, _ = model(g, num_nodes)
        loss = model.mseloss(y_pred, y)
        pred_loss += loss.data
        pred_mae += model.l1loss(y_pred, y).data
        pred_rmse += torch.sqrt(loss).data
        pred_mape += torch.mean(torch.abs((y_pred-y)/y)).data

        pbar.set_description('Test Loss: {:.4f}'.format(pred_loss))

    num_batch = len(test_data)
    pred_rmse = pred_rmse/num_batch
    pred_mae = pred_mae/num_batch
    pred_mape /= num_batch

    print('Test Average Pred RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(pred_rmse, pred_mae, pred_mape))

    return pred_rmse, pred_mae, pred_mape


time = datetime.datetime.now()

'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    train_loss, average_train_loss, average_train_rmse, average_train_mae, average_train_mape = train(epoch, args)
    with open(loss_name, 'a') as loss_file:
        loss_file.write("Epoch:{}, Average Train Loss {:.4f}, Average Train RMSE {:.4f}, Average Train MAE {:.4f}, Average Train MAPE {:.4f}\n".format(
            epoch,
            average_train_loss,
            average_train_rmse,
            average_train_mae,
            average_train_mape
            ))

    scheduler.step(train_loss)
    if epoch > 5 and ((epoch%args.save_interval) == 0) or epoch ==1:
        print("save current model...", epoch, args.save_interval, epoch%args.save_interval)
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)

print("TRAIN TIME", datetime.datetime.now()-time)
'''Testing begins here'''
pred_rmse, pred_mae, pred_mape = test()
with open(test_results_name, 'a') as result_file:
    result_file.write("Epoch {} pred_rmse: {:.4f}, pred_mae: {:.4f}, pred_mape: {:.4f} \n".format(epoch, pred_rmse, pred_mae, pred_mape))
