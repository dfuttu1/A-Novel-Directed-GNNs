from __future__ import division

import time
import os
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch import tensor
from torch.optim import Adam

def run(dataset, name, gpu_no, model, runs, epochs, lam, loss_mask, lr, weight_decay, early_stopping, save_path):
    
    torch.cuda.set_device(gpu_no)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    valacc, val_losses, val_closses, val_reglosses, train_accs, accs, durations = [], [], [], [], [], [], []
    #  run each experiment 20 times with random weight initialization.
    for ru in range(1, runs + 1):
        data = dataset[0]
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        train_acc = 0
        val_regloss = 0
        val_closs = 0

        val_acc = 0
        test_acc = 0
        val_loss_history = []
        flag = 0
        for epoch in range(1, epochs + 1):
            flag = flag + 1
            # train model
            train(model, optimizer, lam, loss_mask, data)
            # val and test model
            eval_info = evaluate(model, lam, loss_mask, data)
            eval_info['epoch'] = epoch

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                val_regloss = eval_info['val_regloss']
                val_closs = eval_info['val_closs']
                train_acc = eval_info['train_acc']
                val_acc = eval_info['val_acc']
                test_acc = eval_info['test_acc']
            print('Run: {:d}, Epoch: {:d}, Train Acc: {:.4f}, Best Val Loss: {:.4f}, Regul Val Loss: {:.4f}, Test Acc: {:.4f}'.
                  format(ru,
                         epoch,
                         train_acc,
                         best_val_loss,
                         eval_info['val_regloss'],
                         test_acc))


            val_loss_history.append(eval_info['val_loss'])

            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # save model
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path + '/' + str(ru) + 'run.pt')

        t_end = time.perf_counter()

        valacc.append(val_acc)
        val_losses.append(best_val_loss)
        val_closses.append(val_closs)
        val_reglosses.append(val_regloss)
        train_accs.append(train_acc)
        accs.append(test_acc)
        durations.append(t_end - t_start)
    vacc, loss, closs, regloss, tracc, acc, duration = tensor(valacc), tensor(val_losses), tensor(val_closses), tensor(val_reglosses), tensor(train_accs), tensor(accs), tensor(durations)

    best_run = acc.argmax().item() + 1
    os.rename(save_path + '/' + str(best_run) + 'run.pt', save_path + '/' + str(best_run) + 'run-best.pt')


    print('Val Acc: {:.4f} ± {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.4f} ± {:.4f}, Test Accuracy: {:.4f} ± {:.4f}, Best Accuracy: {:.4f}, Best Run {:d}, Duration: {:.4f}'.
          format(vacc.mean().item(),
                 vacc.std().item(),
                 loss.mean().item(),
                 tracc.mean().item(),
                 tracc.std().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 acc.max().item(),
                 best_run,
                 duration.mean().item()))
    return loss.mean().item(), closs.mean().item(), regloss.mean().item(),  acc.mean().item(), acc.std().item(), duration.mean().item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def train(model, optimizer, lam, loss_mask, data):
    model.train()
    optimizer.zero_grad()
    out, x_1, x_2 = model(data)
#######################
    # edge_index1 and edge_index2 are outgoing and incoming edges, respectively.
    edge_index1 = data.edge_index
    edge_index2 = data.edge_index2
    device = edge_index1.device
    edge_index1 = edge_index1.cpu()
    edge_index2 = edge_index2.cpu()

    adj1 = sp.coo_matrix((torch.ones(edge_index1.shape[1]), (edge_index1[0, :], edge_index1[1, :])),
                        shape=(x_1.shape[0], x_1.shape[0]),
                        dtype=np.float32)
    adj2 = sp.coo_matrix((torch.ones(edge_index2.shape[1]), (edge_index2[0, :], edge_index2[1, :])),
                        shape=(x_1.shape[0], x_1.shape[0]),
                        dtype=np.float32)

    adj1 = sparse_mx_to_torch_sparse_tensor(adj1).to(device)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to(device)
    
    # classification loss
    loss_1 = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # regularization loss
    norm_1 = torch.norm(x_1, p=2, dim=1).reshape(-1, 1)
    norm_2 = torch.norm(x_2, p=2, dim=1).reshape(-1, 1)
    # calculate the similarity of outgoing and incoming.
    # mask denotes the calculation is just associated to train examples.
    mm = torch.mm(x_1/norm_1, (x_2/norm_2).t()).to(device)
    mm_mask = torch.mm(x_1/norm_1, (x_2/norm_2).t())[data.train_mask, :][:, data.train_mask].to(device)
    # calculate the negative log-likelihood
    likelihood = - torch.mul(mm, adj1.to_dense()) + torch.log(1 + torch.exp(mm))
    likelihood_mask = - torch.mul(mm_mask, adj1.to_dense()[data.train_mask, :][:, data.train_mask]) + torch.log(1 + torch.exp(mm_mask))
    loss_2_total = torch.mean(likelihood)
    loss_2_mask = torch.mean(likelihood_mask)

    if loss_mask:
        loss_2 = loss_2_mask
    else:
        loss_2 = loss_2_total

    if lam > 0:
        loss = loss_1 + lam * loss_2
    else:
        loss = loss_1
#########################
    loss.backward()
    optimizer.step()


def evaluate(model, lam, loss_mask, data):
    model.eval()

    with torch.no_grad():
        logits, x_1, x_2 = model(data)
        label = data.y

        norm_1 = torch.norm(x_1, p=2, dim=1).reshape(-1, 1)
        norm_2 = torch.norm(x_2, p=2, dim=1).reshape(-1, 1)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]

        ###################
        edge_index = data.edge_index
        device = edge_index.device
        edge_index = edge_index.cpu()
        adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                            shape=(x_1.shape[0], x_1.shape[0]),
                            dtype=np.float32)
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

        mm = torch.mm(x_1/norm_1, (x_2/norm_2).t()).to(device)
        mm_mask = torch.mm(x_1/norm_1, (x_2/norm_2).t())[mask, :][:, mask].to(device)
        likelihood = -torch.mul(mm, adj.to_dense()) + torch.log(1 + torch.exp(mm))
        likelihood_mask = -torch.mul(mm_mask, adj.to_dense()[mask, :][:, mask]) + torch.log(1 + torch.exp(mm_mask))
        loss_2_total = torch.mean(likelihood).item()
        loss_2_mask = torch.mean(likelihood_mask).item()
        ###################
        loss_1 = F.nll_loss(logits[mask], data.y[mask]).item()

        if loss_mask:
            loss_2 = loss_2_mask
        else:
            loss_2 = loss_2_total

        if lam > 0:
            loss = loss_1 + lam * loss_2
        else:
            loss = loss_1
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_closs'.format(key)] = loss_1
        outs['{}_regloss'.format(key)] = loss_2
        outs['{}_acc'.format(key)] = acc

    return outs
