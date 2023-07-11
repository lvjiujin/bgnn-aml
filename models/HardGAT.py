"""
Graph Representation Learning via Hard Attention Networks in DGL using Adam optimization.
References
----------
Paper: https://arxiv.org/abs/1907.04652
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.sampling import select_topk
from functools import partial
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
from dgl.base import DGLError
import dgl
import argparse
import numpy as np
import time
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset



class HardGAO(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads=8,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=True,
                 activation=F.elu,
                 k=8,):
        super(HardGAO, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k
        self.residual = residual
        # Initialize Parameters for Additive Attention
        self.fc = nn.Linear(
            self.in_feats, self.out_feats * self.num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, self.out_feats)))
        # Initialize Parameters for Hard Projection
        self.p = nn.Parameter(torch.FloatTensor(size=(1,in_feats)))
        # Initialize Dropouts
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self.residual:
            if self.in_feats == self.out_feats:
                self.residual_module = Identity()
            else:
                self.residual_module = nn.Linear(self.in_feats,self.out_feats*num_heads,bias=False)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.p,gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.residual:
            nn.init.xavier_normal_(self.residual_module.weight,gain=gain)

    def forward(self, graph, feat, get_attention=False):
            # Check in degree and generate error
            if (graph.in_degrees()==0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            # projection process to get importance vector y
            graph.ndata['y'] = torch.abs(torch.matmul(self.p,feat.T).view(-1))/torch.norm(self.p,p=2)
            # Use edge message passing function to get the weight from src node
            graph.apply_edges(fn.copy_u('y','y'))
            # Select Top k neighbors
            subgraph = select_topk(graph.cpu(),self.k,'y').to(graph.device)
            # Sigmoid as information threshold
            subgraph.ndata['y'] = torch.sigmoid(subgraph.ndata['y'])
            # Using vector matrix elementwise mul for acceleration
            feat = subgraph.ndata['y'].view(-1,1)*feat
            feat = self.feat_drop(feat)
            h = self.fc(feat).view(-1, self.num_heads, self.out_feats)
            el = (h * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (h * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # Assign the value on the subgraph
            subgraph.srcdata.update({'ft': h, 'el': el})
            subgraph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            subgraph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(subgraph.edata.pop('e'))
            # compute softmax
            subgraph.edata['a'] = self.attn_drop(edge_softmax(subgraph, e))
            # message passing
            subgraph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = subgraph.dstdata['ft']
            # activation
            if self.activation:
                rst = self.activation(rst)
            # Residual
            if self.residual:
                rst = rst + self.residual_module(feat).view(feat.shape[0],-1,self.out_feats)

            if get_attention:
                return rst, subgraph.edata['a']
            else:
                return rst

class HardGAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 k):
        super(HardGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        gat_layer = partial(HardGAO,k=k)
        muls = heads
        # input projection (no residual)
        self.gat_layers.append(gat_layer(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(gat_layer(
                num_hidden*muls[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(gat_layer(
            num_hidden*muls[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, False, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

def main(args):
    # load and preprocess dataset
    print("args.data = ", args.dataset)
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.num_layers <= 0:
        raise ValueError("num layer must be positive int")
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data[0].number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    print("args.num_heads = ", args.num_heads)
    print("args.num_out_heads = ", args.num_out_heads)
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    print("heads = ", heads)
    model = HardGAT(g,
                    args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual,
                    args.k)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    # parser.add_argument("--dataset", type=str, default='cora')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--k', type=int, default=8,
                        help='top k neighor for attention calculation')
    Args = parser.parse_args()
    print(Args)

    main(Args)

