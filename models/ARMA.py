import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from tqdm import trange
import copy
import torch.optim as optim


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class ARMAConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_stacks,
                 num_layers,
                 activation=None,
                 dropout=0.0,
                 bias=True):
        super(ARMAConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = num_stacks
        self.T = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # init weight
        self.w_0 = nn.ModuleDict({
            str(k): nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K)
        })
        # deeper weight
        self.w = nn.ModuleDict({
            str(k): nn.Linear(out_dim, out_dim, bias=False) for k in range(self.K)
        })
        # v
        self.v = nn.ModuleDict({
            str(k): nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K)
        })
        # bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.K, self.T, 1, self.out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            glorot(self.w_0[str(k)].weight)
            glorot(self.w[str(k)].weight)
            glorot(self.v[str(k)].weight)
        zeros(self.bias)

    def forward(self, g, feats):
        with g.local_scope():
            init_feats = feats
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            output = None

            for k in range(self.K):
                feats = init_feats
                for t in range(self.T):
                    feats = feats * norm
                    g.ndata['h'] = feats
                    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feats = g.ndata.pop('h')
                    feats = feats * norm

                    if t == 0:
                        feats = self.w_0[str(k)](feats)
                    else:
                        feats = self.w[str(k)](feats)
                    
                    feats += self.dropout(self.v[str(k)](init_feats))
                    feats += self.v[str(k)](self.dropout(init_feats))

                    if self.bias is not None:
                        feats += self.bias[k][t]
                    
                    if self.activation is not None:
                        feats = self.activation(feats)
                    
                if output is None:
                    output = feats
                else:
                    output += feats
                
            return output / self.K 

class ARMA4NC(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_stacks,
                 num_layers,
                 activation=None,
                 dropout=0.0):
        super(ARMA4NC, self).__init__()

        self.conv1 = ARMAConv(in_dim=in_dim,
                              out_dim=hid_dim,
                              num_stacks=num_stacks,
                              num_layers=num_layers,
                              activation=activation,
                              dropout=dropout)

        self.conv2 = ARMAConv(in_dim=hid_dim,
                              out_dim=out_dim,
                              num_stacks=num_stacks,
                              num_layers=num_layers,
                              activation=activation,
                              dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, feats):
        feats = F.relu(self.conv1(g, feats))
        feats = self.dropout(feats)
        feats = self.conv2(g, feats)
        return feats


def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    if args.dataset == 'Cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'Citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'Pubmed':
        dataset = PubmedGraphDataset()
    else:
        raise ValueError('Dataset {} is invalid.'.format(args.dataset))

    graph = dataset[0]

    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # retrieve the number of classes
    n_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata.pop('label').to(device).long()

    # Extract node features
    feats = graph.ndata.pop('feat').to(device)
    n_features = feats.shape[-1]

    # retrieve masks for train/validation/test
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    graph = graph.to(device)

    # Step 2: Create model =================================================================== #
    model = ARMA4NC(in_dim=n_features,
                    hid_dim=args.hid_dim,
                    out_dim=n_classes,
                    num_stacks=args.num_stacks,
                    num_layers=args.num_layers,
                    activation=nn.ReLU(),
                    dropout=args.dropout).to(device)

    best_model = copy.deepcopy(model)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamb)

    # Step 4: training epoches =============================================================== #
    acc = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc='Accuracy & Loss')

    for _ in epochs:
        # Training using a full graph
        model.train()

        logits = model(graph, feats)

        # compute loss
        train_loss = loss_fn(logits[train_idx], labels[train_idx])
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

        # backward
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        # Validation using a full graph
        model.eval()

        with torch.no_grad():
            valid_loss = loss_fn(logits[val_idx], labels[val_idx])
            valid_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

        # Print out performance
        epochs.set_description('Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}'.format(
            train_acc, train_loss.item(), valid_acc, valid_loss.item()))

        if valid_acc < acc:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print('Early stop.')
                break
        else:
            no_improvement = 0
            acc = valid_acc
            best_model = copy.deepcopy(model)

    best_model.eval()
    logits = best_model(graph, feats)
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc {:.4f}".format(test_acc))
    return test_acc


if __name__ == "__main__":
    """
    ARMA Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='ARMA GCN')

    # data source params
    parser.add_argument('--dataset', type=str, default='Cora', help='Name of dataset.')
    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs.')
    parser.add_argument('--early-stopping', type=int, default=100, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=5e-4, help='L2 reg.')
    # model params
    parser.add_argument("--hid-dim", type=int, default=16, help='Hidden layer dimensionalities.')
    parser.add_argument("--num-stacks", type=int, default=2, help='Number of K.')
    parser.add_argument("--num-layers", type=int, default=1, help='Number of T.')
    parser.add_argument("--dropout", type=float, default=0.75, help='Dropout applied at all layers.')

    args = parser.parse_args()
    print(args)

    acc_lists = []

    for _ in range(100):
        acc_lists.append(main(args))

    mean = np.around(np.mean(acc_lists, axis=0), decimals=3)
    std = np.around(np.std(acc_lists, axis=0), decimals=3)
    print('Total acc: ', acc_lists)
    print('mean', mean)
    print('std', std)