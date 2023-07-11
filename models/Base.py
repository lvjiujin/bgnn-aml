import itertools
import torch
import sys
from sklearn import preprocessing
import pandas as pd
import torch.nn.functional as F
import numpy as np
import json

from sklearn.metrics import r2_score, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 试验下focal loss 看下效果如何，有没有变好。
# R2 刻画的是回归的好坏，ρ刻画的是两个变量的线性关系。R2 不满足交换律
# 对的，就是要思考一个问题，Base.py 中self.model从何而来的？BGNN中已经知道了有这个self.modle属性或者函数吧。
class BaseModel(torch.nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def logit_adjustment(self,  y_pred, y_true, tau = 1.0,prior=np.array([0.9023924061506743, 0.09760759384932566])):
        """
        ratio_1 =  0.09760759384932566
        ratio_0 =  0.9023924061506743

        """
        #  prior 是通过对原始的全量样本做极大似然估计算出来的。

        log_prior = torch.Tensor(np.log(prior + 1e-8))
        for _ in range(log_prior.dim() - 1):
            log_prior = log_prior.unsqueeze(0)
        # y_pred = torch.tensor(y_pred).float()
        # y_true = t.tensor(y_true)
        y_pred = y_pred + tau * log_prior
        print("after prior, y_pred.shape = ", y_pred.shape)

        # loss = torch.nn.CrossEntropyLoss()(y_pred, y_true.argmax(dim=-1))
        loss = F.cross_entropy(y_pred, y_true)
        # pytorch实现的logits_adjustment最后的loss是平均loss。

        return loss

    def pandas_to_torch(self, *args):
        return [torch.from_numpy(arg.to_numpy(copy=True)).float().squeeze().to(self.device) for arg in args]

    def networkx_to_torch(self, networkx_graph):
        import dgl
        # graph = dgl.DGLGraph()
        graph = dgl.from_networkx(networkx_graph)
        # 转化成dgl表示的图以后重新添加自循环。
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(self.device)
        return graph

    def networkx_to_torch2(self, networkx_graph):
        from torch_geometric.utils import convert
        import torch_geometric.transforms as T
        graph = convert.from_networkx(networkx_graph)
        transform = T.Compose([T.TargetIndegree()])
        graph = transform(graph)
        return graph.to(self.device)

    def move_to_device(self, *args):
        return [arg.to(self.device) for arg in args]

    def init_optimizer(self, node_features, optimize_node_features, learning_rate):

        params = [self.model.parameters()]
        if optimize_node_features:
            params.append([node_features])
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=learning_rate)
        return optimizer

    def log_epoch(self, pbar, metrics, epoch, loss, epoch_time, logging_epochs, metric_name='loss'):
        train_rmse, val_rmse, test_rmse = metrics[metric_name][-1]
        if epoch and epoch % logging_epochs == 0:
            pbar.set_description(
                "Epoch {:05d} | Loss {:.3f} | Loss {:.3f}/{:.3f}/{:.3f} | Time {:.4f}".format(epoch, loss,
                                                                                              train_rmse,
                                                                                              val_rmse, test_rmse,
                                                                                              epoch_time))

    def normalize_features(self, X, train_mask, val_mask, test_mask):
        min_max_scaler = preprocessing.MinMaxScaler()
        A = X.to_numpy(copy=True)
        A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
        A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
        return pd.DataFrame(A, columns=X.columns).astype(float)

    def replace_na(self, X, train_mask):
        # 第一个any() 是按照列来进行检查，第二个any()是在第一个any()的基础上进行整体检查看是否有null的。
        if X.isna().any().any():
            return X.fillna(X.iloc[train_mask].min() - 1)
        return X

    def encode_cat_features(self, X, y, cat_features, train_mask, val_mask, test_mask):
        from category_encoders import CatBoostEncoder
        enc = CatBoostEncoder()
        A = X.to_numpy(copy=True)
        b = y.to_numpy(copy=True)
        A[np.ix_(train_mask, cat_features)] = enc.fit_transform(A[np.ix_(train_mask, cat_features)], b[train_mask])
        A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(A[np.ix_(val_mask + test_mask, cat_features)])
        A = A.astype(float)
        return pd.DataFrame(A, columns=X.columns)

    # 这个地方可以改成focal loss 或者权重交叉熵
    def train_model(self, model_in, target_labels, train_mask, optimizer, class_weights=None):
        y = target_labels[train_mask]
        # print("base.py y.shape = ", y.shape)
        # print("base.py type(y) = ", type(y))
        self.model.train()
        logits = self.model(*model_in).squeeze()
        pred = logits[train_mask]
        # print("base.py train_model pred.shape = ", pred.shape)
        # print("base.py train_model type(pred) = ", type(pred))
        # print("base.py train_model pred[:5] = ", pred[:5])

        if self.task == 'regression':
            loss = torch.sqrt(F.mse_loss(pred, y))
        elif self.task == 'classification':
            # 这个地方修改成权重交叉熵
            # loss = F.cross_entropy(pred, y.long())
            # 感觉focal loss 中α也是超参数，不知道到底要设置多少会比较好呢？
            # 就相当权重交叉熵中的class_weights的参数一样，需要超参数进行调优，针对不同的数据集确定一个好的超参数。
            # class_weights = torch.Tensor([0.09760759384932566, 0.9023924061506743])
            loss = F.cross_entropy(pred, y.long(), weight=class_weights)

            # 对于elliptic dataset数据集是二分类的问题，可以使用binary_cross_entropy 这个函数。
            # 再去看下代码，确认下，是否0代表负样本，1代表正样本。
            # loss = F.binary_cross_entropy_with_logits(pred, y.long(), weight=class_weights)
            # F.binary_cross_entropy 和F.binary_cross_entropy_with_logits 跑不起来，维度不匹配
            # loss = F.binary_cross_entropy(pred, y.long(), weight=class_weights)
            # 老怀疑我是不是用错了，为何我的权重交叉熵，越用效果越差，logit_adjust 也是这样的，越用效果越差。
            # loss = self.logit_adjustment(pred, y.long())

        else:
            raise NotImplemented("Unknown task. Supported tasks: classification, regression.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def evaluate_model(self, logits, target_labels, mask, class_weights=None):
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = logits[mask]
            if self.task == 'regression':
                metrics['loss'] = torch.sqrt(F.mse_loss(pred, y).squeeze() + 1e-8)
                metrics['rmsle'] = torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8)
                metrics['mae'] = F.l1_loss(pred, y)
                metrics['r2'] = torch.Tensor([r2_score(y.cpu().numpy(), pred.cpu().numpy())])

            elif self.task == 'classification':
                # 评价一个模型的时候，是否要用到类别权重参数呢？
                metrics['loss'] = F.cross_entropy(pred, y.long(), weight=class_weights)
                # pred.max(1) 求的是每一行的最大值。pred.max(0)求的是每一列的最大值。
                # pred.max(1)[1] 得出的是每一行最大值的索引的位置（针对的是torch.Tensor类型）。索引刚好可以代表0,1类别。
                y_pred = pred.max(1)[1]
                # metrics['accuracy'] = torch.Tensor([(y == pred.max(1)[1]).sum().item()/y.shape[0]])
                metrics['accuracy'] = torch.Tensor([accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())])
                # 在数据集不平衡时，准确率将不能很好地表示模型的性能。
                # 可能会存在准确率很高，而少数类样本全分错的情况，此时应选择其它模型评价指标，针对反洗钱的问题，这里选择的是f1
                metrics['precision'] = torch.Tensor([precision_score(y.cpu().numpy(), y_pred.cpu().numpy(),
                                                                     zero_division=0)])
                metrics['recall'] = torch.Tensor([recall_score(y.cpu().numpy(), y_pred.cpu().numpy())])
                metrics['f1'] = torch.Tensor([f1_score(y.cpu().numpy(), y_pred.cpu().numpy())])
                # auc 的计算也很有必要
                metrics['auc'] = torch.Tensor([roc_auc_score(y.cpu().numpy(), y_pred.cpu().numpy())])

            return metrics

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_and_evaluate(self, model_in, target_labels, train_mask, val_mask, test_mask,
                           optimizer, metrics, gnn_passes_per_epoch, class_weights=None):
        loss = None
        # 每一个epoch, gnn 要迭代gnn_passes_per_epoch次数。
        # 相当于针对GBDT提供的参数，gnn要训练gnn_passes_per_epoch次数。
        for _ in range(gnn_passes_per_epoch):
            loss = self.train_model(model_in, target_labels, train_mask, optimizer, class_weights)

        # 对的，这个地方非常奇怪，这个self.model是从哪里赋值的？ 这个不行的话可以问下作者，到底是为啥？
        #  self.model =  <class 'bgnn.models.GNN.GNNModelDGL'>
        # 测试的结果表明，这是通过子类来赋值的，这种用法是正确的。小玉的观点这是一种设计模式。
        # 每进行一次epoch, 验证集和测试集也要跟着执行一次。

        self.model.eval()
        logits = self.model(*model_in).squeeze()

        train_results = self.evaluate_model(logits, target_labels, train_mask, class_weights)
        val_results = self.evaluate_model(logits, target_labels, val_mask, class_weights)
        test_results = self.evaluate_model(logits, target_labels, test_mask)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                                        val_results[metric_name].detach().item(),
                                        test_results[metric_name].detach().item()))
        # 这个地方可以将logits 返回，抓取到,传给下一轮的GNN
        return loss, logits

    # 这个函数必须要搞明白，非常关键的。
    def update_early_stopping(self, metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                              metric_name, lower_better=False):
        train_metric, val_metric, test_metric = metrics[metric_name][-1]
        # lower_better means the lower the better such as the loss value.
        if (lower_better and val_metric < best_metric[1]) or (not lower_better and val_metric > best_metric[1]):
            best_metric = metrics[metric_name][-1]
            best_val_epoch = epoch
            epochs_since_last_best_metric = 0
        else:
            epochs_since_last_best_metric += 1
        return best_metric, best_val_epoch, epochs_since_last_best_metric


    def save_metrics(self, metrics, fn):
        with open(fn, "w+") as f:
            json.dump(metrics, f, indent=2)

    def plot(self, metrics, legend, title, output_fn=None, logx=False, logy=False, metric_name='loss'):
        import matplotlib.pyplot as plt
        metric_results = metrics[metric_name]
        xs = [range(len(metric_results))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        plt.rcParams.update({'font.size': 40})
        plt.rcParams["figure.figsize"] = (20, 10)
        lss = ['-', '--', '-.', ':']
        colors = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
        colors = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2),
                  (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
        colors = [[p / 255 for p in c] for c in colors]
        for i in range(len(ys)):
            plt.plot(xs[i], ys[i], lw=4, color=colors[i])
        plt.legend(legend, loc=1, fontsize=30)
        plt.title(title)

        plt.xscale('log') if logx else None
        plt.yscale('log') if logy else None
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.grid()
        plt.tight_layout()

        plt.savefig(output_fn, bbox_inches='tight', format='svg') if output_fn else None
        plt.show()

    def plot_interactive(self, metrics, legend, title, logx=False, logy=False, metric_name='loss', start_from=0):
        import plotly.graph_objects as go
        metric_results = metrics[metric_name]
        xs = [list(range(len(metric_results)))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        fig = go.Figure()
        for i in range(len(ys)):
            fig.add_trace(go.Scatter(x=xs[i][start_from:], y=ys[i][start_from:],
                                     mode='lines+markers',
                                     name=legend[i]))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Epoch',
            yaxis_title='RMSE',
            font=dict(
                size=40,
            ),
            height=600,
        )

        if logx:
            fig.update_layout(xaxis_type="log")
        if logy:
            fig.update_layout(yaxis_type="log")

        fig.show()


def main():
    base_model = BaseModel()
    p = base_model.model.parameters()
    print("base_model.model = ", p)

if __name__ == '__main__':
    main()