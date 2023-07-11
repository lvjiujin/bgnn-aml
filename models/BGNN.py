import itertools
import time
import numpy as np
import torch
import os

from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from .GNN import GNNModelDGL, GATDGL
from .Base import BaseModel
from tqdm import tqdm
import pickle
from collections import defaultdict as ddict


class BGNN(BaseModel):
    def __init__(self,
                 task='regression', iter_per_epoch = 10, lr=0.01, hidden_dim=64, dropout=0.,
                 only_gbdt=False,  train_non_gbdt=False, gnn_residual=False, name='gat',
                 use_leaderboard=False, depth=6, gbdt_lr=0.1, ex_input_feat=False,
                 in_feat=False, all_orig_feat=False, class_weights=None):
        super(BaseModel, self).__init__()
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.dropout = dropout
        self.only_gbdt = only_gbdt
        self.class_weights = class_weights
        self.all_orig_feat = all_orig_feat
        self.in_feat = in_feat
        self.ex_input_feat = ex_input_feat
        self.train_residual = train_non_gbdt
        self.gnn_residual = gnn_residual
        self.name = name
        self.use_leaderboard = use_leaderboard # use_leaderboard 这个参数有何意义？有长教训了，明白了要记下来，否则下次又不明白了。
        self.iter_per_epoch = iter_per_epoch
        self.depth = depth
        self.lang = 'dgl'
        self.gbdt_lr = gbdt_lr

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        return 'BGNN'

    def init_gbdt_model(self, num_epochs, epoch):
        if self.task == 'regression':
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'RMSE' #''RMSEWithUncertainty'
        else:
            # 这个地方这样处理是为什么？不是回归就是分类呀，对于分类任务epoch == 0 cat_boost_model_ojb = CatBoostClassifier.
            # 否则cat_boost_model_obj == CatBoostRegressor, 这是为何？ 损失函数也有区别，why？
            # 明白了，为什么用回归，这是因为为了将得到的特征输入图神经网络，进行图分类。而不是直接通过gbdt得到分类结果。
            if epoch == 0:
                catboost_model_obj = CatBoostClassifier
                catboost_loss_fn = 'MultiClass'
            else:
                catboost_model_obj = CatBoostRegressor
                catboost_loss_fn = 'MultiRMSE'

        if self.task != 'regression' and epoch == 0 and self.class_weights:
            return catboost_model_obj(iterations=num_epochs,
                                      depth=self.depth,
                                      class_weights=self.class_weights,
                                      learning_rate=self.gbdt_lr,
                                      loss_function=catboost_loss_fn,
                                      random_seed=0,
                                      nan_mode='Min')
        else:
            return catboost_model_obj(iterations=num_epochs,
                                      depth=self.depth,
                                      learning_rate=self.gbdt_lr,
                                      loss_function=catboost_loss_fn,
                                      random_seed=0,
                                      nan_mode='Min')



    def fit_gbdt(self, pool, trees_per_epoch, epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch, epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def init_gnn_model(self):
        if self.use_leaderboard:
            self.model = GATDGL(in_feats=self.in_dim, n_classes=self.out_dim).to(self.device)
        else:
            self.model = GNNModelDGL(in_dim=self.in_dim,
                                     hidden_dim=self.hidden_dim,
                                     out_dim=self.out_dim,
                                     name=self.name,
                                     dropout=self.dropout).to(self.device)

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha):

        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch, epoch)
        # 是否在这里提取特征重要性呢？如何不提取
        """
        可以在这里提取特征重要性, 同时添加到X'中，也就是X‘ = f(X) + feature_select (order by feature importance)
        fea_ = cat_model.feature_importances_
        fea_name = cat_model.feature_names_
        """

        if epoch == 0 and self.task == 'classification':
            self.base_gbdt = epoch_gbdt_model
        else:
            self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha])

    def update_node_features(self, node_features, X, encoded_X, in_features, epoch, gnn_logits):
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
            # predictions = self.gbdt_model.virtual_ensembles_predict(X,
            #                                                         virtual_ensembles_count=5,
            #                                                         prediction_type='TotalUncertainty')
        else:
            # self.base_gbdt是一个catboost的分类器
            # predictions 是每一行为一个二维向量的概率值，相加之和为1.
            predictions = self.base_gbdt.predict_proba(X)

            # print("before1 predictions.shape = ", predictions.shape)
            # predictions = self.base_gbdt.predict(X, prediction_type='RawFormulaVal')
            if self.gbdt_model is not None:
                predictions_after_one = self.gbdt_model.predict(X)
                predictions += predictions_after_one
                # 这一步是gbdt的精髓所在，也就是累加目标值。
        # print("predictions.shape = ", predictions.shape)
        # print("type(predictions) = ", type(predictions))
        # print("predictions[:10] = ", predictions[:10])

        # 对的，可以直接在这里添加通过模型筛选的特征重要性，进行特征提取。
        if self.feature_importance:
            print("before concat predictions.dtype = ", predictions.dtype)
            print("before concat type(encoded_X) = ", type(encoded_X))
            print("befoere concat encoded_X.shape = ", encoded_X.shape)
            print("before concat type(predictions) = ", type(predictions))
            print("before concat predictions.shape = ", predictions.shape)

            if self.task == 'classification':

                feature_importances = self.base_gbdt.feature_importances_
                print("feature_importance from base_gbdt = ", feature_importances)
                print("from base_gbdt type(feature_importances) = ", type(feature_importances))
                print("from base_gbdt feature_importances.shape = ", feature_importances.shape)

                # if self.gbdt_model is not None:
                #
                #     # 这个地方还是要亲自验证下才知道。这个地方有问题，暂时关闭掉。
                #     feature_importances_after_one = self.gbdt_model.feature_importances_
                #     print("feature_importances_after_one from gbdt_model = ", type(feature_importances_after_one))
                #     # 这个地方想一下，为何提取不到feature_importances, 而且为空？
                #     feature_importances += feature_importances_after_one
                #     # feature_importances = feature_importances_after_one
                #
                #     print("from gbdt_model type(feature_importances) = ", type(feature_importances))
                #     print("from gbdt_model feature_importances.shape = ", feature_importances.shape)
                # 这个地方根据特征重要性来逆序筛选
                feat_imp_indices = np.argsort(feature_importances)[::-1]
                print("feat_imp_indices = ", feat_imp_indices)
                feat_imp_indices_top = feat_imp_indices[:self.feature_select_top_num]
                print("feat_imp_indices_top = ", feat_imp_indices_top)
                important_features = encoded_X.iloc[:, feat_imp_indices_top]
                print("type(important_features = ", type(important_features))
                print("important_features.shape = ", important_features.shape)
                # append important_features to prediction

                predictions = np.append(important_features, predictions, axis=1)
                predictions = predictions.astype('float64')
                print("after concat predictions.shape = ", predictions.shape)
                print("after concat predictions.dtype = ", predictions.dtype)
                print("after concat type(predictions) = ", type(predictions))


        if self.in_feat:
            # 模仿only_gbdt的写法。
            predictions = np.append(in_features, predictions, axis=1)  # append X to prediction
        # 是否将gnn的结果回传
        if self.gnn_residual:
            if epoch == 0:
                predictions = np.append(predictions, predictions, axis=1)
            else:
                predictions = np.append(gnn_logits.detach().numpy(), predictions, axis=1) # append gnn_logits to prediction


        if not self.only_gbdt:
            # 终于知道only_gbdt参数的含义了，如果only_gbdt==True, 就是特征仅仅包含通过gbdt生成的特征，
            # 如果only_gbdt==False, 特征就是原特征和gbdt特征拼接的结果。
            # 想一下，self.train_residual 这个参数的作用到底是干什么的。
            print("before2 predictions.shape = ", predictions.shape)
            if self.train_residual:
                predictions = np.append(node_features.detach().cpu().data[:, :-self.out_dim], predictions,
                                        axis=1)  # append updated X to prediction
            else:
                predictions = np.append(encoded_X, predictions, axis=1)  # append X to prediction
            # print("type(encoded_X) = ", type(encoded_X))
            # print("type(predictions) = ", type(predictions))
            # print("encoded_X.shape = ", encoded_X.shape)
            # print("predictions.shape = ", predictions.shape)
        predictions = torch.from_numpy(predictions).to(self.device)

        node_features.data = predictions.float().data

    def update_gbdt_targets(self, node_features, node_features_before, train_mask):
        # 这一步就是论文中所提到的：GBDT的new target label (残差）X'_new - X'
        # 这个地方维度 要清晰要搞明白。
        return (node_features - node_features_before).detach().cpu().numpy()[train_mask, -self.out_dim:]

    def init_node_features(self, X):
        # 先将节点特征的形状给设计出来。

        node_features = torch.empty(X.shape[0], self.in_dim, requires_grad=True, device=self.device)
        if not self.only_gbdt:
            node_features.data[:, :-self.out_dim] = torch.from_numpy(X.to_numpy(copy=True))
        return node_features

    def init_node_parameters(self, num_nodes):
        return torch.empty(num_nodes, self.out_dim, requires_grad=True, device=self.device)

    def init_optimizer2(self, node_parameters, learning_rate):
        params = [self.model.parameters(), [node_parameters]]
        return torch.optim.Adam(itertools.chain(*params), lr=learning_rate)

    def update_node_features2(self, node_parameters, X):
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
        else:
            predictions = self.base_gbdt.predict_proba(X)
            if self.gbdt_model is not None:
                predictions += self.gbdt_model.predict(X)

        predictions = torch.from_numpy(predictions).to(self.device)
        node_parameters.data = predictions.float().data

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, cat_features,
            in_features, num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True, feature_importance=False, feature_select_top_num=40
            ):

        # print("X.shape = ", X.shape)

        # initialize for early stopping and metrics
        # metric_name 还是非常关键的，它决定了，到底是按照哪一个metric进行eary_stopping。
        if metric_name in ['r2', 'accuracy', 'f1']:
            best_metric = [np.float('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float('inf')] * 3  # for train/val/test
        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)
        if cat_features is None:
            cat_features = []

        if self.task == 'regression':
            self.out_dim = y.shape[1]
        elif self.task == 'classification':
            # 转化成集合，去掉重复值。类别只保留唯一值。真是见名知意，out_dim 就是输出的维度的大小。
            self.out_dim = len(set(y.iloc[test_mask, 0]))
        print("self.out_dim = ", self.out_dim)
        # self.in_dim = X.shape[1] if not self.only_gbdt else 0
        # self.in_dim += 3 if uncertainty else 1
        # 这个地方我终于知道是为什么了，如果是only_gbdt 的话，那么 in_dim = out_dim, 这是因为only_gbdt不需要拼接原来的特征X。
        # 如果only_gbdt==False的话，需要拼接原来的特征, 就是将gbdt生成的特征和X的特征进行concat起来,送入图神经网络。
        self.in_dim = self.out_dim + X.shape[1] if not self.only_gbdt else self.out_dim
        # print("in_dim = ", self.in_dim)
        # 在这个地方修改一下：添加in_features的形状

        self.in_dim = self.in_dim + in_features.shape[1] if self.in_feat else self.in_dim
        #
        #
        self.feature_importance = feature_importance
        self.feature_select_top_num = feature_select_top_num

        self.in_dim = self.in_dim + self.feature_select_top_num if self.feature_importance else self.in_dim

        self.in_dim = self.in_dim + self.out_dim if self.gnn_residual else self.in_dim

        # print("in_dim_new = ", self.in_dim) # in_dim_new = 4

        self.init_gnn_model() # gnn的模型构建起来了，用的是gat。

        # 开始进入gbdt了，马上要进入模型训练环节了，注意着要和论文中的算法对应起来。
        gbdt_X_train = X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        gbdt_alpha = 1
        self.gbdt_model = None

        encoded_X = X.copy()
        if not self.only_gbdt:
            if len(cat_features):
                encoded_X = self.encode_cat_features(encoded_X, y, cat_features, train_mask, val_mask, test_mask)
            if normalize_features:
                encoded_X = self.normalize_features(encoded_X, train_mask, val_mask, test_mask)
            if replace_na:
                encoded_X = self.replace_na(encoded_X, train_mask)
        # 先将节点特征的形状给设计出来。
        # 绘制节点特征开始:
        # print("begin 绘制节点特征----- ")
        # print("line 303 node_features.shape = ", node_features.shape)
        node_features = self.init_node_features(encoded_X)
        # print("node_features[:5] = ", node_features[:5])
        # print("line 306 , node_features.shape = ", node_features.shape)
        optimizer = self.init_optimizer(node_features, optimize_node_features=True, learning_rate=self.learning_rate)

        y, = self.pandas_to_torch(y)
        self.y = y
        if self.lang == 'dgl':
            graph = self.networkx_to_torch(networkx_graph)
        elif self.lang == 'pyg':
            graph = self.networkx_to_torch2(networkx_graph)
        else:
            # 先这样写，后续会增加 其余图神经网络框架的实现。
            graph = self.networkx_to_torch2(networkx_graph)

        self.graph = graph

        pbar = tqdm(range(num_epochs))
        # config中设置的num_epochs 为200。
        save_logits = None
        print("line 319 , node_features.shape = ", node_features.shape) # 4 * 203769
        for epoch in pbar:
            # 一个epoch就是将所有训练样本训练一次的过程。
            start2epoch = time.time()

            # gbdt part
            self.train_gbdt(gbdt_X_train, gbdt_y_train, cat_features, epoch,
                            self.iter_per_epoch, gbdt_alpha)
            # update_node_features 这一步也就是算法中的由 f(X) -> X' 得到X'。 两种情况由参数only_gbdt来控制。

            self.update_node_features(node_features, X, encoded_X, in_features, epoch, save_logits)
            # 直接在这里修改也是可以的。看下效果如何。
            # 是否要增加一个参数呢？就像之前in_feat一样的参数。还是说直接写。
            # 不要搞那么多参数，直接上吧。


            node_features_before = node_features.clone()
            # print("type(node_features) = ", type(node_features))
            # print("node_features = ", node_features.shape)
            # print("node_features[:5] = ", node_features[:5])
            """"
            type(node_features) =  <class 'torch.Tensor'>
            node_features =  torch.Size([203769, 2])
            node_features[:5] =  tensor([[0.8964, 0.1036],
            [0.8912, 0.1088],
            [0.9431, 0.0569],
            [0.9651, 0.0349],
            [0.9443, 0.0557]], grad_fn=<SliceBackward>)
            """


            model_in = (graph, node_features)
            # print("line 355 , node_features.shape = ", node_features.shape)

            # GNN part
            # 每次判断下，看下save_logits 是否为None, 实际上只有第一次为None。

                # 重新整一下model_in
            # loss, logits = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask,
            #                                optimizer, metrics, self.iter_per_epoch)
            loss, _ = self.train_and_evaluate(model_in, y, train_mask,
                                                   val_mask, test_mask,
                                                    optimizer, metrics, self.iter_per_epoch)

            # save_logits = logits

            """
            type(logits) =  <class 'torch.Tensor'>
            logits.shape =  torch.Size([203769, 2])
            logits[:5] =  tensor([[ 3.1660, -0.2215],
            [ 3.5088, -0.3556],
            [ 3.5416, -0.3660],
            [ 3.6557, -0.4092],
            [ 3.8249, -0.4916]], grad_fn=<SliceBackward>)
            """
            # print("type(logits) = ", type(logits))
            # print("logits.shape = ", logits.shape)
            # print("logits[:5] = ", logits[:5])
            # save_logits = logits
            # 这个地方就是如何提取GNN 的输入
            # 想一下这个如何处理呢？如何处理呢？如何处理呢？

            # 这一步就是论文中所提到的：GBDT的new target label (残差）X'_new - X'
            gbdt_y_train = self.update_gbdt_targets(node_features, node_features_before, train_mask)

            # self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs,
            #                metric_name=metric_name)
            # best_model
            # check early stopping
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                                           metric_name, lower_better=(metric_name not in ['r2', 'accuracy', 'f1']))
            if patience and epochs_since_last_best_metric > patience:
                print("when eary stopping, epoch = ", epoch)
                break
            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Nodes do not change anymore. Stopping... ')
                break
        print("type(metrics) = ", type(metrics))
        # pickle 文件保存下来。

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))

        return metrics

    # def predict(self, graph, X, y, test_mask):
    #     node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
    #     self.update_node_features(node_features, X, X, self.in_features)
    #     return self.evaluate_model((graph, node_features), y, test_mask)
