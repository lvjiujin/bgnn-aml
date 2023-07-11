# from catboost import CatboostError
# import sys
# sys.path.append('../')

import sys
import os
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict as ddict
import pandas as pd
import networkx as nx
import random
import numpy as np
import fire
import ast
from omegaconf import OmegaConf
import pickle
import torch
from sklearn.model_selection import ParameterGrid

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
from models.GBDT import GBDTCatBoost, GBDTLGBM
from models.MLP import MLP
from models.GNN import GNN
from models.BGNN import BGNN
from scripts.utils import NpEncoder

#  NDCG排序相关的指标 NDCG，Normalized Discounted cumulative gain 直接翻译为归一化折损累计增益
class RunModel:
    # 这个后续改写下，给增加一个__init__方法。将属性都放到__init__方法中。
    def read_input(self, input_folder):
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')
        networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
        # 奇怪了，relabel_nodes 这个函数有什么意义吗？
        # networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
        self.networkx_graph = networkx_graph
        categorical_columns = []
        if os.path.exists(f'{input_folder}/cat_features.txt'):
            with open(f'{input_folder}/cat_features.txt') as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())

        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]

            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)

        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)
        else:
            print('Creating and saving train/val/test masks')
            idx = list(range(self.y.shape[0]))
            self.masks = dict()
            # 为了代码的通用性，这里elliptic 的masks.json文件的生成不要放在这里了，放到预处理里面。
            for i in range(self.max_seeds):
                # random.shuffle(idx) 这句代码是点睛之笔，结果的平均就是靠这句话来实现。
                random.shuffle(idx)
                # 最关键的是这个mask要做好。
                # train: val: test = 6:2:2
                r1, r2, r3 = idx[:int(.6 * len(idx))], idx[int(.6 * len(idx)):int(.8 * len(idx))], idx[
                                                                                                   int(.8 * len(idx)):]
                self.masks[str(i)] = {"train": r1, "val": r2, "test": r3}

            with open(f'{input_folder}/masks.json', 'w+') as f:
                json.dump(self.masks, f, cls=NpEncoder)

    def get_input(self, dataset_dir, dataset: str):
        sys_datasets = ('house', 'county', 'vk', 'wiki', 'avazu',
                        'vk_class', 'house_class', 'dblp', 'slap')
        if dataset in sys_datasets:
            input_folder = dataset_dir / dataset
        else:
            input_folder = dataset
        if self.save_folder is None:
            self.save_folder = f'results/{dataset}/{datetime.datetime.now().strftime("%d_%m")}'
        self.read_input(input_folder)
        print('Save to folder:', self.save_folder)

    def run_one_model(self, config_fn, model_name):
        self.config = OmegaConf.load(config_fn)
        grid = ParameterGrid(dict(self.config.hp))
        # 这个地方要修改下，这里将所有的网格参数组合都过一遍，太耗时，想办法改成hyperopt超参数调优。
        # 这个地方有多少组超参数，就会生成多少个文件。
        # 文件要加上日期，同时加上第几次的重复。
        for ps in grid:
            param_string = ''.join([f'-{key}{ps[key]}' for key in ps])
            # 文件名太长了，怎么办？
            exp_name = f'{model_name}{param_string}'
            print(f'\nSeed {self.seed} RUNNING:{exp_name}')
            w = ps["class_weights"]
            if w == "None":
                w = ast.literal_eval(w)
            else:
                w = list(w)
            ps["class_weights"] = w

            # 对的，就在这个位置，在self.define_model之前添加all_orig_feat这个参数的判断。
            if not ps["all_orig_feat"] and "elliptic" in self.dataset:
                self.X = self.X.iloc[:, list(range(94))]
            print("type(self.X) = ", type(self.X))
            # 这个地方直接拼接概率预测的特征。就是可以直接用预训练的模型的特征拼接上原来的特征。
            if ps['ex_input_feat']:
                with open('/root/bgnn/scripts/model4.pkl', 'rb') as fp:
                    _model = pickle.load(fp)
                    # nbn中只适用前面94个自带的特征，不要后来衍生的特征。
                # ex_input_features = ex_model.predict_proba(self.X.iloc[:, list(range(94))])
                ex_input_features = ex_model.transform(self.X.iloc[:, list(range(94))])
                ex_input_columns = [f"local_{i:03d}" for i in range(1, 257)]
                ex_input_features_df = pd.DataFrame(ex_input_features, columns=ex_input_columns)
                self.X = pd.concat([self.X, ex_input_features_df], axis=1)

            # 在这个地方做RBM 的特征预处理。这个地方调用该函数，将这个特征衍生过程写成一个单独的函数。
            # torch 可以直接导出和导入pkl文件。
            # 这个地方就要将in_features 给读取出来。
            if ps['in_feat']:
                with open('/root/bgnn/scripts/dbn_save_model/model1.pkl', 'rb') as fp:
                    ex_model = pickle.load(fp)
                    # nbn中只适用前面94个自带的特征，不要后来衍生的特征。
                self.in_features = ex_model.predict_proba(self.X.iloc[:, list(range(94))])

            else:
                self.in_features = None

            # repeat_exp 这个参数就是针对同一组参数，重复训练的次数。
            runs_loss, runs_r2, runs_elapsed_time = [], [], []
            runs_accuray, runs_precision, runs_f1, runs_recall = [], [], [], []
            # 这个地方将同一组参数，每一次的不同结果保存下来。
            for i in range(self.repeat_exp):
                start = time.time()
                model = self.define_model(model_name, ps)

                inputs = {
                        'X': self.X, 'y': self.y, 'train_mask': self.train_mask,
                        'val_mask': self.val_mask, 'test_mask': self.test_mask,
                        'cat_features': self.cat_features, 'in_features': self.in_features
                        }
                if model_name in ('gnn', 'resgnn', 'bgnn'):
                    inputs['networkx_graph'] = self.networkx_graph
                # patience:  当 early stop 被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练

                if self.task == 'regression':
                    metric_name = 'loss'
                else:
                    metric_name = self.metric

                # 这个地方要思考一个问题，就是当self.repeat_exp >1的时候 文件loss_fn会发生覆盖现象，所以对文件名进行了修改。
                metrics = model.fit(num_epochs=self.config.num_epochs, patience=self.config.patience,
                                    loss_fn=f"{self.seed_folder}/{exp_name}_{i}.json",
                                    metric_name=metric_name, feature_importance=self.feature_importance,
                                    feature_select_top_num=self.feature_select_top_num,
                                    **inputs)


                finish = time.time()
                elapsed_time = finish - start

                best_loss = min(metrics['loss'], key=lambda x: x[1])
                runs_loss.append(best_loss)
                runs_elapsed_time.append(elapsed_time)

                if self.task == 'regression':
                    best_r2 = max(metrics['r2'], key=lambda x: x[1])
                    runs_r2.append(best_r2)
                else:
                    best_accuracy = max(metrics['accuracy'], key=lambda x: x[1])
                    best_precision = max(metrics['precision'], key=lambda x: x[1])
                    best_recall = max(metrics['recall'], key=lambda x: x[1])
                    best_f1 = max(metrics['f1'], key=lambda x: x[1])

                    runs_accuray.append(best_accuracy)
                    runs_precision.append(best_precision)
                    runs_recall.append(best_recall)
                    runs_f1.append(best_f1)

            if self.task == 'regression':
                self.store_results[exp_name] = (list(map(np.mean, zip(*runs_loss))),
                                                list(map(np.mean, zip(*runs_r2))),
                                                np.mean(runs_elapsed_time),)
            else:
                metrics_lst = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'elapsed_time']
                name_lst = ['train', 'val', 'test']

                loss_dict = dict(zip(name_lst, np.mean(runs_loss, axis=0)))
                accuracy_dict = dict(zip(name_lst, np.mean(runs_accuray, axis=0)))
                precision_dict = dict(zip(name_lst, np.mean(runs_precision, axis=0)))
                recall_dict = dict(zip(name_lst, np.mean(runs_recall, axis=0)))
                f1_dict = dict(zip(name_lst, np.mean(runs_f1, axis=0)))

                self.store_results[exp_name] = dict(zip(metrics_lst,
                                                [loss_dict, accuracy_dict, precision_dict,
                                                recall_dict, f1_dict,np.mean(runs_elapsed_time)]))

                # self.store_results[exp_name] = (list(map(np.mean, zip(*runs_loss))),
                #                                 list(map(np.mean, zip(*runs_accuray))),
                #                                 list(map(np.mean, zip(*runs_precision))),
                #                                 list(map(np.mean, zip(*runs_recall))),
                #                                 list(map(np.mean, zip(*runs_f1))),
                #                                 np.mean(runs_elapsed_time),)
            # self.process_metrics(metrics, exp_name, elapsed_time)

    def define_model(self, model_name, ps):
        if model_name == 'catboost':
            return GBDTCatBoost(self.task, **ps)
        elif model_name == 'lightgbm':
            return GBDTLGBM(self.task, **ps)
        elif model_name == 'mlp':
            return MLP(self.task, **ps)
        elif model_name == 'gnn':
            return GNN(self.task, **ps)
        elif model_name == 'resgnn':
            gbdt = GBDTCatBoost(self.task)
            if self.task == 'regression':
                metric_name = 'loss'
            else:
                metric_name = self.metric

            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features=self.cat_features,
                     num_epochs=1000, patience=100,
                     plot=False, verbose=False, loss_fn=None,
                     metric_name=metric_name)
            return GNN(task=self.task, gbdt_predictions=gbdt.model.predict(self.X), **ps)
        elif model_name == 'bgnn':
            return BGNN(self.task, **ps)

    def create_save_folder(self, seed):
        self.seed_folder = f'{self.save_folder}/{seed}'
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        self.train_mask = self.masks[seed]['train']
        self.val_mask = self.masks[seed]['val']
        self.test_mask = self.masks[seed]['test']

    def save_results(self, seed):
        self.seed_results[seed] = self.store_results
        with open(f'{self.save_folder}/seed_results.json', 'w+') as f:
            # 这个地方要用可阅读的格式来写
            json.dump(self.seed_results, f, indent = 4)

        self.aggregated = self.aggregate_results()
        # 这个self.aggregrate_results() 要增加precision, recall, f1的值。
        with open(f'{self.save_folder}/aggregated_results.json', 'w+') as f:
            json.dump(self.aggregated, f, indent = 4)

    def get_model_name(self, exp_name: str, algos: list):
        # get name of the model (for gnn-like models (eg. gat))
        if 'name' in exp_name:
            model_name = '-' + [param[4:] for param in exp_name.split('-') if param.startswith('name')][0]
        else:
            model_name = ''

        # get a model used a MLP (eg. MLP-GNN)
        if 'gnn' in exp_name and 'mlpTrue' in exp_name:
            model_name += '-MLP'

        # algo corresponds to type of the model (eg. gnn, resgnn, bgnn)
        for algo in algos:
            if exp_name.startswith(algo):
                return algo + model_name
        return 'unknown'

    def aggregate_results(self):
        algos = ['catboost', 'lightgbm', 'mlp', 'gnn', 'resgnn', 'bgnn']
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        for seed in results:
            model_results_for_seed = ddict(list)
            for name, output in results[seed].items():
                model_name = self.get_model_name(name, algos=algos)
                if self.task == 'regression':  # rmse metric
                    val_metric, test_metric, elapsed_time = output[0][1], output[0][2], output[2]
                    model_results_for_seed[model_name].append((val_metric, test_metric, elapsed_time))
                else:

                    val_accuracy = output['accuracy']['val']
                    val_precision = output['precision']['val']
                    val_recall = output['recall']['val']
                    val_f1 = output['f1']['val']

                    test_accuracy = output['accuracy']['test']
                    test_precision = output['precision']['test']
                    test_recall = output['recall']['test']
                    test_f1 = output['f1']['test']

                    elapsed_time = output['elapsed_time']


                    model_results_for_seed[model_name].append((val_accuracy, val_precision,
                                                               val_recall, val_f1,
                                                               test_accuracy, test_precision,
                                                               test_recall, test_f1,
                                                               elapsed_time))

            # 最好的结果当然要重新评估，要敢于面对棘手的问题，先吃掉最丑的那只青蛙。
            for model_name, model_results in model_results_for_seed.items():
                if self.task == 'regression':
                    best_result = min(model_results)  # rmse
                else:
                    best_result = max(model_results)  # accuracy
                    # 对于分类问题，可不能只看accuracy了，要看precision, recall, f1 等。

                print("best_result = ", best_result)
                model_best_score[model_name].append(best_result[:-1])

                model_best_time[model_name].append(best_result[-1])

        aggregated = dict()
        # print("model_best_time = ", model_best_time)
        for model, scores in model_best_score.items():
            print("np.shape(scores) = {}".format(np.shape(scores)))
            print("scores = ", scores)
            if self.task == 'regression':
                aggregated[model] = (np.mean(scores), np.std(scores),
                                     np.mean(model_best_time[model]),
                                     np.std(model_best_time[model]))

            # aggregated 必须要有4个metrics才是正确的。不能只有一个，因为要代表accuracy, precision, recall, f1_score.
            # print("np.shape(aggregated[{}]) = {}".format(model, np.shape(aggregated[model])))
            else:
                # 这个地方要以字典的形式呈现出来，这样看太费劲了。不太方便。
                # 以字典的方式呈现出来:
                metrics_name = ['accuracy', 'precision', 'recall', 'f1']
                mean_metrics = np.mean(scores, axis=0)
                std_metrics = np.std(scores, axis=0)
                val_mean_metrics = mean_metrics[:4]
                test_mean_metrics = mean_metrics[4:]
                val_std_metrics = std_metrics[:4]
                test_std_metrics = std_metrics[4:]
                val_mean_dict = dict(zip(metrics_name, val_mean_metrics))
                val_std_dict = dict(zip(metrics_name, val_std_metrics))
                test_mean_dict = dict(zip(metrics_name, test_mean_metrics))
                test_std_dict = dict(zip(metrics_name, test_std_metrics))

                mean_time = np.mean(model_best_time[model])
                std_time = np.std(model_best_time[model])
                # 时间其实是需要区分是验证集还是测试集，不过这里先不管这些细节。仔细想想，这种处理方式是不是自己理想中的处理方式。
                # 就是将结果的处理逻辑想明白了。从seed_result开始思考。也就是seed_results.json这个文件的生成逻辑开始思考。现在就去看，不要停下来。
                merge_dict = dict()
                merge_dict['mean'] = {'val': val_mean_dict,
                                      'test': test_mean_dict,
                                      'time': mean_time}
                merge_dict['std'] = {'val': val_std_dict,
                                     'test': test_std_dict,
                                     'time': std_time}

                aggregated[model] = merge_dict
        return aggregated

    def manual_save_results(self, path):
        self.manual_process_metrics(path)
        # self.seed_results[seed] = self.store_results
        self.save_folder = Path.cwd().parent / path
        with open(f'{self.save_folder}/seed_results.json', 'w+') as f:
            # 这个地方要用可阅读的格式来写
            json.dump(self.seed_results, f, indent = 4)

        self.aggregated = self.aggregate_results()
        # 这个self.aggregrate_results() 要增加precision, recall, f1的值。
        with open(f'{self.save_folder}/aggregated_results.json', 'w+') as f:
            json.dump(self.aggregated, f, indent = 4)

    def manual_process_metrics(self, path):
        # 统计给定路径下的metrics 的统计结果，手工统计的时候，统计不出来时间
        # 先判断路径是否存在，是绝对路径还是相对路径
        # 默认在bgnn目录下
        self.store_results = dict()
        folder_path = Path.cwd().parent / path
        folder_lst = os.listdir(folder_path)
        self.seed_results = dict()
        for seed in folder_lst:
            if len(seed) > 1:
                break
            runs_loss, runs_r2, runs_elapsed_time = [], [], []
            runs_accuray, runs_precision, runs_f1, runs_recall = [], [], [], []
            elapsed_time = 0
            # 这个地方还要继续:
            file_lst = os.listdir(folder_path / seed)
            for file in file_lst:
                exp_name = file[:-7]
                file_path = folder_path / seed / file
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    best_loss = min(metrics['loss'], key=lambda x: x[1])
                    runs_loss.append(best_loss)
                    runs_elapsed_time.append(elapsed_time)

                    if self.task == 'regression':
                        best_r2 = max(metrics['r2'], key=lambda x: x[1])
                        runs_r2.append(best_r2)
                    else:
                        best_accuracy = max(metrics['accuracy'], key=lambda x: x[1])
                        best_precision = max(metrics['precision'], key=lambda x: x[1])
                        best_recall = max(metrics['recall'], key=lambda x: x[1])
                        best_f1 = max(metrics['f1'], key=lambda x: x[1])

                        runs_accuray.append(best_accuracy)
                        runs_precision.append(best_precision)
                        runs_recall.append(best_recall)
                        runs_f1.append(best_f1)

            # 想下，list(map(np.mean, zip(*runs_loss)))
            # 和np.mean(runs_loss, axis = 0) 有什么区别呢？
            if self.task == 'regression':

                self.store_results[exp_name] = (list(map(np.mean, zip(*runs_loss))),
                                                list(map(np.mean, zip(*runs_r2))),
                                                np.mean(runs_elapsed_time),)
            else:
                metrics_lst = ['loss', 'accuracy', 'precision', 'recall','f1', 'elapsed_time']
                name_lst = ['train', 'val', 'test']

                loss_dict = dict(zip(name_lst, np.mean(runs_loss, axis=0)))
                accuracy_dict = dict(zip(name_lst, np.mean(runs_accuray, axis = 0)))
                precision_dict = dict(zip(name_lst, np.mean(runs_precision, axis = 0)))
                recall_dict = dict(zip(name_lst, np.mean(runs_recall, axis = 0)))
                f1_dict = dict(zip(name_lst, np.mean(runs_f1, axis = 0)))

                self.store_results[exp_name] = dict(zip(metrics_lst,
                                                    [loss_dict, accuracy_dict, precision_dict,
                                                     recall_dict, f1_dict,
                                                     np.mean(runs_elapsed_time)]))
            self.seed_results[seed] = self.store_results

    def run(self, dataset: str, *args,
            save_folder: str = None,
            version_num: float = 1.0,
            task: str = 'regression',
            # class_unbalanced: bool = False,
            feature_importance: bool = False,
            feature_select_top_num: int = 0,
            metric: str = None,  # 这个地方可以考虑增加一个参数，metric， 用来指定某种评测指标。便于比较。
            repeat_exp: int = 1,
            max_seeds: int = 5,
            dataset_dir: str = None,
            config_dir: str = None
            ):
        # python scripts/run.py county  bgnn --task classification --save_folder ./county_resgnn_bgnn
        start2run = time.time()
        # save_folder 要体现日期和版本号
        today = datetime.date.today()
        today = today.strftime('%Y%m%d')
        # 获取版本号，由外部传入是不是比较好呢？

        self.dataset = dataset
        if metric is None:
            if task == 'classification':
                metric = 'accuracy'
            else:
                metric = 'RMSE'
        self.metric = metric

        self.save_folder = f"{save_folder}_{metric}_{today}_V{version_num}"
        self.task = task
        # self.class_unbalanced = class_unbalanced
        self.feature_importance = feature_importance
        self.feature_select_top_num = feature_select_top_num

        self.repeat_exp = repeat_exp
        self.max_seeds = max_seeds
        print("dataset = ", dataset)
        print("args = ", args)
        print("task = ", task)
        # print("class_unbalanced = ", class_unbalanced)
        print("feature_importance = ", feature_importance)
        print("feature_select_top_num = ", feature_select_top_num)
        print("metric = ", metric)
        print("repeat_exp = ", repeat_exp)
        print("max_seeds = ", max_seeds)
        print("config_dir = ", config_dir)
        # print(dataset, args, task, repeat_exp, max_seeds, dataset_dir, config_dir)

        dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent.parent / 'datasets'
        # print("dataset_dir=", dataset_dir)
        # print("type(dataset_dir)=", type(dataset_dir))
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / 'configs' / 'model'
        print(dataset_dir, config_dir)


        self.get_input(dataset_dir, dataset)

        self.seed_results = dict()
        # max_seeds 就是K-fold 交叉验证，masks 就是根据max_seeds 这个参数来生成的，就有max_seeds个数据集。
        for ix, seed in enumerate(self.masks):
            print(f'{dataset} Seed {seed}')
            self.seed = seed

            self.create_save_folder(seed)
            # 这一步split_masks 非常关键，分离训练集，验证集合测试集。
            # 这个地方不要这样形成mask 文件，这样的话，每次数据集的切分就不是随机的了。
            # 这种情况下，在验证同一个数据集不同算法的时候，可以用，但是同一个算法要想尽可能稳定，最好不要用这种固定的切分方式。
            #  因为随机的话，我们容易发现一些意外的惊喜。
            # 目前先这样，后续调整。
            self.split_masks(seed)

            self.store_results = dict()
            for arg in args:
                if arg == 'all':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                    self.run_one_model(config_fn=config_dir / 'mlp.yaml', model_name="mlp")
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                    break
                elif arg == 'catboost':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                elif arg == 'lightgbm':
                    self.run_one_model(config_fn=config_dir / 'lightgbm.yaml', model_name="lightgbm")
                elif arg == 'mlp':
                    self.run_one_model(config_fn=config_dir / 'mlp.yaml', model_name="mlp")
                elif arg == 'gnn':
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                elif arg == 'resgnn':
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                elif arg == 'bgnn':
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")

            self.save_results(seed)
            if ix + 1 >= max_seeds:
                break

        print(f'Finished {dataset}: {time.time() - start2run} sec.')


if __name__ == '__main__':
    fire.Fire(RunModel().run)
    # manual process
    # model = RunModel()
    # model.task = 'classification'
    # path = "elliptic_bgnn_20210409_V1.0"
    # model.manual_save_results(path)


