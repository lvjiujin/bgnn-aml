import os
import sys
import json
import pathlib
import random
import time
import networkx as nx
import numpy as np
import pandas as pd
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
from bgnn.scripts.utils import NpEncoder

max_seeds = 5
raw_data = pathlib.Path.cwd().parent / "elliptic_bitcoin_dataset/"
processed_data = pathlib.Path.cwd().parent / 'datasets' / 'elliptic'

# check whether the exported file path exists, create it if it does not exist
os.makedirs(processed_data, exist_ok=True)


file_classes = raw_data / "elliptic_txs_classes.csv"
df_classes = pd.read_csv(file_classes, index_col='txId')
# In the raw data, '-1' denotes as  unlabeled nodes,
# '2' denotes as 'labeled licit nodes,
# '1' denotes as 'labeled illicit nodes

df_classes["class"] = df_classes["class"].replace(
    {"unknown": -1,  # unlabeled nodes
     "2": 0,  # labeled licit nodes
     # "1": 1,  # labeled illicit nodes
     }
).astype(int)

# generate target file 'y.csv'
df_classes.to_csv(processed_data / "y.csv", index=False)

# generate the feature matrix file 'X.csv'
file_features = raw_data / 'elliptic_txs_features.csv'
df_features = pd.read_csv(file_features, index_col=0, header=None)
rename_dict = dict(
    zip(
        range(1, 167),
        ["time_step", ] + [f"local_{i:02d}" for i in range(1, 94)]
        + [f"aggr_{i:02d}" for i in range(1, 73)],
    )
)
df_features.rename(columns=rename_dict, inplace=True)
df_features.head()

df_features.to_csv(processed_data / 'X.csv', index=False)


# begin to process the edge file
df_index = df_classes.reset_index().reset_index().drop(columns='class').set_index('txId')

file_edges = raw_data / "elliptic_txs_edgelist.csv"
df_edges = pd.read_csv(file_edges).join(df_index, on='txId1', how='inner')
df_edges = df_edges.join(df_index, on='txId2', how='inner', rsuffix='2').drop(columns=['txId1', 'txId2'])

# generate the graph.graphml file so as to satisfy the bgnn model

g_nx = nx.MultiDiGraph()
g_nx.add_nodes_from(
    zip(df_index["index"], [{"label": v} for v in df_classes["class"]])
)
g_nx.add_edges_from(zip(df_edges["index"], df_edges["index2"]))

nx.write_graphml_lxml(g_nx, processed_data / "graph.graphml")

# masks.json file, split the data into train/valid/test

idx = list(range(df_classes.shape[0]))

y_cls = df_classes.reset_index().reset_index().set_index('txId')
y_abs = set(y_cls[y_cls['class'] != -1]['index'].values)

masks = dict()
# train: val: test = 6:2:2
for i in range(max_seeds):
    random.shuffle(idx)
    r1, r2, r3 = idx[:int(.6*len(idx))], idx[int(.6*len(idx)):int(.8*len(idx))], idx[int(.8*len(idx)):]
    r1 = [x for x in r1 if x in y_abs]
    r2 = [x for x in r2 if x in y_abs]
    r3 = [x for x in r3 if x in y_abs]
    masks[str(i)] = {"train": r1, "val": r2, "test": r3}

with open(processed_data / 'masks.json', 'w+') as f:
    json.dump(masks, f, cls=NpEncoder)

# generate the categorical feature file, there is only one categorical feature.
with open(processed_data / 'cat_features.txt', 'w+') as f:
    f.write('time_step')
# 这里需要确认下，是否需要在训练集中将类别为-1的排除掉，看下pyg中的实现吧。
