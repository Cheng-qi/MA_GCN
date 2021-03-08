#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.7
@author: Qi Cheng
@contact: chengqi@hrbeu.edu.cn
@site: https://github.com/Cheng-qi
@software: PyCharm
@file: main.py
@time: 2020/3/25 9:29
"""
from copy import deepcopy
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from GINModels import *
from sklearn.model_selection import train_test_split

def randSplit(data, train_pro, val_pro, random_seed = 0):
    nodes_nums = data.x.shape[0]
    train_val_nodes_nums = data.x.shape[0]-1000
    val_train_labels = data.y.numpy()[:train_val_nodes_nums]

    train_num = int(nodes_nums * train_pro)
    val_num = int(nodes_nums * val_pro)
    val_train_nodes = np.array(list(range(train_val_nodes_nums)))
    train_nodes, val_nodes, train_labels, val_labels = \
        train_test_split(val_train_nodes, val_train_labels, train_size=train_num, test_size=val_num,
                         stratify=val_train_labels, random_state=random_seed)
    data.train_mask[:] = False
    data.train_mask[train_nodes] = True
    data.val_mask[:] = False
    data.val_mask[val_nodes] = True

    return train_nodes, val_nodes

params = {
    "lr": 0.01,      # learning rate
    "num_hidden_unit":30,
    "num_layer":2,
    "max_epoch":500,
    "is_rand_split":False,
    "train_pro":0.1,
    "data_name": "cora",  # cora or citeseer
    "early_stop":20
}

# data_name = 'citeseer'
# data_name = 'cora' #0.01



dataset = Planetoid(root='../../data/'+params["data_name"], name=params["data_name"])
data = dataset[0]

if(params["is_rand_split"]):
    randSplit(data, params["train_pro"], 0.2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
acc_np = np.ones((100))
max_accs = np.ones((100))
max_val_acc_model = []

data.x = F.normalize(data.x, p=1)


model = GIN(dataset.num_node_features,
                       dataset.num_classes,
                       params["num_hidden_unit"],
                       params["num_layer"]).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
# optimizer = torch.optim.Adam([{"params":model.gcns[0].parameters(), "weight_decay":5e-4},
#                               ], lr=0.01)
model.train()
max_val_acc = 0
min_val_loss = 100
max_val_acc_model = 0
cur_step = 0
for epoch in range(params["max_epoch"]):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        if (float(val_loss.cpu().detach().numpy()) < min_val_loss):
            min_val_loss = float(val_loss.cpu().detach().numpy())
            max_val_acc_model = deepcopy(model.state_dict())
            cur_step = 0
        else:
            cur_step += 1
            if (cur_step > params["early_stop"]):
                # print("early stop")
                break
model.load_state_dict(max_val_acc_model)
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
correct_train = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
acc_test = correct / data.test_mask.sum().item()
# acc_train = correct_train / data.train_mask.sum().item()
print("Test set results:",
      "accuracy=", "{:.5f}".format(acc_test))

