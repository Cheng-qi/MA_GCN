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
import scipy.sparse as sp
import torch
import numpy as np
import torch.nn.functional as F
# from model_integration import IntegrationGNN
from torch_geometric.datasets import Planetoid
# from GNNModels import *
from MAGCN import *
from sklearn.model_selection import train_test_split


def randSplit(data, train_pro, val_pro, random_seed = 0):
    nodes_nums = data.x.shape[0]
    train_val_nodes_nums = data.x.shape[0]-1000
    val_train_labels = data.y.numpy()[:train_val_nodes_nums]

    train_num = int(nodes_nums * train_pro)
    val_num = int(nodes_nums * val_pro)
    # test_num = nodes_nums - train_num - val_num

    val_train_nodes = np.array(list(range(train_val_nodes_nums)))
    # val_train_nodes, test_nodes, val_train_labels, test_labels = \
    #     train_test_split(nodes_arr, data.y.numpy(), train_size=train_num + val_num, test_size=test_num,
    #                      stratify=data.y.numpy(), random_state=random_seed)
    train_nodes, val_nodes, train_labels, val_labels = \
        train_test_split(val_train_nodes, val_train_labels, train_size=train_num, test_size=val_num,
                         stratify=val_train_labels, random_state=random_seed)

    # train_nodes, val_nodes, test_nodes = randSplit(data, 0.1, 0.1)
    data.train_mask[:] = False
    data.train_mask[train_nodes] = True
    data.val_mask[:] = False
    data.val_mask[val_nodes] = True
    # data.test_mask[:] = False
    # data.test_mask[test_nodes] = True

    return train_nodes, val_nodes


# data_name = 'Citeseer'
# data_name = 'Cora' #0.01
# data_name = 'PubMed' # 0.001

# randSplit(data, 0.2, 0.1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')



params = {
    "lr": 0.001,      # learning rate
    "num_hidden_unit":30,
    "num_layer":2,
    "max_epoch":500,
    "is_rand_split":False,
    "train_pro":0.2,
    "data_name": "cora",  # cora or citeseer
    "early_stop":20,
}
print(params)
data_name = params["data_name"]
dataset = Planetoid(root='../data/'+data_name, name=data_name)
data = dataset[0]
if(params["is_rand_split"]):
    randSplit(data, params["train_pro"], 0.2)


acc_np = np.ones((100))
max_accs = np.ones((100))
max_val_acc_model = []

## 特征归一化
data.x = F.normalize(data.x, p=1)

# 记录不同层数的结果


model = MAGCN(dataset.num_node_features,
                       dataset.num_classes,
                       model_num=params["num_layer"],
                       graph_hidden_dims = params["num_hidden_unit"],
                       # graph_hidden_dims = num_layer,
                       bias = False,
                       device = device,
                       act= F.elu,
                       is_pre_feature=False,
                       # act= lambda x:x,
                       dropout=0).to(device)
data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
optimizer = torch.optim.Adam([{"params":model.gnn_models.parameters(), "weight_decay":0.01},
                              {"params":model.re_dim_models.parameters(), "weight_decay":0.05},
                              {"params": model.gates.parameters(), "weight_decay":0}], lr=params["lr"])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
model.train()
max_val_acc = 0
min_val_loss = 10000
max_val_acc_model = 0
# ave = np.mean(data.x)
# std = np.std(data.x)
cur_step = 0
for epoch in range(params["max_epoch"]):
    # max_val_acc = 0

    optimizer.zero_grad()
    # out = model(data)
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    # val_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
    loss.backward()
    # val_loss.backward()
    optimizer.step()
    # scheduler.step()
    if epoch%1==0:
        if(float(val_loss.cpu().detach().numpy())<min_val_loss):
            min_val_loss = float(val_loss.cpu().detach().numpy())
            max_val_acc_model = deepcopy(model.state_dict())
            cur_step = 0
            # print(cur_step)
        else:
            cur_step +=1
            # print(cur_step)
            if(cur_step>params["early_stop"]):
                print("early stop...")
                break


model.load_state_dict(max_val_acc_model)
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
correct_train = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
acc_train = correct_train / data.train_mask.sum().item()
# print('第{:d}次迭代'.format(i+1))
print('Accuracy: {:.8f}'.format(acc))
# print("第%d次迭代准确率%.5f" % (i, acc))
# acc_np[i] = acc

    # print("层数"+str(num_layer)+"Average accuracy of running 100 times is %.5f, std is %.5f" % (acc_np.mean(), acc_np.std()))
    # with open("cora_result_diff_layer_num_ma_gcn_0_2.csv", "a") as f:
    #     f.write("\n%d,%.6f,%.6f,%.6f,%.6f"%(num_layer,acc_np.mean(), acc_np.std(),acc_np.max(),acc_np.min()))
# print(acc_np.mean())
# print(max_accs.mean())

