#!/usr/bin/env python
# encoding: utf-8
# 带有注意力机制
"""
@version: 3.7
@author: Qi Cheng
@contact: chengqi@hrbeu.edu.cn
@site: https://github.com/Cheng-qi
@software: PyCharm
@file: model.py
@time: 2020/3/25 9:28
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv

# class Gate(torch.nn.Module):
#     def __init__(self):
#






class MAGCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 graph_out_channels,
                 model_num,
                 dense_out_channels = 2,
                 graph_hidden_dims = 16,
                 dense_hidden_dims = 256,
                 gnn_type = GCNConv,
                 device = torch.device('cuda'),
                 shared_weight = False,
                 bias = True,
                 dropout = 0.5,
                 task_type = "node_classify",
                 act = torch.relu,
                 is_pre_feature = False):
        super(MAGCN, self).__init__()
        self.device = device
        self.model_num = model_num
        self.share_weight = shared_weight
        self.act = act
        self.bias = bias
        self.dropout = dropout
        self.task_type = task_type
        self.is_pre_feature = is_pre_feature

        self.gs = [0]*model_num # 记录g_vector

        self.re_dim_models = nn.ModuleList().to(self.device) # 降维
        self.gnn_models = nn.ModuleList().to(self.device) # GCN模型
        self.gates = nn.ModuleList().to(self.device) # 门向量p

        self.nor = nn.BatchNorm1d(in_channels)
        self.dense_models = nn.ModuleList([nn.Linear(graph_out_channels, dense_hidden_dims),
                                           nn.Linear(dense_hidden_dims, dense_out_channels)]).to(self.device) # 图分类全连接层
        if self.is_pre_feature:
            self.pre_feature = nn.Linear(in_channels,in_channels,bias = self.bias)


        self.gnn_models.append(GCNConv(in_channels, graph_hidden_dims, bias = self.bias))
        for i in range(model_num-2):
            self.gnn_models.append(GCNConv(graph_hidden_dims, graph_hidden_dims, bias = self.bias))
        self.gnn_models.append(GCNConv(graph_hidden_dims, graph_out_channels, bias = self.bias))


        for i in range(model_num-1):
            self.re_dim_models.append(nn.Linear(in_channels, graph_hidden_dims, bias = self.bias)) ## W^(2)_0和W^(1)_0
            self.gates.append(nn.Linear(graph_hidden_dims*2,1, bias = False))


    def forward(self, xs, edge_indexes):
        if self.task_type == "node_classify":
            return self.node_classify(xs, edge_indexes)
        elif self.task_type == "graph_classify":
            return self.graph_classify(xs, edge_indexes)


    def node_emd(self, x, edge_index):
        if self.is_pre_feature:
            x = self.pre_feature(x)
        # else:
            # x = x
        h = x.clone()
        # x = self.nor(x)
        # if self.dropout:
        #     h = F.dropout(h, self.dropout, training=self.training)

        h0 = self.re_dim_models[0](x)
        # h0 = x
        h1 = self.gnn_models[0](h, edge_index)
        h1 = self.act(h1)
        # g = torch.relu(torch.sigmoid(self.gates[0](h1)/torch.norm(self.gates[0].weight, 2))-0.1)
        # g = torch.sigmoid(self.gates[0](h1)/torch.norm(self.gates[0].weight, 2))
        # g = torch.sigmoid(self.gates[0](torch.cat((h1,h0), dim = 1))/torch.norm(self.gates[0].weight, 2))
        g = torch.sigmoid(self.gates[0](torch.cat((h1,h0), dim = 1)))
        self.gs[0] = g.detach().cpu().numpy()
        h = g*h1 +  (1-g)* h0
        for i in range(self.model_num-2):
            if self.dropout:
                h = F.dropout(h, self.dropout, training=self.training)
            h0 = self.re_dim_models[i+1](x)
            # h0 = x
            h1 = self.gnn_models[i+1](h, edge_index)
            h1 = self.act(h1)
            # g = torch.relu(torch.sigmoid(self.gates[i+1](h1) / torch.norm(self.gates[i+1].weight, 2))-0.1)
            # g = torch.sigmoid(self.gates[i+1](h1) / torch.norm(self.gates[i+1].weight, 2))
            g = torch.sigmoid(self.gates[i+1](torch.cat((h1,h0),dim = 1 )) / torch.norm(self.gates[i+1].weight, 2))

            self.gs[i+1]=g.detach().cpu().numpy()
            h = g*h1 +  (1-g)*  h0
            # h =   g *  h0

        # if self.dropout:
            # h = F.dropout(h, self.dropout, training=self.training)
        out = self.gnn_models[-1](h, edge_index)

        return out


    def node_classify(self, x, edge_index):
        return F.log_softmax(self.node_emd(x, edge_index), dim = 1)



    def graph_classify(self, xs, edge_indexes):
        o_hs = []
        for edge_index, x in zip(xs, edge_indexes):
            hs = self.node_emd(x, edge_index)
            h = self.readout(hs)
            o_hs.append(h)
        hs = torch.stack(o_hs, 0)
        if(self.dropout != None):
            hs = F.dropout(hs, self.dropout, training=self.training)
        hs =self.dense_models[0](hs)
        if(self.dropout != None):
            hs = F.dropout(hs, self.dropout, training=self.training)
        hs = self.dense_models[1](hs)
        return F.log_softmax(hs, dim = 1)

    def readout(self, hs):
        # 图读出
        h_max = [torch.max(h, 0)[0] for h in hs]
        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h










