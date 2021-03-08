#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.7
@author: Qi Cheng
@contact: chengqi@hrbeu.edu.cn
@site: https://github.com/Cheng-qi
@software: PyCharm
@file: GNNModels.py
@time: 2020/11/25 17:31
"""

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from mlp import MLP

class GIN(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dims, depth):
        super().__init__()
        self.depth = depth
        self.gcns = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.gcns.append(GINConv(MLP(2,input_channel, hidden_dims, hidden_dims), train_eps = True))
        for i in range(depth-2):
            self.gcns.append(GINConv(MLP(2,hidden_dims, hidden_dims, hidden_dims), train_eps = True))
        self.gcns.append(GINConv(MLP(2, hidden_dims,hidden_dims, output_channel), train_eps = True))
        for i in range(depth-1):
            self.batch_norm.append(nn.BatchNorm1d(hidden_dims))

    def forward(self, x, edges):
        for i in range(self.depth-1):
            x = F.dropout(x, p=0.5,training=self.training)
            x = F.relu(self.gcns[i](x, edges))
            # x = self.batch_norm[i](x)
        x = F.dropout(x, training=self.training)
        o = self.gcns[-1](x, edges)
        # o = x
        return F.log_softmax(o,dim=1)


