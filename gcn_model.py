#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:20
# @Author  : Li Xiao
# @File    : gcn_model.py
from torch import nn
import torch.nn.functional as F
from layer import GraphConvolution

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        #self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)

        x = self.fc(x)

        return x