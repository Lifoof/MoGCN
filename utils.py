#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:21
# @Author  : Li Xiao
# @File    : utils.py
import pandas as pd
import numpy as np

def load_data(adj, fea, lab, threshold=0.005):
    '''
    :param adj: the similarity matrix filename
    :param fea: the omics vector features filename
    :param lab: sample labels  filename
    :param threshold: the edge filter threshold
    '''
    print('loading data...')
    adj_df = pd.read_csv(adj, header=0, index_col=None)
    fea_df = pd.read_csv(fea, header=0, index_col=None)
    label_df = pd.read_csv(lab, header=0, index_col=None)

    if adj_df.shape[0] != fea_df.shape[0] or adj_df.shape[0] != label_df.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    fea_df.rename(columns={fea_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample'}, inplace=True)

    #align samples of different data
    adj_df.sort_values(by='Sample', ascending=True, inplace=True)
    fea_df.sort_values(by='Sample', ascending=True, inplace=True)
    label_df.sort_values(by='Sample', ascending=True, inplace=True)

    print('Calculating the laplace adjacency matrix...')
    adj_m = adj_df.iloc[:, 1:].values
    #The SNF matrix is a completed connected graph, it is better to filter edges with a threshold
    adj_m[adj_m<threshold] = 0

    # adjacency matrix after filtering
    exist = (adj_m != 0) * 1.0
    #np.savetxt('result/adjacency_matrix.csv', exist, delimiter=',', fmt='%d')

    #calculate the degree matrix
    factor = np.ones(adj_m.shape[1])
    res = np.dot(exist, factor)     #degree of each node
    diag_matrix = np.diag(res)  #degree matrix
    #np.savetxt('result/diag.csv', diag_matrix, delimiter=',', fmt='%d')

    #calculate the laplace matrix
    d_inv = np.linalg.inv(diag_matrix)
    adj_hat = d_inv.dot(exist)

    return adj_hat, fea_df, label_df

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)