#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 14:01
# @Author  : Li Xiao
# @File    : SNF.py
import snf
import pandas as pd
import numpy as np
import argparse
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean',
                        help='Distance metric to compute. Must be one of available metrics in :py:func scipy.spatial.distance.pdist.')
    parser.add_argument('--K', '-k', type=int, default=20,
                        help='(0, N) int, number of neighbors to consider when creating affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 20.')
    parser.add_argument('--mu', '-mu', type=int, default=0.5,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 0.5.')
    args = parser.parse_args()

    print('Load data files...')
    omics_data_1 = pd.read_csv(args.path[0], header=0, index_col=None)
    omics_data_2 = pd.read_csv(args.path[1], header=0, index_col=None)
    omics_data_3 = pd.read_csv(args.path[2], header=0, index_col=None)
    print(omics_data_1.shape, omics_data_2.shape, omics_data_3.shape)

    if omics_data_1.shape[0] != omics_data_2.shape[0] or omics_data_1.shape[0] != omics_data_3.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    omics_data_1.rename(columns={omics_data_1.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data_2.rename(columns={omics_data_2.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data_3.rename(columns={omics_data_3.columns.tolist()[0]: 'Sample'}, inplace=True)

    # align samples of different data
    omics_data_1.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data_2.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data_3.sort_values(by='Sample', ascending=True, inplace=True)

    print('Start similarity network fusion...')
    affinity_nets = snf.make_affinity([omics_data_1.iloc[:, 1:].values.astype(np.float), omics_data_2.iloc[:, 1:].values.astype(np.float), omics_data_3.iloc[:, 1:].values.astype(np.float)],
                                      metric=args.metric, K=args.K, mu=args.mu)

    fused_net =snf.snf(affinity_nets, K=args.K)

    print('Save fused adjacency matrix...')
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = omics_data_1['Sample'].tolist()
    fused_df.index = omics_data_1['Sample'].tolist()
    fused_df.to_csv('result/SNF_fused_matrix.csv', header=True, index=True)

    np.fill_diagonal(fused_df.values, 0)
    fig = sns.clustermap(fused_df.iloc[:, :], cmap='vlag', figsize=(8,8),)
    fig.savefig('result/SNF_fused_clustermap.png', dpi=300)
    print('Success! Results can be seen in result file')