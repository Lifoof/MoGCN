#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 14:01
# @Author  : Li Xiao
# @File    : autoencoder_model.py
import torch
from torch import nn
from matplotlib import pyplot as plt

class MMAE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, a=0.4, b=0.3, c=0.3):
        '''
        :param in_feas_dim: a list, input dims of omics data
        :param latent_dim: dim of latent layer
        :param a: weight of omics data type 1
        :param b: weight of omics data type 2
        :param c: weight of omics data type 3
        '''
        super(MMAE, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        #encoders, multi channel input
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.in_feas[0], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.in_feas[1], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        self.encoder_omics_3 = nn.Sequential(
            nn.Linear(self.in_feas[2], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        #decoders
        self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))

        #Variable initialization
        for name, param in MMAE.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2, omics_3):
        '''
        :param omics_1: omics data 1
        :param omics_2: omics data 2
        :param omics_3: omics data 3
        '''
        encoded_omics_1 = self.encoder_omics_1(omics_1)
        encoded_omics_2 = self.encoder_omics_2(omics_2)
        encoded_omics_3 = self.encoder_omics_3(omics_3)
        latent_data = torch.mul(encoded_omics_1, self.a) + torch.mul(encoded_omics_2, self.b) + torch.mul(encoded_omics_3, self.c)
        decoded_omics_1 = self.decoder_omics_1(latent_data)
        decoded_omics_2 = self.decoder_omics_2(latent_data)
        decoded_omics_3 = self.decoder_omics_3(latent_data)
        return latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0       #Record the loss of each epoch
            for (x,y) in train_loader:
                omics_1 = x[:, :self.in_feas[0]]
                omics_2 = x[:, self.in_feas[0]:self.in_feas[0]+self.in_feas[1]]
                omics_3 = x[:, self.in_feas[0]+self.in_feas[1]:self.in_feas[0]+self.in_feas[1]+self.in_feas[2]]

                omics_1 = omics_1.to(device)
                omics_2 = omics_2.to(device)
                omics_3 = omics_3.to(device)

                latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(omics_1, omics_2, omics_3)
                loss = self.a*loss_fn(decoded_omics_1, omics_1)+ self.b*loss_fn(decoded_omics_2, omics_2) + self.c*loss_fn(decoded_omics_3, omics_3)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))

            #save the model every 10 epochs, used for feature extraction
            if (epoch+1) % 10 ==0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch+1))

        #draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('result/AE_train_loss.png')