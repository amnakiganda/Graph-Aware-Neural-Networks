# -*- coding: utf-8 -*-
"""
FCNN, GDNN and GNN layer definitions

"""

import os
import numpy as np
import pandas as pd
from numpy import random
import math
import cmath

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import networkx as nx
import scipy.sparse as sp

import torch
from torch import nn


import os
from tqdm import tqdm
import os.path as osp

import torch.nn.functional as F

import matplotlib.pyplot as plt

np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scipy.io

import params


class Dense(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
       
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        
        return output 
    
def create_mask(branch, hidden_size, cond_units):
    
    batch_size = (branch.size())[0]
    edge_size =  (branch.size())[1]
    
    _branch_mask = (branch.repeat(1, cond_units)).to(params.device)
    branch_mask = ((torch.ones(batch_size, (_branch_mask.size())[1])).to(params.device) - _branch_mask).to(params.device)
    
    s_mask =  (torch.ones(batch_size, (hidden_size - (cond_units*edge_size)))).to(params.device)
    #print(s_mask.size(), branch_mask.size())
    
    mask = torch.cat((s_mask, branch_mask),1)
    
    return mask

class DropConnect(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(DropConnect, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
       
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, branch, cu_size):
        
        output = torch.matmul(input, self.weight)
        drop_connect = create_mask(branch, self.out_features, cu_size)
        output = output * drop_connect
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
class GD(nn.Module):
    def __init__(self, n_layersGD, n_layersFC, layersGD, layersFC, n_b, dropout, sigma, readout):
        super(GD, self).__init__()
               
        
        self.dropout = dropout
        self.readout = readout        
        self.sigma = sigma
        self.n_b = n_b
        
        # assert len(layersGNN) == n_layersGNN + 1
        # assert len(layersFC) == n_layersFC
        
        self.layersGD = layersGD
        self.layersFC = layersFC
        
        #for l in range(3):
        self.lin_enc = Dense(self.n_b *params.layersEn_Dc[0], self.n_b *params.layersEn_Dc[1], bias=True)
        self.lin_dec = Dense(self.n_b * layersGD[-1], self.n_b *params.layersEn_Dc[-1], bias=True)
        
        self.guided = torch.nn.ModuleList()
        self.guided.append(DropConnect(self.n_b *params.layersEn_Dc[1], self.n_b *layersGD[0], bias=True))
        for l in range(n_layersGD - 2):
            self.guided.append(
                DropConnect(self.n_b *layersGD[l], self.n_b *layersGD[l+1], bias=True))
        self.guided.append(DropConnect(self.n_b * layersGD[-2], self.n_b *layersGD[-1], bias=True))  
        
        ## Readout layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(Dense(self.n_b * layersGD[-1], self.n_b *layersFC[0], bias=True))
        for l in range(n_layersFC - 2):
            self.lins.append(Dense(self.n_b *layersFC[l], self.n_b *layersFC[l+1], bias=True))
        self.lins.append(Dense(self.n_b *layersFC[-2], self.n_b * self.layersFC[-1], bias=True))
        
        
    def reset_parameters(self):
        self.lin_enc.reset_parameters()
        self.lin_dec.reset_parameters()
        
        for gd in self.guided:
            gd.reset_parameters()
            
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, edge, cond_units):
        x = x.reshape((-1, params.n_b * params.num_infeat))
        
        x = self.lin_enc(x)
        
        for gd in self.guided[:-1]:
            x = gd(x, edge, cond_units)
            x = self.sigma(x)
        x = self.guided[-1](x, edge, cond_units)
        
        if self.readout:
            ##....................................
            x = x.reshape((-1, self.n_b * self.layersGD[-1]))
            x  = torch.squeeze(x) 
            for lin in self.lins[:-1]:
                x = lin(x)
                x = self.sigma(x)
                x = F.dropout(x, p = self.dropout)
            x = self.lins[-1](x)
            # x = x.reshape((-1, self.layersFC[-1], self.n_b ))
            # x  = torch.squeeze(x)
        # else:
        #     x = x.reshape((-1, self.layersGD[-1], self.n_b ))
        #     x  = torch.squeeze(x)
            
        x = self.lin_dec(x)
        x = x.reshape((-1, params.layersEn_Dc[-1], self.n_b ))
        x  = torch.squeeze(x)
        
        return x
    
    
def FilterFunction(h, S, x, bias):
    
    # Number of output features
    F = h.shape[0]    
    # Number of filter taps/hops
    K = h.shape[1]    
    # Number of input features
    G = h.shape[2]    
    # Number of nodes
    N = S.shape[1]    
    # Batch size
    B = x.shape[0]

    # Create concatenation dimension and initialize concatenation tensor z
    x = x.reshape([B, 1, G, N])
    S = S.reshape([B, 1, N, N])
    z = x
    
    # Loop over the number of filter taps
    for k in range(1, K):        
        # S*x
        x = torch.matmul(x, S)        
        
        xS = x.reshape([B, 1, G, N])        
        
        z = torch.cat((z, xS), dim=1)
    
    # Multiply by h - parameter matrix
    y = torch.matmul(z.permute(0, 3, 1, 2).reshape([B, N, K*G]), 
                     h.reshape([F, K*G]).permute(1, 0)).permute(0, 2, 1)
    return y + bias #y = B, F, N


#Node Variant Graph filter    
def NVGF(h, S, x, bias):
    """
    Inherited from: 
    https://github.com/alelab-upenn/graph-neural-networks
    
    """
    
    # h is output_features x edge_weights x filter_taps x input_features x number_nodes
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    N = h.shape[4]
    
    B = x.shape[0]
   
    # h in F x E x K x G x N
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    xr = x.reshape([B, 1, G, N])
    Sr = S.reshape([B, 1, N, N])
    z = xr.reshape([B, 1, 1, G, N]).repeat(1, 1, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        xr = torch.matmul(xr, Sr) # B x E x G x N
        xS = xr.reshape([B, 1, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the filter taps.
    # This multiplication with filter taps is ``element wise'' on N since for each node we have a different element
    # First, add the extra dimension (F for z, and B for h)
    z = z.reshape([B, 1, 1, K, G, N])
    h = h.reshape([1, F, 1, K, G, N])
    # Now let's do elementwise multiplication
    zh = z * h
    # And sum over the dimensions E, K, G to get B x F x N
    y = torch.sum(zh, dim = 4) # Sum over G
    y = torch.sum(y, dim = 3) # Sum over K
    y = torch.sum(y, dim = 2) # Sum over E
    # Finally, add the bias
    
    return y + bias

class NodeVariantGF(nn.Module):
    
    """
    Inherited from: 
    https://github.com/alelab-upenn/graph-neural-networks
    
    """
    def __init__(self, G, F, K, M):
        # G: Number of input features
        # F: Number of output features
        # K: Number of filter shift taps
        # M: Number of filter node taps
        # GSOs will be added later.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.M = M
        #self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, 1, K, G, M))
        self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.G * self.K * self.M)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

                    

    def forward(self, x, S):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        self.S = S
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        
        self.h = self.weight
        
        u = NVGF(self.h, self.S, x, self.bias)
        
        return u

#Edge Variant Graph filter 
def EVGF(S, x, bias):  
    """
    Inherited from: 
    https://github.com/alelab-upenn/graph-neural-networks
    
    """
    
    # S is output_features x edge_features x filter_taps x input_features
    # x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = S.shape[0+1]
    E = S.shape[1+1]
    K = S.shape[2+1]
    G = S.shape[3+1]
    N = S.shape[4+1]
    
    B = x.shape[0]
    
    # S in F x E x K x G x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    #
    x = x.reshape([B, 1, 1, G, N, 1])
    # When we do index_select along dimension K we get rid of this dimension
    Sk = torch.index_select(S, 3, torch.tensor(0).to(S.device)).squeeze(2)
    # Sk in F x E x G x N x N
    
    # Matrix multiplication
    x = torch.matmul(Sk, x) # B x F x E x G x N x 1
    # And we collect this for every k in a vector z, along the K dimension
    z = x.reshape([B, F, E, 1, G, N, 1]).squeeze(6) # B x F x E x 1 x G x N
    # Now we do all the matrix multiplication
    for k in range(1,K):
        # Extract the following k
        Sk = torch.index_select(S, 3, torch.tensor(k).to(S.device)).squeeze(2)
        
        # Multiply with the previously cumulative Sk * x
        x = torch.matmul(Sk, x) # B x F x E x G x N x N
        # Get rid of the last dimension (of a column vector)
        Sx = x.reshape([B, F, E, 1, G, N, 1]).squeeze(6) # B x F x E x 1 x G x N
        # Add to the z
        z = torch.cat((z, Sx), dim = 2) # B x F x E x k x G x N
    # Sum over G
    z = torch.sum(z, dim = 4)
    # Sum over K
    z = torch.sum(z, dim = 3)
    # Sum over E
    y = torch.sum(z, dim = 2)
    
    return y + bias


class EdgeVariantGF(nn.Module):
    
    """
    Inherited from: 
    https://github.com/alelab-upenn/graph-neural-networks
    
    """
    
    def __init__(self, G, F, K, N):
        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = 1
        self.M = N # Number of selected nodes
        self.N = N # Total number of nodes
        self.S = None
        # Create parameters for the Edge-Variant part:
        self.weightEV = nn.parameter.Parameter(torch.Tensor(F, 1, K, G, N, N))
        
        # If we want a hybrid, create parameters
        self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.G * self.K * self.N)
        self.weightEV.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        B = S.shape[0]
        N = S.shape[1]
        S = S.reshape([B, 1, N, N])
        
        self.S = S
        multipleIdentity = torch.eye(self.N).reshape([1, self.N, self.N]).repeat(B, self.E, 1, 1).to(S.device)
        # Compute the nonzero elements of S+I_{N}
        sparsityPattern = ((torch.abs(S) + multipleIdentity) > 1e-9)
        
        # Change from byte tensors to float tensors (or the same type of data as
        # the GSO)
        sparsityPattern = sparsityPattern.type(S.dtype)
        
        hybridMask = torch.ones([self.N, self.N])
        # Now that we have the hybrid mask, we need to mask the sparsityPattern
        # we got so far
        hybridMask = hybridMask.reshape([1, self.N, self.N]).to(S.device)
        #   1 x N x N
        sparsityPattern = sparsityPattern * hybridMask
        
        self.sparsityPattern = sparsityPattern.to(S.device)
        #   B x E x N x N
        
        # The sparsity pattern has shape E x N x N. And we want to make it
        # 1 x E x K x 1 x N x N. The K dimension is to guarantee that for k=0
        # we have the identity
        multipleIdentity = (multipleIdentity * hybridMask).reshape([B,  1, self.E, 1, 1, self.N, self.N])
        if self.K > 1:
            # This gives a 1 x E x 1 x 1 x N x N identity matrix
            sparsityPattern = sparsityPattern.reshape([B, 1, self.E, 1, 1, self.N, self.N])
            # This gives a 1 x E x 1 x 1 x N x N sparsity pattern matrix
            sparsityPattern = sparsityPattern.repeat(1,1, 1, self.K-1, 1, 1, 1)
            # This repeats the sparsity pattern K-1 times giving a matrix of shape
            #   1 x E x (K-1) x 1 x N x N
            sparsityPattern = torch.cat((multipleIdentity,sparsityPattern), dim = 3)
        else:
            sparsityPattern = multipleIdentity
        # This sholud give me a 1 x E x K x 1 x N x N matrix with the identity
        # in the first element
        self.sparsityPatternFull = sparsityPattern.type(S.dtype).to(S.device)

    def forward(self, x, S):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        #B = x.shape[0]
        #F = x.shape[1]
        #Nin = x.shape[2]
        
        self.addGSO(S)
        
        # Mask the parameters
        self.Phi = self.weightEV * self.sparsityPatternFull
               
        # Compute the filter output for the EV part
        uEV = EVGF(self.Phi, x, self.bias)
        
        u = uEV 
        
        return u

class GNN(nn.Module):
    def __init__(self, n_layersGNN, n_layersFC, k, layersGNN, layersFC, n_b, dropout, sigma, readout):
        super().__init__()
               
        self.n_b = n_b
        self.dropout = dropout
        self.readout = readout        
        self.sigma = sigma
        
        #assert len(layersGNN) == n_layersGNN + 1
        assert len(layersFC) == n_layersFC
        
        self.layersGNN = layersGNN
        self.layersFC = layersFC
        
        self.lin_enc = Dense(self.n_b *params.layersEn_Dc[0], self.n_b *params.layersEn_Dc[1], bias=True)
        self.lin_dec = Dense(self.n_b * layersGNN[-1], self.n_b *params.layersEn_Dc[-1], bias=True)
        
        
        self.convs = torch.nn.ModuleList()        
                
        #With encoder Node variant GF
        self.convs.append(NodeVariantGF(params.layersEn_Dc[1], layersGNN[0], k[0], self.n_b))
        for l in range(n_layersGNN - 2):
            self.convs.append(
                NodeVariantGF(layersGNN[l], layersGNN[l+1], k[l+1], self.n_b))
        self.convs.append(NodeVariantGF(layersGNN[-2], layersGNN[-1], k[-1], self.n_b))
        
        #With encoder Edge variant GF 
        # self.convs.append(EdgeVariantGF(params.layersEn_Dc[1], layersGNN[0], k[0], self.n_b))
        # for l in range(n_layersGNN - 2):
        #     self.convs.append(
        #         EdgeVariantGF(layersGNN[l], layersGNN[l+1], k[l+1], self.n_b))
        # self.convs.append(NodeVariantGF(layersGNN[-2], layersGNN[-1], k[-1], self.n_b))
                
        ## Readout layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(Dense(self.n_b * layersGNN[-1], self.n_b *layersFC[0], bias=True))
        for l in range(n_layersFC - 2):
            self.lins.append(Dense(self.n_b *layersFC[l], self.n_b *layersFC[l+1], bias=True))
        self.lins.append(Dense(self.n_b *layersFC[-2], self.n_b * self.layersFC[-1], bias=True))
             
        
    def reset_parameters(self):
        self.lin_enc.reset_parameters()
        self.lin_dec.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
            
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, gso):
        
        x = x.reshape((-1, params.n_b * params.num_infeat))
        
        x = self.lin_enc(x)
        x = self.sigma(x)
        x = x.reshape((-1, params.layersEn_Dc[1], self.n_b ))
               
        
        for conv in self.convs[:-1]:
            x = conv(x, gso)
            x = self.sigma(x)
            #x = F.dropout(x)
        x = self.convs[-1](x, gso)
                
        #Decoder
        x = x.reshape((-1, self.n_b * self.layersGNN[-1]))
        x = self.lin_dec(x)
        x = x.reshape((-1, params.layersEn_Dc[-1], self.n_b ))
        x  = torch.squeeze(x)
         
        return x 