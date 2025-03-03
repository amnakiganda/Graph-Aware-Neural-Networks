# -*- coding: utf-8 -*-
"""
Importing data files into GD and GNN models
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

base_adj = []
base_edgelist = []

#Scaling the data
def normalize_data(x, is_scale = False):
    
    if is_scale:
        x_mean = torch.mean(x,0)
        x_std = torch.std(x,0)
        x_norm = (x -torch.mean(x,0))/(torch.std(x,0) + 1e-8)
        x_norm[torch.isnan(x_norm)] = 0
        x_norm[torch.isinf(x_norm)] = 0
    if not is_scale:
        x_norm = x
        x_mean = torch.tensor(0.0, dtype=torch.float) 
        x_std = torch.tensor(0.0, dtype=torch.float) 
    return x_norm, x_mean, x_std, 

def denorm_data(x_norm, x_mean, x_std):
    x = x_norm*x_std + x_mean
    return x

##Normalization
def scale_data(x):
    x_mean, _ = torch.min(x,0)
    x_std, _ = torch.max(x,0)
    x_norm = (x -x_mean)/(x_std -x_mean)
    x_norm[torch.isnan(x_norm)] = 0
    x_norm[torch.isinf(x_norm)] = 0
        
    return x_norm, x_mean, x_std,
        
def descale_data(x_norm, x_mean, x_std):
    x = x_norm*(x_std-x_mean) + x_mean
    return x

#Load data files from compressed

def get_dataset_N01():
    node_vars = (np.load(os.path.join(params.folder,'node_feats.npz')))["arr_0"]
    edge_vars = (np.load(os.path.join(params.folder,'edge_feats.npz')))["arr_0"]
    line_vars = (np.load(os.path.join(params.folder,'line_feats.npz')))["arr_0"]

    for n in range(params.n_files):   	
    	
        
        _cnode_data = (np.load(os.path.join(params.folder,str(n)+'_cnode_feats.npz')))["arr_0"]
        _cedge_data = (np.load(os.path.join(params.folder,str(n)+'_cedge_feats.npz')))["arr_0"]
        _cline_data = (np.load(os.path.join(params.folder,str(n)+'_cline_feats.npz')))["arr_0"]
       
        
        if n == 0:
            cnode_vars = _cnode_data
            cedge_vars = _cedge_data
            cline_vars = _cline_data
        else:
            cnode_vars = np.concatenate((_cnode_data , cnode_vars), axis=0)
            cedge_vars = np.concatenate((_cedge_data , cedge_vars), axis=0)
            cline_vars = np.concatenate((_cline_data , cline_vars), axis=0)
            
    Tnode_vars = np.concatenate((node_vars , cnode_vars), axis=0)
    Tedge_vars = np.concatenate((edge_vars , cedge_vars), axis=0)
    Tline_vars = np.concatenate((line_vars , cline_vars), axis=0)
    
    global base_adj
    global base_edgelist
    
    base_adj = edge_vars[0,...,0]
    base_edgelist = np.argwhere(np.asarray(base_adj) == 1)  
 
    return Tnode_vars, Tedge_vars, Tline_vars

def get_dataset_N2():    
    edge_vars = (np.load(os.path.join(params.folder,'edge_feats.npz')))["arr_0"]
            
    global base_adj
    global base_edgelist
    
    base_adj = edge_vars[0,...,0]
    base_edgelist = np.argwhere(np.asarray(base_adj) == 1)
    
    for n in range(params.n2_files):  
    	        
        _cnode_data = (np.load(os.path.join(params.folder,str(n)+'_c2node_feats.npz')))["arr_0"]
        _cedge_data = (np.load(os.path.join(params.folder,str(n)+'_c2edge_feats.npz')))["arr_0"]
        _cline_data = (np.load(os.path.join(params.folder,str(n)+'_c2line_feats.npz')))["arr_0"]
       
        
        if n == 0:
            c2node_vars = _cnode_data
            c2edge_vars = _cedge_data
            c2line_vars = _cline_data
        else:
            c2node_vars = np.concatenate((_cnode_data , c2node_vars), axis=0)
            c2edge_vars = np.concatenate((_cedge_data , c2edge_vars), axis=0)
            c2line_vars = np.concatenate((_cline_data , c2line_vars), axis=0)
            
    return c2node_vars, c2edge_vars, c2line_vars

def get_dataset_N3():
    
    edge_vars = (np.load(os.path.join(params.folder,'edge_feats.npz')))["arr_0"]
        
    global base_adj
    global base_edgelist
    
    base_adj = edge_vars[0,...,0]
    base_edgelist = np.argwhere(np.asarray(base_adj) == 1)
    
    for n in range(params.n3_files):  
    	        
        _cnode_data = (np.load(os.path.join(params.folder,str(n)+'_c3node_feats.npz')))["arr_0"]
        _cedge_data = (np.load(os.path.join(params.folder,str(n)+'_c3edge_feats.npz')))["arr_0"]
        _cline_data = (np.load(os.path.join(params.folder,str(n)+'_c3line_feats.npz')))["arr_0"]
       
        
        if n == 0:
            c3node_vars = _cnode_data
            c3edge_vars = _cedge_data
            c3line_vars = _cline_data
        else:
            c3node_vars = np.concatenate((_cnode_data , c3node_vars), axis=0)
            c3edge_vars = np.concatenate((_cedge_data , c3edge_vars), axis=0)
            c3line_vars = np.concatenate((_cline_data , c3line_vars), axis=0)
 
    return c3node_vars, c3edge_vars, c3line_vars

def get_dataset_collocation():
    node_vars = (np.load(os.path.join(params.folder,'node_col_pts.npz')))["arr_0"]
    edge_vars = (np.load(os.path.join(params.folder,'edge_col_pts.npz')))["arr_0"]
    line_vars = (np.load(os.path.join(params.folder,'line_col_pts.npz')))["arr_0"]

    for n in range(params.n_files):           
        _cnode_data = (np.load(os.path.join(params.folder,str(n)+'_cnode_col_pts.npz')))["arr_0"]
        _cedge_data = (np.load(os.path.join(params.folder,str(n)+'_cedge_col_pts.npz')))["arr_0"]
        _cline_data = (np.load(os.path.join(params.folder,str(n)+'_cline_col_pts.npz')))["arr_0"]
       
        
        if n == 0:
            cnode_vars = _cnode_data
            cedge_vars = _cedge_data
            cline_vars = _cline_data
        else:
            cnode_vars = np.concatenate((_cnode_data , cnode_vars), axis=0)
            cedge_vars = np.concatenate((_cedge_data , cedge_vars), axis=0)
            cline_vars = np.concatenate((_cline_data , cline_vars), axis=0)
            
    Tnode_vars = np.concatenate((node_vars , cnode_vars), axis=0)
    Tedge_vars = np.concatenate((edge_vars , cedge_vars), axis=0)
    Tline_vars = np.concatenate((line_vars , cline_vars), axis=0)
    
     
    return Tnode_vars, Tedge_vars, Tline_vars

# ## Data processing

def data_preparation(node_vars, edge_vars, line_vars):
    
    x, y, edge_attr, adj, adj_weighted, r_adj, ybus = [], [], [], [], [], [], []
    s_inj = []
    edge_list = []
    l_feats = []
    
    n_pts = node_vars.shape[0]
    #print(n_pts)
    
    #Node and line input matrices 
    #node_vars = {0:v_x, 1:v_y, 2:pload, 3:qload, 4:pgen, 5:qgen, 6:p_inj, 7:q_inj}
    #edge_vars = {0:A, 1:A_weighted, 2:A_nom, 3:A_weighted_nom, 4:A_red, 5:Y_bus} 
    #line_vars = {0:pl_fro_to, 1:ql_fro_to, 2:Y_line, 3:Y_shunt, 4:ln_load}
    
    #Node and line feature inputs 
    for pt in range(n_pts):
        
        #node features
        node_feats = []        
        node_feats = [node_vars[pt,..., 2], node_vars[pt,..., 3], node_vars[pt,..., 4]]
        node_feats = (np.asarray(node_feats))      

        #edge features
        edge_feats = []  
        edge_feats =  [edge_vars[pt,..., 5]]
        edge_feats = (np.asarray(edge_feats))     
            
        #targets
        targets = []
        #targets = [node_vars[pt,..., 0], node_vars[pt,..., 1], node_vars[pt,..., 5]]
        targets = [node_vars[pt,..., 0], node_vars[pt,..., 1], node_vars[pt,..., 5], node_vars[pt,..., 6], node_vars[pt,..., 7]]
        targets = (np.asarray(targets))
        
        #Edge lists
        ad = edge_vars[pt,..., 0]
        edges = np.argwhere(ad == 1)
        edges = edges.tolist()
        b_edges = base_edgelist.tolist()
        con_edges = [i for i, item in enumerate(b_edges) if item in edges]
        
        #Line features
        line_feats = []        
        line_feats = [line_vars[pt,..., 0], line_vars[pt,..., 1], line_vars[pt,..., 2], line_vars[pt,..., 3], line_vars[pt,..., 4], edge_vars[pt,..., 5] ]
        line_feats = (np.asarray(line_feats))
        
        elist = np.zeros(len(b_edges))
        for e in con_edges:
            elist[e] = 1
        
        
        x.append(node_feats)
        y.append(targets)
        edge_attr.append(edge_feats)
        edge_list.append(elist)
        l_feats.append(line_feats)
           
    #adjacency
    adj = np.real(edge_vars[..., 0])
    adj_weighted = edge_vars[..., 1]
    #adj_weighted = np.nan_to_num((1/adj_weighted), nan=0.0, posinf=0.0, neginf=0.0)
    
    #Identity
    I= torch.eye(params.n_b)#.reshape([1, 5, 5]).repeat(2, 1, 1, 1)#.to(S.device)
    adj = torch.tensor(adj, dtype=torch.float)
    adj = torch.add(adj , I).to(adj.device) # A +I_{N}
    
    # adj_weighted = torch.tensor(adj_weighted, dtype=torch.float)
    # adj_weighted = torch.add(adj_weighted , I).to(adj.device) 
    
    r_adj = edge_vars[..., 4]
    r_ybus = np.real(edge_vars[..., 5])
    im_ybus = np.imag(edge_vars[..., 5])
    
    x, y, edge_attr, adj, adj_weighted = np.asarray(x), np.asarray(y), np.asarray(edge_attr), np.asarray(adj), np.asarray(adj_weighted)
    r_adj = np.asarray(r_adj)
    r_ybus = (np.asarray(r_ybus))
    im_ybus = (np.asarray(im_ybus))
    edge_list = np.asarray(edge_list)
    l_feats = np.asarray(l_feats)#, dtype='complex_')
    
    
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    adj = torch.tensor(adj, dtype=torch.float)
    adj_weighted = torch.tensor(adj_weighted, dtype=torch.float)
    r_adj = torch.tensor(r_adj, dtype=torch.float)
    r_ybus = torch.tensor(r_ybus, dtype=torch.float)
    im_ybus = torch.tensor(im_ybus, dtype=torch.float)
    edge_list = torch.squeeze(torch.tensor(edge_list, dtype=torch.float))
    l_feats = torch.squeeze(torch.tensor(l_feats, dtype=torch.cfloat))
        
    return x, y, edge_attr, adj, adj_weighted, r_adj, r_ybus, im_ybus, edge_list, l_feats

def data_preparation_colocation(node_vars, edge_vars, line_vars):
    
    x, y, edge_attr, adj, adj_weighted, r_adj, ybus = [], [], [], [], [], [], []
    s_inj = []
    edge_list = []
    l_feats = []
    
    n_pts = node_vars.shape[0]
    #print(n_pts)
    
    #Node and line input matrices 
    #node_vars = {0:pload, 1:qload, 2:pgen}
    #edge_vars = {0:A, 1:A_weighted, 2:A_nom, 3:A_weighted_nom, 4:A_red, 5:Y_bus} 
    #line_vars = {0:Y_line, 1:Y_shunt, }
    
        
    #Node and line feature inputs 
    for pt in range(n_pts):
        
        #node features
        node_feats = []        
        node_feats = [node_vars[pt,..., 0], node_vars[pt,..., 1], node_vars[pt,..., 2]]
        node_feats = (np.asarray(node_feats))      

        #edge features
        edge_feats = []  
        edge_feats =  [edge_vars[pt,..., 5]]
        edge_feats = (np.asarray(edge_feats))     
            
        #targets
        targets = []
        targets = (np.asarray(targets))
        
        #Edge lists
        ad = edge_vars[pt,..., 0]
        edges = np.argwhere(ad == 1)
        edges = edges.tolist()
        b_edges = base_edgelist.tolist()
        con_edges = [i for i, item in enumerate(b_edges) if item in edges]
        
        #Line features
        line_feats = []
        line_feats = (np.asarray(line_feats))
        
        elist = np.zeros(len(b_edges))
        for e in con_edges:
            elist[e] = 1
        
        x.append(node_feats)
        y.append(targets)
        edge_attr.append(edge_feats)
        edge_list.append(elist)
        l_feats.append(line_feats)
           
    #adjacency
    adj = np.real(edge_vars[..., 0])
    adj_weighted = edge_vars[..., 1]
    #adj_weighted = np.nan_to_num((1/adj_weighted), nan=0.0, posinf=0.0, neginf=0.0)
    
    #Identity
    I= torch.eye(params.n_b)#.reshape([1, 5, 5]).repeat(2, 1, 1, 1)#.to(S.device)
    adj = torch.tensor(adj, dtype=torch.float)
    adj = torch.add(adj , I).to(adj.device) # A +I_{N}
    
    # adj_weighted = torch.tensor(adj_weighted, dtype=torch.float)
    # adj_weighted = torch.add(adj_weighted , I).to(adj.device) 
    
    r_adj = edge_vars[..., 4]
    r_ybus = np.real(edge_vars[..., 5])
    im_ybus = np.imag(edge_vars[..., 5])
    
    x, y, edge_attr, adj, adj_weighted = np.asarray(x), np.asarray(y), np.asarray(edge_attr), np.asarray(adj), np.asarray(adj_weighted)
    r_adj = np.asarray(r_adj)
    r_ybus = (np.asarray(r_ybus))
    im_ybus = (np.asarray(im_ybus))
    edge_list = np.asarray(edge_list)
    l_feats = np.asarray(l_feats)#, dtype='complex_')
    
    
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    adj = torch.tensor(adj, dtype=torch.float)
    adj_weighted = torch.tensor(adj_weighted, dtype=torch.float)
    r_adj = torch.tensor(r_adj, dtype=torch.float)
    r_ybus = torch.tensor(r_ybus, dtype=torch.float)
    im_ybus = torch.tensor(im_ybus, dtype=torch.float)
    edge_list = torch.squeeze(torch.tensor(edge_list, dtype=torch.float))
    l_feats = torch.squeeze(torch.tensor(l_feats, dtype=torch.cfloat))
        
    return x, y, edge_attr, adj, adj_weighted, r_adj, r_ybus, im_ybus, edge_list, l_feats

class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, adj, adj_wgtd, r_adj, r_ybus, i_ybus, e_list, ln_feats):
        
        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda:0" if use_cuda else "cpu")
        
        self.features = (features)#.to(device)
        self.targets = (targets)#.to(device)
        self.adj = (adj)#.to(device)
        self.adj_wgtd = (adj_wgtd)#.to(device)
        self.r_adj = (r_adj)#.to(device)
        self.r_ybus = (r_ybus)#.to(device)
        self.i_ybus = (i_ybus)#.to(device)
        self.e_list = (e_list)#.to(device)
        self.ln_feats = (ln_feats)
        
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        # Select sample
        X = self.features[idx,:]  
        Y = self.targets[idx,: ] 
        A = self.adj[idx,: ]
        A_wghtd = self.adj_wgtd[idx,: ]
        r_A = self.r_adj[idx,: ]
        r_Y_bus = self.r_ybus[idx,: ]
        im_Y_bus = self.i_ybus[idx,: ]
        edge_list = self.e_list[idx,: ]
        edge_list = self.e_list[idx,: ]
        line_feats = self.ln_feats[idx,: ]
        return X, Y, A, A_wghtd, r_A, r_Y_bus, im_Y_bus, edge_list, line_feats


def dataset_generator(typ, data_params): 
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.device(device) 
    
    if typ ==1 or typ == 2 or typ == 3:        
        if typ == 1:
            node_vars, edge_vars, line_vars = get_dataset_N01()
            X, y, edge_attr, adj, adj_weighted, r_adj, rybus, iybus, elist, l_feats = data_preparation(node_vars, edge_vars, line_vars)
            
        elif typ == 2:
            node_vars, edge_vars,line_vars = get_dataset_N2()
            X, y, edge_attr, adj, adj_weighted, r_adj, rybus, iybus, elist, l_feats = data_preparation(node_vars, edge_vars, line_vars)

        
        elif typ == 3:
            node_vars, edge_vars, line_vars  = get_dataset_N3()
            X, y, edge_attr, adj, adj_weighted, r_adj, rybus, iybus, elist, l_feats = data_preparation(node_vars, edge_vars, line_vars)
    
        #X, y, edge_attr, adj, adj_weighted, r_adj, rybus, iybus, elist, l_feats = data_preparation(node_vars, edge_vars, line_vars)

    elif typ ==4:
        node_vars, edge_vars, line_vars = get_dataset_collocation()
        X, y, edge_attr, adj, adj_weighted, r_adj, rybus, iybus, elist, l_feats = data_preparation_colocation(node_vars, edge_vars, line_vars)


    X_norm, X_mean, X_std = normalize_data(X, is_scale = False)
    y_norm, y_mean, y_std = normalize_data(y, is_scale = False)

    X_mean = X_mean.to(params.device)
    X_std = X_std.to(params.device)
    y_mean = y_mean.to(params.device)
    y_std = y_std.to(params.device)

    
    from sklearn.model_selection import train_test_split  
    
    if typ == 1:
        test_size=params.test_size_Tdata
    elif typ == 2:
        test_size=params.test_size_Cdata
    
    elif typ == 3:
        test_size=params.test_size_Cdata
    elif typ == 4:
        test_size=params.test_size_col
    
    
    x_train,x_test,y_train, y_test,adj_train, adj_test,adj_weighted_train,adj_weighted_test, r_adj_train, r_adj_test, \
        rybus_train, rybus_test, iybus_train, iybus_test, elist_train, elist_test, l_feats_train, l_feats_test = train_test_split(X_norm, y_norm,\
                     adj, adj_weighted, r_adj, rybus, iybus, elist, l_feats, test_size=test_size, train_size =params.train_size, random_state=18)
        
    
    # Dataset Generators 
    training_set = Dataset(features = x_train, targets = y_train, adj = adj_train, adj_wgtd=adj_weighted_train,
                           r_adj = r_adj_train, r_ybus = rybus_train, i_ybus = iybus_train, e_list = elist_train, ln_feats = l_feats_train)
    train_loader = torch.utils.data.DataLoader(training_set,**data_params)

    test_set = Dataset(features = x_test, targets = y_test, adj = adj_test, adj_wgtd=adj_weighted_test,
                           r_adj = r_adj_test, r_ybus = rybus_test, i_ybus = iybus_test, e_list = elist_test, ln_feats = l_feats_test)
    test_loader = torch.utils.data.DataLoader(test_set,**data_params)

    num_nodes = x_train.shape[2]
    num_infeat = x_train.shape[1]
    num_outfeat = y_train.shape[1]
    
    print(typ)
    
    print('Test set: ', len(test_set), ' Training set: ',len(training_set))
    
    return train_loader, test_loader, num_nodes, num_infeat, num_outfeat, y_mean, y_std

