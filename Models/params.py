# -*- coding: utf-8 -*-
"""
File to set the test network, training model and training parameters
"""


import numpy as np
import pandas as pd
from numpy import random
import math

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top


import torch
from torch import nn

import os
import os.path as osp

import torch.nn.functional as F



np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
torch.cuda.empty_cache()


#Data parameters
test_size_Tdata = 0.3
test_size_Cdata = 0.3
test_size_col = 0.1
train_size = 0.7

#Test network
data_args = {    
    'test_nw': pn.case24_ieee_rts(),  
}

n_b = (data_args['test_nw'].bus).shape[0]  #Number of buses
n_l = (data_args['test_nw'].line).shape[0]  #Number of lines
n_t = (data_args['test_nw'].trafo).shape[0]  #Number of transformers
#df_lines = data_args['test_nw'].line
#df_trans = data_args['test_nw'].trafo
ext =np.array(data_args['test_nw'].ext_grid.bus, dtype = np.int8) # slack

#Number of files in which to save data. Note that large datasets require excess RAM during loading
n_files = math.ceil(n_l/100)
n2_files = 3
n3_files = 5

base_mva = data_args['test_nw'].sn_mva
#v_base = (data_args['test_nw'].bus.vn_kv)[0]
#z_base = ((v_base*1e3)**2)/(base_mva*1e6) 
shunt_idx = np.asarray(data_args['test_nw'].shunt.bus)
line_fr_to = (data_args['test_nw'].line[['from_bus', 'to_bus']]).to_numpy()
tra_hv_lv = (data_args['test_nw'].trafo[['hv_bus', 'lv_bus']]).to_numpy()
l_df = data_args['test_nw'].line

vb = (data_args['test_nw'].bus[['vn_kv']]).to_numpy()
base_trafo = (data_args['test_nw'].trafo.sn_mva).to_numpy()

imax_ = np.array((l_df.loc[l_df['in_service'] == True, 'max_i_ka']).tolist())
parallel = np.array((l_df.loc[l_df['in_service'] == True, 'parallel']).tolist())
df = np.array((l_df.loc[l_df['in_service'] == True, 'df']).tolist())
imax = imax_*parallel*df

bus_shunt = np.zeros(n_b, dtype = 'complex_')
for n in range(data_args['test_nw'].shunt.shape[0]):

    if data_args['test_nw'].shunt['in_service'][n] == True:

        i = data_args['test_nw'].shunt['bus'][n]
        y_shunt = ((data_args['test_nw'].shunt['p_mw'][n] + 1j*data_args['test_nw'].shunt['q_mvar'][n])*data_args['test_nw'].shunt['step'][n])/data_args['test_nw'].sn_mva
        bus_shunt[i] = y_shunt

def get_gen_qlim(net):
    base_mva =net.sn_mva
    qgen = net.gen
    
    n_b = (net.bus).shape[0]
    qmin = np.zeros((n_b,1))
    qmax = np.zeros((n_b,1))
    for b in range(n_b):
        if b in qgen['bus'].unique():
            qmin[b,0] = qgen.loc[qgen['bus'] ==b, 'min_q_mvar'].item()/base_mva
            qmax[b,0] = qgen.loc[qgen['bus'] ==b, 'max_q_mvar'].item()/base_mva
    return qmin.T, qmax.T

qg_min, qg_max = get_gen_qlim(data_args['test_nw'])

def get_bus_vlim(net):
    bus = net.bus 	
    
    n_b = (net.bus).shape[0]
    vmin = np.zeros((n_b,1))
    vmax = np.zeros((n_b,1))
    for b in range(n_b):
        vmin[b,0] = bus.loc[bus.index ==b, 'min_vm_pu'].item()
        vmax[b,0] = bus.loc[bus.index ==b, 'max_vm_pu'].item()
    return vmin.T, vmax.T
vb_min, vb_max = get_bus_vlim(data_args['test_nw'])

folder = str(n_b) +'bus_data'


data_params = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 4,
	  'pin_memory': True}

#All Models training Parameters
args = {
    'n_layersGNN': 2, #Should not be less than 2
    'n_layersFC': 2,  #Should not be less than 2
    'n_layersGD': 3,
    'dropout': 0.1,
    'lr': 0.0005,
    'epochs': 1000,
    'n_hops' : 2,
    'readout' : False
     }
    
num_infeat = 3  
num_outfeat = 3
num_nodes = n_b

#GNN model params
 
#Without encoder   
# layersGNN = [num_infeat, 64,  64]#num_outfeat]

#with encoder :
layersGNN = [32,  32]
    
hops = [2,2,2]
layersFC = [5, num_outfeat]
sigma =  torch.tanh#nn.PReLU()#nn.LeakyReLU(0.1) #torch.tanh#F.relu torch.tanh
drop_edge = False
weighted = False
isploss = 1
patience = 5
case='c2'

#Guided dropout model params
#layersGD = [num_infeat, 10,  10, num_outfeat]
layersGD = [24 , 24, 24]
cond_units = 2
layersEn_Dc = [num_infeat, 1, num_outfeat]
#layersFC = [5, 5]

#Guided 2 params
hidden_size = 10*n_b

# model_folder = str(n_b) +'best_model'
# if not osp.exists(model_folder):
#     os.makedirs(model_folder)

#res_file = open(os.path.join(model_folder,'results.txt'), 'w')

''''
Set the percentage of collocation to be applied for each training instance with PINN
1 = 0% collocation points
0 = 100% collocation points
The size of the array indicates the number of models that will be trained
''''
#col_range = [0.8, 0.6, 0.4, 0.2, 0.01]
col_range = [1.0, 0.5]

n_collocation = 0.0
n_simulation = 1.0

ploss_factor = 0.1
mloss_factor = 1.0

#ploss_value = [0, 0.01, 0.01]
ploss_value = [0, 0.01]

#Include physics into model training
#physics = ['False', 'False', 'True']
physics = [ 'False', 'True']

'''
Models
GNN
Guided
Onehot

'''
#Choice of model to train
model = 'GNN'

if model == 'GNN':
    args['readout'] = True
    
model_folder = str(n_b)+model+'_'+str(data_params['batch_size'])+'_'+str(args['lr'])+'_' +'results'
if not osp.exists(model_folder):
    os.makedirs(model_folder)