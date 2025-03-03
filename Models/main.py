# -*- coding: utf-8 -*-
"""
Train and test graph-aware models
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
import csv
import sys

import params
import data
import train
import time


#print(params.device)
#start_time = time.time()


## Get data
## Dataset types
#Type  = 1 : combined n-0 and n-1
#Type  = 2 : n-2
#Type  = 3 : n-3
#Type  = 4 : collocation

with open(os.path.join(params.model_folder,str(params.n_b)+params.model +'results.csv'), mode="w+") as res_file:
    
   writer = csv.writer(res_file)
   #writer.writerow(('Run','col','Val loss', 'var','s loss', 'var', 'test loss', 'var','test lossv', 'varv', 'Mae', 'Maev', 'Mape', 'Mapev','no param'))
   writer.writerow(('run', 'n_collocation', 'physics', 'k', 'time',\
        'bus_L2', 'var_bus_L2', 'bus_AE', 'var_bus_AE', 'bus_MAPE', 'var_bus_MAPE',\
        'line_L2', 'var_line_L2', 'line_AE', 'var_line_AE', 'line_MAPE', 'var_line_MAPE',\
        'Bus accuracy', 'Bus precision', 'Line accuracy', 'line precision'        
         ))
    
    
   for p, l, physics in zip(params.col_range, params.ploss_value, params.physics):
        n_simulation = p
        n_collocation = 1 - p 
        ploss = l
       
        for run in range(1):          
            '''
            #Model training ----------------------------------------------------------------------------------------
            '''            
           
            train.main(params.device, case='c2', run = run, folder = params.model_folder, res_file = res_file, n_simulation=n_simulation, n_collocation=n_collocation, ploss = ploss, physics = physics)
           
            
            '''
            #Model testing ----------------------------------------------------------------------------------------
            ''' 
            ##Predictions
           
            file_name = str(run)+'_'+str(params.n_b) + '_' + str(int(n_collocation*100))+'_' + physics +'_' + str(ploss)+'_' + params.model 

            train.model_predictions(params.device, n_collocation, file_name, params.model_folder, run, case='c2', res_file= res_file, physics= physics, k = 2)
            train.model_predictions(params.device, n_collocation, file_name, params.model_folder, run, case='c2', res_file= res_file, physics= physics, k = 3)

