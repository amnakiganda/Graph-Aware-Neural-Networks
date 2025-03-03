# -*- coding: utf-8 -*-
"""
Training and testing the models
"""


import numpy as np

import math

import torch
from torch import nn


import os
import os.path as osp

import torch.nn.functional as F

import matplotlib.pyplot as plt

np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import csv
from time import process_time 

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import params
import models
import data

torch.set_float32_matmul_precision("high")

def lst_reshape(lst):
    array = []
    
    for x in lst:
        if x.shape[0] >= 1 :           
            if len(array) == 0 :
                 array = x
            else:
                array = np.concatenate((array , x ), axis=0)
    return array

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def confusionmatrix(actual, predicted, classes, normalize = False):
    """
    Generate a confusion matrix for multiple classification
    
    """
    l = actual.shape[1]
    con = []
    con_norm = []
    c = []
    normalize = True
    actual = np.array(actual, dtype='int') 
    predicted = np.array(predicted, dtype='int') 
    
    for i in range(l):
        classes = classes #sorted(set(actual.ravel()))
        matrix = [[0 for _ in classes] for _ in classes]
        imap   = {key: i for i, key in enumerate(classes)}
                
        for p, a in zip(predicted[:,i], actual[:,i]):
            matrix[imap[p]][imap[a]] += 1
        
        if normalize:
            sigma = sum([sum(matrix[imap[i]]) for i in classes])
            matrix_nom = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
           
        con.append((matrix))
        con_norm.append((matrix_nom))
        
        for i in range(len(classes)):
            row = sum(matrix[i][:])
            col = sum([row[i] for row in matrix]) # Can't Access Matrix Col Using matrix[:][i] With Vanilla Python
            n = sum([ sum(row) for row in matrix ]) 
            tp  = matrix[i][i]
            fp  = row - tp
            fn  = col - tp
            tn  = n - row - col + tp
            c.append(([tp,fp,fn,tn]))
            
    conf = np.reshape(np.array(c), (l,len(classes),4))
    
    return con, con_norm, conf

def get_conf_metrics(conf):
    TP = conf[...,0]
    FP = conf[...,1]
    FN = conf[...,2]
    TN =conf[...,3]
    Accuracy = (TP + TN)/(TN+TP+FN+FP)
    
    TP_Rate = TP/(TP+FN)
    #TN_Rate = TN/(FP+TN)
    precision = TP/(TP+FP)
    
    return Accuracy, TP_Rate,  precision

# ## Train model

def loss_function(pred, target):         
    
    _L2 = ((target[:,0:3,:] -pred)**2)
    
    MSE = torch.mean(_L2, axis = (0,2))
    
    _L1 = (torch.abs((target[:,0:3,:] - pred)))
    L1 = torch.mean(_L1, axis = (0,2))
    Linf = torch.linalg.vector_norm(_L2, ord =  float('inf'), dim=(0,2))
    
    return torch.sum(MSE) + 0.1*torch.sum(Linf)


def physics_loss(pred, mean, std, r_Y_bus, i_Y_bus, feat):
    
    #pred = descale_data(pred, mean, std)
    #E = diag(Vreal)
    #F = diag(Vimag)
    #(E · G + F · B) · Vreal + (F · G − E · B) · Vimag + Pload − Pgen = 0 
    #(F · G − E · B) · Vreal + (E · G + F · B) · Vimag + Qload − Qgen = 0 
        
    
    vdiagp = []
    vdiagq = []
    pl, ql, pg, qg = [], [],[],[]
    for i in range(len(pred)):
        vdiagp.append((pred[i,0,:]))
        vdiagq.append((pred[i,1,:]))
        pl.append((feat[i,0,:]))
        ql.append((feat[i,1,:]))
        pg.append((feat[i,2,:]))
        qg.append((pred[i,2,:]))
        
        #vdiagp.append((targets[i,0,:]))
        #vdiagq.append((targets[i,1,:]))
     
    vdiagp = (torch.stack(vdiagp))#.to(params.device)
    vdiagq = (torch.stack(vdiagq))#.to(params.device)
    
    pl = (torch.stack(pl))#.to(params.device)
    ql = (torch.stack(ql))#.to(params.device)
    pg = (torch.stack(pg))#.to(params.device)
    qg = (torch.stack(qg))#.to(params.device)    
    
         
    v = torch.complex(vdiagp, vdiagq).to(params.device)
    Y_bus = torch.complex(r_Y_bus, i_Y_bus).to(params.device)
    sl =  torch.complex(pl, ql).to(params.device)
    sg =  torch.complex(pg, qg).to(params.device)
    #s_shunt = (torch.tensor(params.bus_shunt, dtype=torch.cfloat))
    
    si = torch.zeros(v.shape).to(params.device)
    si_pred_l = si.type(torch.complex64)
    si_pred_r = si.type(torch.complex64)
    
    # #Power injection at node
    for i in range(params.n_b):
        si_pred_r[:,i] = sl[:,i] - sg[:,i] #+ s_shunt[i]*(v[:,i])**2
        
        for j in range(params.n_b):
            si_pred_l[:,i] = si_pred_l[:,i] + torch.conj_physical(v[:,i])*(v[:,j]*(1*Y_bus[:,i,j]))
    
    ##removing injection at slack bus
    slack = params.ext
    si_pred_l[:,slack] = 0 
    
    pi_loss = si_pred_l.real.abs() - si_pred_r.real.abs()
    qi_loss = si_pred_l.imag.abs() - si_pred_r.imag.abs()
    
    
    #Qlimit satisfaction
    m = F.relu
    qg_min = torch.from_numpy((params.qg_min)).to(params.device)
    qg_max = torch.from_numpy((params.qg_max)).to(params.device)
    qmin_fes = m(qg_min - qg)
    qmax_fes = m(qg - qg_max)
        
    loss_p = torch.mean(pi_loss**2) #0.01*torch.mean(pi_loss)
    loss_q = torch.mean(qi_loss**2) + torch.mean(qmin_fes**2) + torch.mean(qmax_fes**2) #0.01*torch.mean(qi_loss)
    
    return loss_p, loss_q 

def line_loss(line_batch, pred, target):
    vp_pred = (pred[:,0,:]).to(params.device)
    vq_pred = (pred[:,1,:]).to(params.device)
    pl_target = (line_batch[:,0,:].real).to(params.device)
    ql_target = (line_batch[:,1,:].real).to(params.device)
    y_line = (line_batch[:,2,:]).to(params.device)
    y_shnt = (line_batch[:,3,:]).to(params.device)
    y_bus = (line_batch[:,5,:]).to(params.device)
    ln_load = (line_batch[:,4,:]).to(params.device)
        
    V = torch.complex(vp_pred, vq_pred)   
    
    Sij = torch.zeros((len(pred), params.n_b, params.n_b), dtype = torch.cfloat)
    Iij = torch.zeros((len(pred), params.n_b, params.n_b), dtype = torch.cfloat)
    ln_trafo_loading_pred = torch.zeros((len(pred), params.n_l + params.n_t), dtype = torch.float)
    ln_trafo_loading_tag = torch.zeros((len(pred), params.n_l + params.n_t), dtype = torch.float)
    
    for l in range(params.n_l):
        i = params.line_fr_to[l,0]
        j = params.line_fr_to[l,1] 
        Iij[:,i,j] =(torch.mul( y_line[:,i,j],(V[:,i] - V[:,j])) + torch.mul(y_shnt[:,i,j],V[:,i] )).to(params.device)
        Iij[:,j,i] =(torch.mul( y_line[:,j,i],(V[:,j] - V[:,i])) + torch.mul(y_shnt[:,j,i],V[:,j] )).to(params.device)
        
        #Iij_conj =  torch.conj_physical(Iij[:,i,j]).to(params.device)
        Sij[:,i,j] =torch.mul( V[:,i], torch.conj_physical(Iij[:,i,j]).to(params.device))
        Sij[:,j,i] =torch.mul( V[:,j], torch.conj_physical(Iij[:,j,i]).to(params.device)) 
        
        ib = params.base_mva/(params.vb[i]*math.sqrt(3))
        #print(params.imax[l])
        #ln_trafo_loading[:,i,j] = ln_trafo_loading[:,j,i] = torch.max(torch.abs(Iij[:,i,j]) , torch.abs(Iij[:,j,i]))*ib[0]*100/params.imax[l]
        ln_trafo_loading_pred[:,l]  =  torch.max(torch.abs(Iij[:,i,j]) , torch.abs(Iij[:,j,i]))*ib[0]*100/params.imax[l]
        ln_trafo_loading_tag[:,l]  = ln_load[:,i,j]
    
    indx = params.n_l-1
    indx = indx + 1
    for t in range(params.n_t):
        i = params.tra_hv_lv[t,0]
        j = params.tra_hv_lv[t,1]
        Iij[:,i,j] =(torch.mul( y_line[:,i,j],(V[:,i] )) + torch.mul(y_bus[:,i,j],V[:,j] )).to(params.device)
        Iij[:,j,i] =(torch.mul( y_line[:,j,i],(V[:,j] )) + torch.mul(y_bus[:,i,j],V[:,i] )).to(params.device)
        
        Sij[:,i,j] =torch.mul( V[:,i], torch.conj_physical(Iij[:,i,j]).to(params.device))
        Sij[:,j,i] =torch.mul( V[:,j], torch.conj_physical(Iij[:,j,i]).to(params.device))  
        
        
        ln_trafo_loading_pred[:,indx]  =  torch.max(torch.abs(Iij[:,i,j]) , torch.abs(Iij[:,j,i]))*100*params.base_mva/params.base_trafo[t]
        ln_trafo_loading_tag[:,indx]  = ln_load[:,i,j]
        indx = indx + 1
        
    Sij_target = ((torch.complex(pl_target, ql_target)).abs()).to(params.device)
    Sij_pred = ((Sij).abs()).to(params.device)
    
    s_AE =  (torch.abs((Sij_target - Sij_pred))).cpu().detach().numpy()
    
    s_MAPE = ((torch.abs((Sij_pred - Sij_target) / (Sij_target  - torch.mean(Sij_target)))) * 100).cpu().detach()
    
    
    s_MAPE = np.nan_to_num(s_MAPE, nan=0.0, posinf=0.0, neginf=0.0)
    s_L2 = ((Sij_target -Sij_pred)**2).cpu().detach().numpy()
        
    ln_load_enc_target = torch.where(ln_trafo_loading_tag.real.to(params.device) > 100, 1, 0)
    ln_load_enc_pred = torch.where(ln_trafo_loading_pred > 100, 1, 0)
    classes = [0,1]
        
    ln_load_enc_target =  ln_load_enc_target.cpu().detach().numpy()
    ln_load_enc_pred = ln_load_enc_pred.cpu().detach().numpy()
    
    #print('====================================================================')
    con, con_norm, conf = confusionmatrix(ln_load_enc_target, ln_load_enc_pred, classes)
    #print('====================================================================')
    
    Sij_target = Sij_target.cpu().detach().numpy()
    Sij_pred =Sij_pred.cpu().detach().numpy()
    ln_trafo_loading_tag=ln_trafo_loading_tag.cpu().detach().numpy() 
    ln_trafo_loading_pred =ln_trafo_loading_pred.cpu().detach().numpy()

    
    return s_MAPE, s_AE, s_L2, conf, Sij_target, Sij_pred, ln_trafo_loading_tag, ln_trafo_loading_pred


def train(n_model, train_data, col_train_data, y_mean, y_std, optimizer, device, drop_edge, weighted, isploss, _n_sim, _n_col, model, ploss, physics):
    
    n_model.train()
    
    running_loss = 0.0
    step = 0
    train_acc =[]
    
    for batch_input, batch_labels, adj, adj_weighted, r_adj, r_Y_bus, i_Y_bus, edges, _ in (train_data):    
        
        batch_input = (batch_input).to(device)
        batch_labels = (batch_labels).to(device) 
        adj = (adj).to(device) 
        adj_weighted = (adj_weighted).to(device) 
        r_adj = (r_adj).to(device) 
        r_Y_bus = (r_Y_bus).to(device)
        i_Y_bus = (i_Y_bus).to(device)
        edges = (edges).to(device)
               
        #optimizer.zero_grad() 
        for param in n_model.parameters():              
    	    param.grad = None
        
        #Forward pass
    
        if model == 'GNN':
            if drop_edge:
                 n_pred = n_model(batch_input, r_adj)
            elif weighted:
                 n_pred = n_model(batch_input, adj_weighted)
            else:
                 n_pred = n_model(batch_input, adj)
                 
         
        if model == 'Guided':
            n_pred = n_model(batch_input, edges, params.cond_units)
    
        
        n_col = math.ceil(batch_input.size(dim=0) * _n_col)
        n_sim = batch_input.size(dim=0) - n_col
        #n_tot = n_col + n_sim
        
        #Batch split
        batch_spl = torch.split(batch_input, [n_sim, n_col])
        pred_spl = torch.split(n_pred, [n_sim, n_col])
        label_spl = torch.split(batch_labels, [n_sim, n_col])
        rY_spl = torch.split(r_Y_bus, [n_sim, n_col])
        iY_spl = torch.split(i_Y_bus, [n_sim, n_col])
        
        # Calculating the loss and gradients
        loss = torch.tensor(0)
        pi_loss, qi_loss = torch.tensor(0), torch.tensor(0)
        if n_sim > 0:            
            if n_col == 0:                 
                loss = loss_function(pred_spl[0], label_spl[0])                                 
            elif n_col > 0:
                loss = loss_function(pred_spl[0], label_spl[0]) 
                #pi_loss_s, qi_loss_s = physics_loss(pred_spl[0], y_mean, y_std, rY_spl[0], iY_spl[0],  batch_spl[0])  
                pi_loss_c, qi_loss_c = physics_loss(pred_spl[1], y_mean, y_std, rY_spl[1], iY_spl[1],  batch_spl[1])  
                pi_loss = pi_loss_c #+ pi_loss_s
                qi_loss = qi_loss_c #+ qi_loss_s
        
        
        tloss =  (params.mloss_factor*torch.squeeze(loss)) + torch.squeeze(ploss*pi_loss + ploss*qi_loss)
        
        # Backward pass + weight update
        tloss.backward()  
        optimizer.step() 
        
        # Update tracking
        running_loss += tloss.item()
        step += 1
          
    if physics:
        for batch_input, _ , adj, adj_weighted, r_adj, r_Y_bus, i_Y_bus, edges, _ in (col_train_data):
            batch_input = (batch_input).to(device)
            batch_labels = (batch_labels).to(device) 
            adj = (adj).to(device) 
            adj_weighted = (adj_weighted).to(device) 
            r_adj = (r_adj).to(device) 
            r_Y_bus = (r_Y_bus).to(device)
            i_Y_bus = (i_Y_bus).to(device)
            edges = (edges).to(device)
                   
            optimizer.zero_grad(set_to_none=True)
            
            #Forward pass        
            if model == 'GNN':
                if drop_edge:
                     n_pred = n_model(batch_input, r_adj)
                elif weighted:
                     n_pred = n_model(batch_input, adj_weighted)
                else:
                     n_pred = n_model(batch_input, adj)                     
             
            if model == 'Guided':
                n_pred = n_model(batch_input, edges, params.cond_units)
        
            
            n_col = math.ceil(batch_input.size(dim=0) * _n_col)
            n_sim = batch_input.size(dim=0) - n_col
            
            #Batch split
            batch_spl = batch_input
            pred_spl = n_pred
            rY_spl = r_Y_bus
            iY_spl = i_Y_bus
            
            # Calculating the loss and gradients
            pi_loss, qi_loss = torch.tensor(0), torch.tensor(0)
            pi_loss_c, qi_loss_c = physics_loss(pred_spl, y_mean, y_std, rY_spl, iY_spl,  batch_spl)  
            pi_loss = pi_loss_c 
            qi_loss = qi_loss_c
            
            tloss = torch.squeeze(ploss*pi_loss + ploss*qi_loss)
            
            # Backward pass + weight update
            tloss.backward()  
            optimizer.step() 
            
            # Update tracking
            running_loss += loss.item()
            step += 1
    
    return running_loss/step


def validate(n_model, test_data,y_mean, y_std, device, drop_edge, weighted, model):
    
    n_model.eval()
    
    running_loss = 0.0
    step = 0
    
    AE = []
    L2 = []             
    
    for batch_input, batch_labels, adj, adj_weighted, r_adj, r_Y_bus, i_Y_bus, edges, lines  in (test_data):
        
        batch_input = (batch_input).to(device)
        batch_labels = (batch_labels).to(device) 
        adj = (adj).to(device) 
        adj_weighted = (adj_weighted).to(device) 
        r_adj = (r_adj).to(device) 
        r_Y_bus = (r_Y_bus).to(device)
        i_Y_bus = (i_Y_bus).to(device)
        edges = (edges).to(device)
        lines = (lines).to(device)
        batch_labels = batch_labels[:,0:3,:]
        
        with torch.no_grad():
            if model == 'GNN':
                if drop_edge:
                     n_pred = n_model(batch_input, r_adj)
                elif weighted:
                     n_pred = n_model(batch_input, adj_weighted)
                else:
                     n_pred = n_model(batch_input, adj)
                     
             
            if model == 'Guided':
                
                n_pred = n_model(batch_input, edges, params.cond_units)
                           
            loss_fn = nn.MSELoss()
            #loss_fn = nn.L1Loss()           
                
            loss = loss_fn(n_pred, batch_labels)          
            
        running_loss += loss.item()
        step += 1  
        
        
        _L2 = ((batch_labels -n_pred)**2).cpu().detach().numpy()

        error = (torch.abs((batch_labels - n_pred))).cpu().detach().numpy()
        
        AE.append(error)
               
    AE_ = lst_reshape(AE)
    L2_ = lst_reshape(L2)
         
    
    return running_loss/step, AE_, L2_

def validate_best(n_model, test_data,y_mean, y_std, device, drop_edge, weighted, model, folder,file_name, k):
    
    n_model.eval()
    
    running_loss = 0.0
    step = 0
    
    AE = []
    MAPE = []
    L2 = [] 
    S_AE = []
    S_MAPE = []
    S_L2 = [] 
    line_CONF = []  
    node_CONF = []

    predicted_node = []
    actual_node = []
    
    predicted_line = []
    actual_line = []
    predicted_line_load = []
    actual_line_load = []
    
    for batch_input, batch_labels, adj, adj_weighted, r_adj, r_Y_bus, i_Y_bus, edges, lines  in (test_data):
        
        batch_input = (batch_input).to(device)
        batch_labels = (batch_labels).to(device) 
        adj = (adj).to(device) 
        adj_weighted = (adj_weighted).to(device) 
        r_adj = (r_adj).to(device) 
        r_Y_bus = (r_Y_bus).to(device)
        i_Y_bus = (i_Y_bus).to(device)
        edges = (edges).to(device)
        lines = (lines).to(device)
        batch_labels = batch_labels[:,0:3,:]
        
        with torch.no_grad():
            if model == 'GNN':
                if drop_edge:
                     n_pred = n_model(batch_input, r_adj)
                elif weighted:
                     n_pred = n_model(batch_input, adj_weighted)
                else:
                     n_pred = n_model(batch_input, adj)
                     
             
            if model == 'Guided':
                n_pred = n_model(batch_input, edges, params.cond_units)
            
            loss_fn = nn.MSELoss()
            #loss_fn = nn.L1Loss()           
                
            loss = loss_fn(n_pred, batch_labels)          
            
        running_loss += loss.item()
        step += 1  
        
        _MAPE = ((torch.abs((batch_labels - n_pred) / (batch_labels ))) * 100).cpu().detach()#.numpy()
              
        
        _MAPE[_MAPE != _MAPE] = 0
        _MAPE = np.nan_to_num(_MAPE, nan=0.0, posinf=0.0, neginf=0.0) 
        _L2 = ((batch_labels -n_pred)**2).cpu().detach().numpy()

        error = (torch.abs((batch_labels - n_pred))).cpu().detach().numpy()
        
        AE.append(error)
        MAPE.append(_MAPE)
        L2.append(_L2)
        
        node_vol_target =  (torch.complex(batch_labels[:,0,:], batch_labels[:,1,:])).abs()
        node_vol_pred =  (torch.complex(n_pred[:,0,:], n_pred[:,1,:])).abs()
        #limits
        vb_min = torch.from_numpy((params.vb_min)).to(params.device)
        vb_max = torch.from_numpy((params.vb_max)).to(params.device)
                    
        node_vol_enc_target = torch.where((node_vol_target < vb_min) , 1, node_vol_target)
        node_vol_enc_target = torch.where( (vb_max < node_vol_target), 2, node_vol_enc_target)
        node_vol_enc_target = torch.where((node_vol_target >= vb_min) & (node_vol_target <=vb_max ), 0, node_vol_enc_target)
        
                
        node_vol_enc_pred = torch.where((node_vol_pred < vb_min) , 1, node_vol_pred)
        node_vol_enc_pred = torch.where( (vb_max < node_vol_pred), 2, node_vol_enc_pred)
        node_vol_enc_pred = torch.where((node_vol_pred >= vb_min) & (node_vol_pred <= vb_max ), 0, node_vol_enc_pred)
        classes = [0,1,2]
        
        node_vol_enc_target =  node_vol_enc_target.cpu().detach().numpy()
        node_vol_enc_pred = node_vol_enc_pred.cpu().detach().numpy()
        con, con_norm, conf = confusionmatrix(node_vol_enc_target, node_vol_enc_pred, classes)       
        node_CONF.append(conf) 
        
        s_MAPE, s_AE, s_L2, line_conf, Sij_target, Sij_pred, ln_trafo_loading_tag, ln_trafo_loading_pred = line_loss(lines, n_pred, batch_labels)
        S_AE.append(s_AE)
        S_MAPE.append(s_MAPE)
        S_L2.append(s_L2)
        
        line_CONF.append(line_conf)
        
        predicted_node.append((torch.complex(n_pred[:,0,:], n_pred[:,1,:])).cpu().detach().numpy()) 
        actual_node.append((torch.complex(batch_labels[:,0,:], batch_labels[:,1,:])).cpu().detach().numpy()) 
        
        predicted_line.append(Sij_pred) 
        actual_line.append(Sij_target) 
        
        predicted_line_load.append(ln_trafo_loading_pred) 
        actual_line_load.append(ln_trafo_loading_tag)
        
    AE_ = lst_reshape(AE)
    MAPE_ = lst_reshape(MAPE)
    L2_ = lst_reshape(L2)
    
    S_AE_ = lst_reshape(S_AE)
    S_MAPE_ = lst_reshape(S_MAPE)
    S_L2_ = lst_reshape(S_L2)  
    
    predicted_node = lst_reshape(predicted_node)
    actual_node = lst_reshape(actual_node)
    
    predicted_line = lst_reshape(predicted_line)
    actual_line = lst_reshape(actual_line)
    
    predicted_line_load = lst_reshape(predicted_line_load)
    actual_line_load = lst_reshape(actual_line_load)
    
    np.savez_compressed( os.path.join(folder,file_name+'predicted_node_n'+str(k)), predicted_node)
    np.savez_compressed( os.path.join(folder,file_name+'actual_node_n'+str(k)), actual_node)
    np.savez_compressed( os.path.join(folder,file_name+'predicted_line_n'+str(k)), predicted_line)
    np.savez_compressed( os.path.join(folder,file_name+'actual_line_n'+str(k)), actual_line)
    np.savez_compressed( os.path.join(folder,file_name+'predicted_line_load_n'+str(k)), predicted_line_load)
    np.savez_compressed( os.path.join(folder,file_name+'actual_line_load_n'+str(k)), actual_line_load)      
    
    return running_loss/step, AE_, MAPE_, L2_, S_AE_,  S_MAPE_,  S_L2_, line_CONF, node_CONF


def model_testing(t_model, test_data, _mean, _std, device, model, folder,file_name, k):
        t_model.eval()
        
        running_loss = 0.0
        step = 0
        AE = []
        MAPE = []
        L2 = [] 
        S_AE = []
        S_MAPE = []
        S_L2 = []
        line_CONF = []  
        node_CONF = []  
        
        predicted_node = []
        actual_node = []
        
        predicted_line = []
        actual_line = []
        
        predicted_line_load = []
        actual_line_load = []
        
        
        for batch_input, batch_labels, adj, adj_weighted, r_adj, r_Y_bus, i_Y_bus, edges, lines  in (test_data):
            
            batch_input = (batch_input).to(device)
            batch_labels = (batch_labels).to(device) 
            adj = (adj).to(device) 
            adj_weighted = (adj_weighted).to(device) 
            r_adj = (r_adj).to(device) 
            r_Y_bus = (r_Y_bus).to(device)
            i_Y_bus = (i_Y_bus).to(device)          
    
            #pred = t_model(batch_input, adj)   
            
            if model == 'GNN':
                if  params.weighted:
                     pred = t_model(batch_input, adj_weighted)
                else:
                     pred = t_model(batch_input, adj)
             
            if model == 'Guided':
                #pred = t_model(batch_input, edges)
                pred = t_model(batch_input, edges, params.cond_units)
    
                        
            target = batch_labels[:,0:3,:]
                        
            #node loss metrics    
            loss = loss_function(pred, target) 
    
            _MAPE = ((torch.abs((target - pred) / (target))) * 100).cpu().detach()
            
            _MAPE[_MAPE != _MAPE] = 0
            _MAPE = np.nan_to_num(_MAPE, nan=0.0, posinf=0.0, neginf=0.0)
            _L2 = ((target -pred)**2).cpu().detach()
    
            running_loss += loss.item()
    
            error = (torch.abs((target - pred))).cpu().detach()
            
            AE.append(error)
            MAPE.append(_MAPE)
            L2.append(_L2)
            
            node_vol_target =  (torch.complex(target[:,0,:], target[:,1,:])).abs()
            node_vol_pred =  (torch.complex(pred[:,0,:], pred[:,1,:])).abs()
            #limits
            vb_min = torch.from_numpy((params.vb_min)).to(params.device)
            vb_max = torch.from_numpy((params.vb_max)).to(params.device)
                        
            node_vol_enc_target = torch.where((node_vol_target < vb_min) , 1, node_vol_target)
            node_vol_enc_target = torch.where( (vb_max < node_vol_target), 2, node_vol_enc_target)
            node_vol_enc_target = torch.where((node_vol_target >= vb_min) & (node_vol_target <= vb_max ), 0, node_vol_enc_target)
            
            
            node_vol_enc_pred = torch.where((node_vol_pred < 0.9) , 1, node_vol_pred)
            node_vol_enc_pred = torch.where( (1.1 < node_vol_pred), 2, node_vol_enc_pred)
            node_vol_enc_pred = torch.where((node_vol_pred >= 0.9) & (node_vol_pred <= 1.1 ), 0, node_vol_enc_pred)
            
            classes = [0,1,2]
            
            node_vol_enc_target =  node_vol_enc_target.cpu().detach().numpy()
            node_vol_enc_pred = node_vol_enc_pred.cpu().detach().numpy()
            con, con_norm, conf = confusionmatrix(node_vol_enc_target, node_vol_enc_pred, classes)
            
            
            node_CONF.append(conf) 
            
            #line loss metrics            
            s_MAPE, s_AE, s_L2, line_conf, Sij_target, Sij_pred, ln_trafo_loading_tag, ln_trafo_loading_pred = line_loss(lines, pred, batch_labels)
            
            S_AE.append(s_AE)
            S_MAPE.append(s_MAPE)
            S_L2.append(s_L2)
            
            line_CONF.append(line_conf) 
            
            predicted_node.append((torch.complex(pred[:,0,:], pred[:,1,:])).cpu().detach().numpy()) 
            actual_node.append((torch.complex(target[:,0,:], target[:,1,:])).cpu().detach().numpy()) 
            
            predicted_line.append(Sij_pred) 
            actual_line.append(Sij_target) 
            
            predicted_line_load.append(ln_trafo_loading_pred) 
            actual_line_load.append(ln_trafo_loading_tag)
            
            step += 1 
            
        _loss = running_loss/step
        
        AE_ = lst_reshape(AE)
        MAPE_ = lst_reshape(MAPE)
        L2_ = lst_reshape(L2)
        
        S_AE_ = lst_reshape(S_AE)
        S_MAPE_ = lst_reshape(S_MAPE)
        S_L2_ = lst_reshape(S_L2)
        
        predicted_node = lst_reshape(predicted_node)
        actual_node = lst_reshape(actual_node)
        
        predicted_line = lst_reshape(predicted_line)
        actual_line = lst_reshape(actual_line)
        
        predicted_line_load = lst_reshape(predicted_line_load)
        actual_line_load = lst_reshape(actual_line_load)
        
        np.savez_compressed( os.path.join(folder,file_name+'predicted_node_n'+str(k)), predicted_node)
        np.savez_compressed( os.path.join(folder,file_name+'actual_node_n'+str(k)), actual_node)
        np.savez_compressed( os.path.join(folder,file_name+'predicted_line_n'+str(k)), predicted_line)
        np.savez_compressed( os.path.join(folder,file_name+'actual_line_n'+str(k)), actual_line)
        np.savez_compressed( os.path.join(folder,file_name+'predicted_line_load_n'+str(k)), predicted_line_load)
        np.savez_compressed( os.path.join(folder,file_name+'actual_line_load_n'+str(k)), actual_line_load)
        
            
        return MAPE_, AE_, L2_, S_MAPE_, S_AE_, S_L2_, line_CONF, node_CONF
    

def model_predictions(device, n_collocation, file_name, folder, run, case,res_file, physics, k):
    '''
    #Best Model testing ----------------------------------------------------------------------------------------
    '''    
    model = params.model
    
    if model == 'GNN':
        t_model = models.GNN(params.args['n_layersGNN'], params.args['n_layersFC'], params.hops, params.layersGNN, params.layersFC, \
                             params.num_nodes, params.args['dropout'], params.sigma, params.args['readout']).to(device)
    
    if model == 'Guided':
        t_model = models.GD(params.args['n_layersGD'], params.args['n_layersFC'], params.layersGD, params.layersFC, params.n_b, params.args['dropout'], params.sigma, params.args['readout']).to(params.device) 
   
    if n_collocation > 0:
        t_model.load_state_dict(torch.load(os.path.join(params.model_folder,file_name+'_'+'pinn_best_model.pth')), strict=False)   
    else:        
        t_model.load_state_dict(torch.load(os.path.join(params.model_folder,file_name+'_'+'best_model.pth')), strict=False)
        
    _mean = 0 #c2y_mean
    _std = 0 #c2y_std
    
    ##### N-2 testing 
       
    #k = 2
    if k == 2:
        c2train_loader, _, c2num_nodes, c2num_infeat, c2num_outfeat, c2y_mean, c2y_std =   data.dataset_generator(typ = 2, data_params=params.data_params)
        
        test_data = c2train_loader        
    
        start_test_n2 = process_time()
        bus_MAPE2, bus_AE2, bus_L22, line_MAPE2, line_AE2,  line_L22, line_conf, bus_conf   = model_testing(t_model, test_data, _mean, _std, device, model,folder,file_name,k)
        end_test_n2 = process_time()
        test_n2_time = end_test_n2 - start_test_n2
        with open(os.path.join(params.model_folder,'time_' + file_name +'.txt'), "a") as text_file:
            text_file.write("N2_test_time: %f \n" % test_n2_time)
            
        bus_MAPE2, bus_AE2, bus_L22, line_MAPE2, line_AE2,  line_L22 = np.array(bus_MAPE2), np.array(bus_AE2), np.array(bus_L22), np.array(line_MAPE2),  np.array(line_AE2),  np.array(line_L22)
        #sTP, sTN, sFP, sFN  = np.array(sTP), np.array(sTN), np.array(sFP), np.array(sFN) 
        
        line_conf = np.array(line_conf)
        line_conf = np.nan_to_num(line_conf, nan=0.0, posinf=0.0, neginf=0.0)
        line_conf_sum = np.sum(line_conf, axis=(0))
        
        line_conf_sum = np.array(line_conf_sum)
        line_Accuracy, line_TP_Rate, line_precision = get_conf_metrics(line_conf_sum)
        
            
        bus_conf = np.array(bus_conf)    
        bus_conf_sum = np.sum(bus_conf, axis=(0))
        
        
        bus_conf_sum = np.array(bus_conf_sum)
        bus_Accuracy, bus_TP_Rate, bus_precision = get_conf_metrics(bus_conf_sum)
        
        bus_TP_Rate[bus_TP_Rate == 0] = np.nan
        bus_precision[bus_precision ==0] = np.nan
        
        np.savez_compressed( os.path.join(folder,file_name+'_bus_Accuracy_n2'), bus_Accuracy)
        np.savez_compressed( os.path.join(folder,file_name +'_bus_TP_Rate_n2'), bus_TP_Rate)
        np.savez_compressed( os.path.join(folder,file_name+'_bus_precision_n2'), bus_precision)
        
        np.savez_compressed( os.path.join(folder,file_name+'_line_Accuracy_n2'), line_Accuracy)
        np.savez_compressed( os.path.join(folder,file_name+'_line_TP_Rate_n2'), line_TP_Rate)
        np.savez_compressed( os.path.join(folder,file_name+'_line_precision_n2'), line_precision)
            
        
        print('.................N-2..........................................')
        print_res(bus_MAPE2, bus_AE2, bus_L22, line_MAPE2, line_AE2,  line_L22, k, run, case, n_collocation, folder, file_name)
        
        bus_MAPE2 = np.squeeze(np.array(bus_MAPE2))
        bus_AE2 = np.squeeze(np.array(bus_AE2))
        bus_L22 = np.squeeze(np.array(bus_L22))
        line_MAPE2 = np.squeeze(np.array(line_MAPE2))
        line_AE2 = np.squeeze(np.array(line_AE2))
        line_L22  = np.squeeze(np.array(line_L22))
        
        writer = csv.writer(res_file)
        row2 = (
            run,
            n_collocation, physics, k, test_n2_time,
            np.mean(bus_L22), np.var(bus_L22),
            np.mean(bus_AE2), np.var(bus_AE2),
            np.mean(bus_MAPE2), np.var(bus_MAPE2),
            np.mean(line_L22), np.var(line_L22),
            np.mean(line_AE2), np.var(line_AE2),
            np.mean(line_MAPE2), np.var(line_MAPE2),
            np.nanmean(bus_Accuracy, axis=(1)), 
            np.nanmean(bus_precision, axis=(0)),
            np.nanmean(line_Accuracy, axis=(1)), 
            np.nanmean(line_precision, axis=(0)),
            )
        writer.writerow(row2)
    
        
    ##### N-3 testing  
    if k == 3:
        c3train_loader, _, c3num_nodes, c3num_infeat, c3num_outfeat, c3y_mean, c3y_std =   data.dataset_generator(typ = 3, data_params=params.data_params)
        test_data2 = c3train_loader
        
        #k = 3
        start_test_n3 = process_time()
        bus_MAPE3, bus_AE3, bus_L23, line_MAPE3, line_AE3,  line_L23, line_conf, bus_conf = model_testing(t_model, test_data2, _mean, _std, device, model,folder,file_name,k)
        end_test_n3 = process_time()
        
        test_n3_time = end_test_n3 - start_test_n3 
        
        with open(os.path.join(params.model_folder,'time_' + file_name +'.txt'), "a") as text_file:
            text_file.write("N3_test_time: %f \n" % test_n3_time)
        
        bus_MAPE3, bus_AE3, bus_L23, line_MAPE3, line_AE3,  line_L23 = np.array(bus_MAPE3), np.array(bus_AE3), np.array(bus_L23), np.array(line_MAPE3),  np.array(line_AE3),  np.array(line_L23)
        #sTP, sTN, sFP, sFN  = np.array(sTP), np.array(sTN), np.array(sFP), np.array(sFN)  
        
            
        line_conf = np.array(line_conf)
        line_conf = np.nan_to_num(line_conf, nan=0.0, posinf=0.0, neginf=0.0)
        line_conf_sum = np.sum(line_conf, axis=(0))
        
        line_conf_sum = np.array(line_conf_sum)
        line_Accuracy, line_TP_Rate, line_precision = get_conf_metrics(line_conf_sum)
        
            
        bus_conf = np.array(bus_conf)    
        bus_conf_sum = np.sum(bus_conf, axis=(0))        
        
        bus_conf_sum = np.array(bus_conf_sum)
        bus_Accuracy, bus_TP_Rate, bus_precision = get_conf_metrics(bus_conf_sum)
        
        
        bus_TP_Rate[bus_TP_Rate == 0] = np.nan
        bus_precision[bus_precision ==0] = np.nan
        
        np.savez_compressed( os.path.join(folder,file_name+'_bus_Accuracy_n3'), bus_Accuracy)
        np.savez_compressed( os.path.join(folder,file_name+'_bus_TP_Rate_n3'), bus_TP_Rate)
        np.savez_compressed( os.path.join(folder,file_name+'_bus_precision_n3'), bus_precision)
        
        np.savez_compressed( os.path.join(folder,file_name+'_line_Accuracy_n3'), line_Accuracy)
        np.savez_compressed( os.path.join(folder,file_name+'_line_TP_Rate_n3'), line_TP_Rate)
        np.savez_compressed( os.path.join(folder,file_name+'_line_precision_n3'), line_precision)
        
        print('.................N-3..........................................')
        print_res(bus_MAPE3, bus_AE3, bus_L23, line_MAPE3, line_AE3,  line_L23, k, run, case, n_collocation, folder, file_name)    
        
        bus_MAPE3 = np.squeeze(np.array(bus_MAPE3))
        bus_AE3 = np.squeeze(np.array(bus_AE3))
        bus_L23 = np.squeeze(np.array(bus_L23))
        line_MAPE3 = np.squeeze(np.array(line_MAPE3))
        line_AE3 = np.squeeze(np.array(line_AE3))
        line_L23  = np.squeeze(np.array(line_L23))  
        
        writer = csv.writer(res_file)
        row3 = (
            run,
            n_collocation,  physics, k,test_n3_time,
            np.mean(bus_L23), np.var(bus_L23),
            np.mean(bus_AE3), np.var(bus_AE3),
            np.mean(bus_MAPE3), np.var(bus_MAPE3),
            np.mean(line_L23), np.var(line_L23),
            np.mean(line_AE3), np.var(line_AE3),
            np.mean(line_MAPE3), np.var(line_MAPE3),
            np.nanmean(bus_Accuracy), 
            np.nanmean(bus_precision, axis=(0)),
            np.nanmean(line_Accuracy), 
            np.nanmean(line_precision, axis=(0)),
            )
        writer.writerow(row3)
    

def print_res(bus_MAPE, bus_AE, bus_L2, line_MAPE, line_AE,  line_L2, k, run, case, n_collocation, folder, file_name):
        
        bus_MAPE = np.squeeze(np.array(bus_MAPE))
        bus_AE = np.squeeze(np.array(bus_AE))
        bus_L2 = np.squeeze(np.array(bus_L2))
        line_MAPE = np.squeeze(np.array(line_MAPE))
        line_AE = np.squeeze(np.array(line_AE))
        line_L2  = np.squeeze(np.array(line_L2))
        
        np.savez_compressed( os.path.join(folder,file_name +'bus_test'+str(k)+'_AE'), bus_AE)
        np.savez_compressed( os.path.join(folder,file_name+'bus_test'+str(k)+'_MAPE'), bus_MAPE)
        np.savez_compressed( os.path.join(folder,file_name+'bus_test'+str(k)+'_L2'), bus_L2)
    
        np.savez_compressed( os.path.join(folder,file_name+'line_test'+str(k)+'_sAE'), line_AE)
        np.savez_compressed( os.path.join(folder,file_name+'line_test'+str(k)+'_sMAPE'), line_MAPE)
        np.savez_compressed( os.path.join(folder,file_name+'line_test'+str(k)+'_sL2'), line_L2)
    
     
        print(np.array(bus_MAPE).shape, np.array(bus_AE).shape)
       
        print('Nodal MAE:', np.mean(bus_AE, axis = (0,2)))
        #print('MAE:', np.mean(bus_AE))
        print('Std dev:', np.var(bus_AE, axis = (0,2)))
       
        print('Nodal MAPE:', np.mean(bus_MAPE, axis = (0,2)))
        #print('MAPE:', np.mean(bus_MAPE))
        #print('Std dev:', np.var(bus_MAPE, axis = (0,2)))
       
        print('Nodal MSE:', np.mean(bus_L2, axis = (0,2)))
        print('Std dev:', np.var(bus_L2, axis = (0,2)))
        #print('MSE:', np.mean(bus_L2))
        #print('Std dev:', np.var(bus_L2))
        
        print('Line MAE:', np.mean(line_AE))
        print('Std dev:', np.var(line_AE))
       
        print('Line MAPE:', np.mean(line_MAPE))
        ##print('Std dev:', np.var(line_MAPE))
       
        print('Line MSE:', np.mean(line_L2))
        print('Std dev:', np.var(line_L2))
         

def main(device, case, run, folder, res_file, n_simulation, n_collocation, ploss, physics):

    train_loader, test_loader, _, _, _, _, _ =   data.dataset_generator(typ=1, data_params=params.data_params)
    
    col_train_loader, col_test_loader, _, _, _, _, _ =   data.dataset_generator(typ = 4, data_params=params.data_params)
    
    y_mean_tr = 0
    y_std_tr = 0 
    y_mean_tst = 0
    y_std_tst = 0
     
    # Loss 
    tr_loss =[]
    tr_lossplt =[]
    tst_loss = []
    min_val_loss = torch.inf
    model = params.model
    
    file_name = str(run)+'_'+str(params.n_b) + '_' + str(int(n_collocation*100))+'_' + physics +'_' + str(ploss)+'_' + params.model 
    
    if model == 'GNN':
        n_model = models.GNN(params.args['n_layersGNN'], params.args['n_layersFC'], params.hops, params.layersGNN, params.layersFC, params.num_nodes, params.args['dropout'], params.sigma, params.args['readout']).to(device)
    
    if model == 'Guided':
        n_model = models.GD(params.args['n_layersGD'], params.args['n_layersFC'], params.layersGD, params.layersFC, params.n_b, params.args['dropout'], params.sigma, params.args['readout']).to(params.device)
    
    n_model = n_model.float()
    optimizer = torch.optim.Adam(n_model.parameters(),
                                  lr=params.args['lr'])
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    
    #training time
    train_start = process_time()
    for epoch in range(1, 1 + params.args['epochs']):
        train_loss=0.0
        loss = train(n_model, train_loader, col_train_loader, y_mean_tr, y_std_tr, optimizer, device, params.drop_edge, params.weighted, params.isploss, n_simulation, n_collocation, model, ploss, physics)
        train_loss+=loss
        tr_loss.append(train_loss)
                
        ## Validation
        eval_loss = 0.0
        if epoch % 10 == 0:
            val_loss, AE, L2 = validate(n_model, test_loader, y_mean_tst, y_std_tst, device, params.drop_edge, params.weighted,model)
            
            AE,  L2 = np.array(AE), np.array(L2)
              # early stopping
            if val_loss < min_val_loss:
                count = 0                
                
                min_val_loss = val_loss  
                v_model = n_model
                
                k = 1
                
                if n_collocation == 0:
                    torch.save(n_model.state_dict(), os.path.join(folder,file_name +'_'+'best_model.pth'))
                else:
                    torch.save(n_model.state_dict(), os.path.join(folder,file_name+'_'+'pinn_best_model.pth'))                    
                   
                best_epoch  = epoch
                best_loss = [train_loss, val_loss]
                #print(f" Epoch {epoch} | Train Loss {loss} | Validation Loss {val_loss}")
            else:
                count += 1
                if count > params.patience:
                    train_end = process_time()
                    train_time = train_end - train_start
                    print('......................................................................\n')
                    print(f"Early stop: Epoch {epoch} | Train Loss {loss} | Validation Loss {val_loss}")
                    print(f"Best model: Epoch {best_epoch} | Train Loss : Validation Loss {best_loss}")
                    
                    
                    val_loss, AE, MAPE, L2, S_AE,  S_MAPE,  S_L2, line_conf, bus_conf = validate_best(v_model, test_loader, y_mean_tst, y_std_tst, device, params.drop_edge, params.weighted,model, folder,file_name,k)
                    
                    AE, MAPE, L2, S_AE,  S_MAPE,  S_L2 = np.array(AE), np.array(MAPE), np.array(L2), np.array(S_AE),  np.array(S_MAPE),  np.array(S_L2)
                    
                    VAE = AE
                    
                    bus_MAPE = MAPE, 
                    bus_AE = AE, 
                    bus_L2 = L2 
                    line_MAPE = S_MAPE, 
                    line_AE = S_AE,  
                    line_L2 = S_L2

                    line_conf = np.array(line_conf)
                    line_conf = np.nan_to_num(line_conf, nan=0.0, posinf=0.0, neginf=0.0)
                    line_conf_sum = np.sum(line_conf, axis=(0))
                    
                    line_conf_sum = np.array(line_conf_sum)
                    line_Accuracy, line_TP_Rate, line_precision = get_conf_metrics(line_conf_sum)
                    
                        
                    bus_conf = np.array(bus_conf)    
                    bus_conf_sum = np.sum(bus_conf, axis=(0))
                    
                    
                    bus_conf_sum = np.array(bus_conf_sum)
                    bus_Accuracy, bus_TP_Rate, bus_precision = get_conf_metrics(bus_conf_sum)
                    
                    bus_TP_Rate[bus_TP_Rate == 0] = np.nan
                    bus_precision[bus_precision ==0] = np.nan                    
                    
                    print_res(bus_MAPE, bus_AE, bus_L2, line_MAPE, line_AE,  line_L2, k, run, case, n_collocation, folder, file_name)
                    
                    np.savez_compressed( os.path.join(folder,file_name+'_bus_Accuracy_n1'), bus_Accuracy)
                    np.savez_compressed( os.path.join(folder,file_name +'_bus_TP_Rate_n1'), bus_TP_Rate)
                    np.savez_compressed( os.path.join(folder,file_name+'_bus_precision_n1'), bus_precision)
                    
                    np.savez_compressed( os.path.join(folder,file_name+'_line_Accuracy_n1'), line_Accuracy)
                    np.savez_compressed( os.path.join(folder,file_name+'_line_TP_Rate_n1'), line_TP_Rate)
                    np.savez_compressed( os.path.join(folder,file_name+'_line_precision_n1'), line_precision)
                    
                    break
            
            eval_loss += val_loss
            tst_loss.append(eval_loss)
            tr_lossplt.append(train_loss)
            #print(f"Epoch {epoch} | Train Loss {loss} | Validation Loss {val_loss} ")
    
    with open(os.path.join(params.model_folder,'time_' + file_name +'.txt'), "w") as text_file:
        text_file.write('Validation MAE: ' + str(np.mean(VAE, axis = (0,2)))+'\n')
        text_file.write('Overall Validation MAE:'+str(np.mean(VAE))+'\n')
        text_file.write('number of parameters: %d \n'%(sum(temp.numel() for temp in n_model.parameters() if temp.requires_grad)))
        text_file.write("Train_time: %f \n" %train_time )
        
    
    bus_MAPE = np.squeeze(np.array(bus_MAPE))
    bus_AE = np.squeeze(np.array(bus_AE))
    bus_L2 = np.squeeze(np.array(bus_L2))
    line_MAPE = np.squeeze(np.array(line_MAPE))
    line_AE = np.squeeze(np.array(line_AE))
    line_L2  = np.squeeze(np.array(line_L2))
    
    writer = csv.writer(res_file)
    row1 = (
        run,
        n_collocation,  physics, k, train_time,
        np.mean(bus_L2), np.var(bus_L2),
        np.mean(bus_AE), np.var(bus_AE),
        np.mean(bus_MAPE), np.var(bus_MAPE),
        np.mean(line_L2), np.var(line_L2),
        np.mean(line_AE), np.var(line_AE),
        np.mean(line_MAPE), np.var(line_MAPE),
        np.nanmean(bus_Accuracy), 
        np.nanmean(bus_precision, axis=(0)),
        np.nanmean(line_Accuracy), 
        np.nanmean(line_precision, axis=(0)),
        
        )
    writer.writerow(row1)
    
        
    
    
