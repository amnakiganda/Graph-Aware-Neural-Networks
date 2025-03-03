

#Creating coloocation data


import copy
import time
import itertools
import scipy.io
import warnings
import os
import numpy as np
import pandas as pd
from numpy import random
import math
import cmath
import numba

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import networkx as nx
import scipy.sparse as sp
from scipy.stats import qmc

import torch

import matplotlib.pyplot as plt
from time import process_time 

np.set_printoptions(precision=9)
np.set_printoptions(suppress=True)

warnings.simplefilter(action='ignore', category=FutureWarning)

import params


class Collocation_Data_Generation:
  def __init__(self, data_args, is_rectangular):

      self.data_args = data_args
      self.is_rectangular = is_rectangular
      self.nw = nw

      self.n_b = (data_args['test_nw'].bus).shape[0]  # Number of buses
      self.n_l = (data_args['test_nw'].line).shape[0]  # Number of lines
      
      self.n_t = (data_args['test_nw'].trafo).shape[0] # Number of transformers
      self.df_lines = data_args['test_nw'].line
      self.df_trans = data_args['test_nw'].trafo

      self.tst_net = self.data_args['test_nw']
      data_pts = self.data_args['n_pts']
      self.data_pts = data_pts

      self.vb = (self.tst_net.bus[['vn_kv']]).to_numpy()

      self.pload_var, self.qload_var = self.define_data_pts(
          self.tst_net, data_pts)
      self.pgen_var = self.define_gen_data_pts(self.tst_net, data_pts)
      

      l_df = self.tst_net.line
      trans_df = self.tst_net.trafo
      from_bus = (l_df.loc[l_df['in_service'] == True, 'from_bus']).tolist()
      to_bus = (l_df.loc[l_df['in_service'] == True, 'to_bus']).tolist()
      hv_bus = (trans_df.loc[trans_df['in_service']
                == True, 'hv_bus']).tolist()
      lv_bus = (trans_df.loc[trans_df['in_service']
                == True, 'lv_bus']).tolist()
      line_indices = [from_bus + hv_bus, to_bus + lv_bus]  # Undirected
      self.line_indices = np.array(line_indices)
      self.line_indices = self.line_indices.T

      imax = np.array(
          (l_df.loc[l_df['in_service'] == True, 'max_i_ka']).tolist())
      parallel = np.array(
          (l_df.loc[l_df['in_service'] == True, 'parallel']).tolist())
      df = np.array((l_df.loc[l_df['in_service'] == True, 'df']).tolist())

      self.imax = imax*parallel*df

      self.folder = str(self.n_b) + 'bus_data'
      if not os.path.exists(self.folder):
         os.makedirs(self.folder)
      
      #save data points
      np.savez_compressed(os.path.join(self.folder, str(self.n_b)+'pload_col'), self.pload_var)
      np.savez_compressed(os.path.join(self.folder, str(self.n_b)+'qload_col'), self.qload_var)
      np.savez_compressed(os.path.join(self.folder, str(self.n_b)+'pgen_col'), self.pgen_var)
      
            
  def get_dataset_N1_lines(self):
      lines_n1 = (np.load(os.path.join(self.folder,str(self.n_b)+'lines_n_1.npz')))["arr_0"]
      
      return lines_n1

  def get_data(self):
      tst_net = copy.deepcopy(self.tst_net)  

      lines_n_1 = self.get_dataset_N1_lines()
      
      start1 = process_time()  
      self.data_gen_N0(tst_net, folder=self.folder)

      start2 = process_time()   
      self.data_gen_N1(tst_net, lines_n_1, folder=self.folder)
            

  #Scaling the data
  def normalize_data(x, is_scale=False):

      if is_scale:
          x_mean = torch.mean(x, 0)
          x_std = torch.std(x, 0)
          x_norm = (x - torch.mean(x, 0))/(torch.std(x, 0) + 1e-8)
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
      x_mean, _ = torch.min(x, 0)
      x_std, _ = torch.max(x, 0)
      x_norm = (x - x_mean)/(x_std - x_mean)
      x_norm[torch.isnan(x_norm)] = 0
      x_norm[torch.isinf(x_norm)] = 0

      return x_norm, x_mean, x_std,

  def descale_data(x_norm, x_mean, x_std):
      x = x_norm*(x_std-x_mean) + x_mean
      return x

  def lst_reshape(self, lst):
      array = []
      lst = [x for x in lst if x.size > 0]
      #array = np.asarray(lst, dtype = object)

      for x in lst:
          if x.size > 0:
              if len(array) == 0:
                  array = x
              else:
                  array = np.concatenate((array, x), axis=0)
      return array

  def polar2z(self, r, theta):
    return r * np.exp(1j * theta)

  def cart2pol(self, x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

  def pol2cart(self, rho, theta):
     x = rho * np.cos(theta)
     y = rho * np.sin(theta)
     return x + 1j*y

  #Data points
  #Load variation
  def define_data_pts(self, tst_net, n_pts):
      n_load = (tst_net.load).shape[0]

      sampler = qmc.LatinHypercube(d=n_load)
      pload_var = sampler.random(n=n_pts)
      pload_var_scaled = qmc.scale(pload_var, 0.6, 1.2)

      # qload_var = sampler.random(n = n_pts)
      # qload_var_scaled = qmc.scale(pload_var, 0.6, 1.2)
      # Retaining simililar power factor for the loads
      qload_var_scaled = pload_var_scaled

      return pload_var_scaled, qload_var_scaled

  #Data points
  #gen variation
  def define_gen_data_pts(self, tst_net, n_pts):
      n_gen = (tst_net.gen).shape[0]
      sampler = qmc.LatinHypercube(d=n_gen)
      pgen_var = sampler.random(n=n_pts)
      pgen_var_scaled = qmc.scale(pgen_var, 0.8, 1.0)

      return pgen_var_scaled

  def get_adj(self, tst_net):

      n_b = (tst_net.bus).shape[0]
      n_l = (tst_net.line).shape[0]
      v_base = (tst_net.bus.vn_kv)[0]
      base_mva = tst_net.sn_mva
      v_b = (tst_net.bus[['vn_kv']]).to_numpy()
      line_fr_to = (tst_net.line[['from_bus', 'to_bus']]).to_numpy()

      #z_base = ((v_base*1e3)**2)/(base_mva*1e6)

      l_df = tst_net.line
      from_bus = (l_df.loc[l_df['in_service'] == True, 'from_bus']).tolist()
      to_bus = (l_df.loc[l_df['in_service'] == True, 'to_bus']).tolist()
      length = (l_df.loc[l_df['in_service'] == True, 'length_km']).to_numpy()
      
      trans_df = tst_net.trafo
      hv_bus = (trans_df.loc[trans_df['in_service']
                 == True, 'hv_bus']).tolist()
      lv_bus = (trans_df.loc[trans_df['in_service']
                 == True, 'lv_bus']).tolist()
           
      edge_indices = [from_bus + to_bus + hv_bus + lv_bus,
                      to_bus + from_bus + lv_bus + hv_bus]  # Undirected
      edge_indices = np.array(edge_indices)
      

      data = np.ones(edge_indices.shape[1])
      S = sp.coo_matrix(
          (data, (edge_indices[0, :], edge_indices[1, :])), shape=(self.n_b, self.n_b)).toarray()
      
      spl = np.split(edge_indices, 2, 1)

      rng = np.random.default_rng()

      perm = rng.permutation(np.array(edge_indices), axis=1)
      preserve = int(2 * n_l * 0.8)
      perm = np.split(perm, [preserve, ], 1)
      perm = perm[0]

      S_red = sp.coo_matrix(
          (np.ones(perm.shape[1]), (perm[0, :], perm[1, :])), shape=(n_b, n_b)).toarray()

      return S, S_red

  def add_edge_features(self, tst_net, l_drop):

      n_b = (tst_net.bus).shape[0]
      n_line = (tst_net.line).shape[0]
      l_df = tst_net.line
      from_bus = (l_df['from_bus']).tolist()
      to_bus = (l_df['to_bus']).tolist()
      parallel = (l_df.loc[l_df['in_service'] == True, 'parallel']).tolist()

      v_b = (tst_net.bus[['vn_kv']]).to_numpy()
      line_fr_to = (tst_net.line[['from_bus', 'to_bus']]).to_numpy()
      nl = line_fr_to.shape[0]
      ntrafo = (tst_net.trafo).shape[0]

      z_l = ((tst_net.line['r_ohm_per_km']).to_numpy() + ((tst_net.line['x_ohm_per_km']
                                                           ).to_numpy())*1j)*(tst_net.line['length_km']/tst_net.line['parallel']).to_numpy()
      y_s = (((tst_net.line['g_us_per_km']).to_numpy())*1e-6 + ((2*math.pi*60)*((tst_net.line['c_nf_per_km']
                                                                                 ).to_numpy())*1e-9*1j*1))*(tst_net.line['length_km'])*(tst_net.line['parallel']).to_numpy()

      z_line = np.zeros(nl, dtype='complex_')
      y_line = np.zeros(nl, dtype='complex_')
      y_lshnt = np.zeros(nl, dtype='complex_')
      Y_bus = np.zeros([n_b, n_b], dtype='complex_')
      ln_imp = np.zeros([n_b, n_b], dtype='complex_')
      ln_shnt = np.zeros([n_b, n_b], dtype='complex_')

      for l in range(nl):
          if l_df['in_service'][l] == True:
              i = line_fr_to[l, 0]
              j = line_fr_to[l, 1]
              z_b = (((v_b[i])**2)/tst_net.sn_mva)
              z_line[l] = z_l[l]/z_b[0]
              y_line[l] = 1/z_line[l]
              y_lshnt[l] = y_s[l]*z_b[0]

      for l in range(n_line):
          i = (tst_net.line['from_bus'][l])
          j = (tst_net.line['to_bus'][l])
          Y_bus[i, i] = Y_bus[i, i] + y_line[l] + 0.5*y_lshnt[l]
          Y_bus[j, j] = Y_bus[j, j] + y_line[l] + 0.5*y_lshnt[l]
          Y_bus[i, j] = Y_bus[i, j] - y_line[l]
          Y_bus[j, i] = Y_bus[j, i] - y_line[l]

          ln_imp[i, j] = ln_imp[j, i] = y_line[l]
          ln_shnt[i, j] = ln_shnt[j, i] = 0.5*y_lshnt[l]

      trans_df = tst_net.trafo
      hv_bus = (trans_df.loc[trans_df['in_service']
                == True, 'hv_bus']).tolist()
      lv_bus = (trans_df.loc[trans_df['in_service']
                == True, 'lv_bus']).tolist()
      vn_lv_kv = (trans_df.loc[trans_df['in_service']
                  == True, 'vn_lv_kv']).to_numpy()
      vn_hv_kv = (trans_df.loc[trans_df['in_service']
                  == True, 'vn_hv_kv']).to_numpy()
      tsn_mva = (trans_df.loc[trans_df['in_service']
                 == True, 'sn_mva']).to_numpy()
      vk_percent = (trans_df.loc[trans_df['in_service']
                    == True, 'vk_percent']).to_numpy()
      vkr_percent = (
          trans_df.loc[trans_df['in_service'] == True, 'vkr_percent']).to_numpy()
      i0_percent = (trans_df.loc[trans_df['in_service']
                    == True, 'i0_percent']).to_numpy()
      pfe_kw = (trans_df.loc[trans_df['in_service']
                == True, 'pfe_kw']).to_numpy()
      tra_hv_lv = (tst_net.trafo[['hv_bus', 'lv_bus']]).to_numpy()

      z_trf = np.zeros(vn_lv_kv.shape[0], dtype='complex_')
      ym_trf = np.zeros(vn_lv_kv.shape[0], dtype='complex_')
      y_trf = np.zeros(vn_lv_kv.shape[0], dtype='complex_')

      for l in range(vn_lv_kv.shape[0]):
          i = tra_hv_lv[l, 0]
          j = tra_hv_lv[l, 1]

          if trans_df['in_service'][l] == True:
              if trans_df['tap_step_percent'][l] > 0:
                  n = 1 + (trans_df['tap_pos'][l] - trans_df['tap_neutral']
                           [l])*trans_df['tap_step_percent'][l]/100
              else:
                  n = vn_hv_kv[l] * (tst_net.bus['vn_kv'][j]) / \
                      (vn_lv_kv[l] * (tst_net.bus['vn_kv'][i]))

              #z_tr = (vk_percent[l]*tst_net.sn_mva/(tsn_mva[l]*100))
              z_tr = (vk_percent[l]*tst_net.sn_mva/(tsn_mva[l]*100))
              r_tr = (vkr_percent[l]*tst_net.sn_mva/(tsn_mva[l]*100))
              x_tr = np.sqrt(z_tr**2 - r_tr**2)
              z_trans = (r_tr + 1j*x_tr)

              ym = i0_percent[l]/100
              gm = pfe_kw[l]*tst_net.sn_mva/(tsn_mva[l]*tsn_mva[l]*1000)
              bm = np.sqrt(ym**2 - gm**2)
              y_m_trans = gm - 1j*bm

              #Rated values
              zN = ((tst_net.bus['vn_kv'][j])**2)/tst_net.sn_mva
              z_ref = ((vn_lv_kv[l])**2)*tst_net.sn_mva/tsn_mva[l]

              z_trf[l] = z_trans
              y_trf[l] = (1/z_trf[l])  # *(zN/z_ref)
              ym_trf[l] = (y_m_trans)  # *(zN/z_ref)

              Y_bus[i, i] = Y_bus[i, i] + (y_trf[l] + 0.5*ym_trf[l])/n**2
              Y_bus[j, j] = Y_bus[j, j] + (y_trf[l] + 0.5*ym_trf[l])
              Y_bus[i, j] = Y_bus[i, j] - y_trf[l]/n
              Y_bus[j, i] = Y_bus[j, i] - y_trf[l]/n

              ln_imp[i, j] = ln_imp[j, i] = y_trf[l]
              ln_shnt[i, j] = ln_shnt[j, i] = 0.5*ym_trf[l]

      bus_shunt = np.zeros(n_b, dtype='complex_')
      for n in range(tst_net.shunt.shape[0]):

          if tst_net.shunt['in_service'][n] == True:

              i = tst_net.shunt['bus'][n]
              y_shunt = ((tst_net.shunt['p_mw'][n] + 1j*tst_net.shunt['q_mvar']
                         [n])*tst_net.shunt['step'][n])/tst_net.sn_mva
              bus_shunt[i] = y_shunt

              Y_bus[i,i] = Y_bus[i,i] - y_shunt
        

      for l in range(nl):
          i = line_fr_to[l, 0]
          j = line_fr_to[l, 1]
          z_b = (((v_b[i])**2)/tst_net.sn_mva)

      
      edge_indices = [from_bus + to_bus + hv_bus + lv_bus,
                      to_bus + from_bus + lv_bus + hv_bus]  # Undirected
      edge_indices = np.array(edge_indices)

      
      return Y_bus, ln_imp, ln_shnt

  def normalize_adj(self, adj):
      # obtain eigenvalues
      eigenvalues, _ = np.linalg.eig(adj)

      if np.max(np.abs(eigenvalues)) > 0:
          # normalize by eigenvalue with largest absolute value
          norm = adj / np.max(np.abs(eigenvalues))
      else:
          norm = adj * 0

      return norm

  def get_load(self, net):
      l_df = net.load
      load_bus = net.load['bus']
      l_df['bus'] = load_bus
      n_b = (net.bus).shape[0]
      pload = np.zeros((n_b, 1))
      qload = np.zeros((n_b, 1))
      for b in range(n_b):
          if b in l_df['bus'].unique():
              pload[b, 0] = l_df.loc[l_df['bus'] == b, 'p_mw'].item()
              qload[b, 0] = l_df.loc[l_df['bus'] == b, 'q_mvar'].item()
      return pload, qload

  def get_gen(self, net):
      gen_bus = net.gen['bus']
      g_df = net.gen
      g_df['bus'] = gen_bus
      n_b = (net.bus).shape[0]
      pgen = np.zeros((n_b, 1))
      #qgen = np.zeros((n_b, 1))
      #vgen = np.zeros((n_b, 1))
      for b in range(n_b):
          if b in g_df['bus'].unique():
              pgen[b, 0] = g_df.loc[g_df['bus'] == b, 'p_mw'].item()
              #qgen[b, 0] = g_df.loc[g_df['bus'] == b, 'q_mvar'].item()
              #vgen[b, 0] = g_df.loc[g_df['bus'] == b, 'vm_pu'].item()
      
      return pgen#, qgen, vgen

  def getlinestrafo_n1(self, tst_net):

      G, _, _ = self.get_adj(tst_net)

      d = np.sum(G, axis=1)

      _deg1_nodes = []
      _degn_nodes = []

      for n in range(len(d)):
          if d[n] == 1:
              _deg1_nodes.append(n)
          else:
              _degn_nodes.append(n)
      max_deg = max(d)

      df_lines = tst_net.line
      df_trans = tst_net.trafo
      lines_n_0 = []
      trans_n_0 = []
      for n in _deg1_nodes:
          l = df_lines.index[(df_lines['from_bus'] == n) |
                             (df_lines['to_bus'] == n)].tolist()
          #lines_n_1.append(l)
          lines_n_0 = lines_n_0 + l

      for n in _deg1_nodes:
          l = df_trans.index[(df_trans['hv_bus'] == n) |
                             (df_trans['lv_bus'] == n)].tolist()
          #lines_n_1.append(l)
          trans_n_0 = trans_n_0 + l
      nl = len(df_lines)

      lines_n_0 = np.unique(np.squeeze(np.array(lines_n_0)))
      trans_n_0 = np.unique(np.squeeze(np.array(trans_n_0)))

      l = np.array(list(range(0, len(df_lines))))
      if len(lines_n_0) == 0:
          lines_n_1 = l
      else:
          lines_n_1 = np.delete(l, lines_n_0)

      t = np.array(list(range(0, len(df_trans))))
      if len(trans_n_0) == 0:
          trans_n_1 = t
      else:
          trans_n_1 = np.delete(t, trans_n_0)

      return lines_n_1, trans_n_1, lines_n_0, trans_n_0
  

  def get_data_pts(self, tst_net, c1, c2, c3, l, l2, l3):
      v_pf, v_ang, p_inj, q_inj, pload, qload, pgen, qgen, vgen = [], [], [], [], [], [], [], [], []
      pl_fro_to, ql_fro_to, ln_load = [], [], []
      A, A_weighted, A_nom, A_weighted_nom = [], [], [], []
      A_red = []
      Y_bus,  Y_line, Y_shunt = [], [], []

      n_smp = 0
      net_v = copy.deepcopy(self.tst_net)  # (self.tst_net)
      
      pload_var = self.pload_var 
      qload_var = self.qload_var  
      pgen_var = self.pgen_var

      l_off = np.array([])


      if c1:
          net_v.line.loc[l, 'in_service'] = False
          l_off = np.array([l])
          
          pload_var = self.pload_var 
          qload_var = self.qload_var  
          pgen_var = self.pgen_var
                
          
      for pl, ql, pg in zip(pload_var, qload_var,  pgen_var):

          error = 0

          # set to nan to simplify phase shift
          net_v.trafo.tap_step_percent = float('nan')

          net_v.load['p_mw'] = (net_v.load['p_mw'])*pl
          net_v.load['q_mvar'] = (net_v.load['q_mvar'])*ql

          net_v.gen['p_mw'] = (net_v.gen['p_mw'])*pg
          #net_v.gen['p_mw'] = (net_v.gen['vm_pu'])*pg

          line_fr_to = (net_v.line[['from_bus', 'to_bus']]).to_numpy()

          
          if error == 0:
              isl_v = (net_v.res_bus.vm_pu).isnull().values.any()
              isl_ang = (net_v.res_bus.va_degree).isnull().values.any()

              if isl_v == False or isl_ang == False:
                
                  y_bus, ln_imp, ln_shnt = self.add_edge_features(
                      net_v, l_off)                  
                  Y_line.append(ln_imp)
                  Y_shunt.append(ln_shnt)
                  Y_bus.append(y_bus)

                  pload.append(self.get_load(net_v)[0])
                  qload.append(self.get_load(net_v)[1])
                  pgen.append(self.get_gen(net_v))
                  #qgen.append(self.get_gen(net_v)[1])

                  A.append(self.get_adj(net_v)[0])
                  A_weighted.append(np.abs(ln_imp + ln_shnt))
                  A_nom.append(self.normalize_adj(self.get_adj(net_v)[0]))
                  A_weighted_nom.append(
                      self.normalize_adj(np.abs(ln_imp + ln_shnt)))
                  A_red.append(self.get_adj(net_v)[1])
                  
              else:
                  continue

          n_smp = n_smp + 1
          error = 0
          #net_v = tst_net

          net_v.load['p_mw'] = (net_v.load['p_mw'])/pl
          net_v.load['q_mvar'] = (net_v.load['q_mvar'])/ql

          net_v.gen['p_mw'] = (net_v.gen['p_mw'])/pg

      
      base_mva = net_v.sn_mva
      Y_line = (np.asarray(Y_line, dtype=np.complex64))
      Y_shunt = (np.array(Y_shunt, dtype=np.complex64))

      pload = np.squeeze(np.array(pload)/base_mva)
      qload = np.squeeze(np.array(qload)/base_mva)
      pgen = np.squeeze(np.array(pgen)/base_mva)
      #qgen = np.squeeze(np.array(qgen)/base_mva)
      
      
      A = (np.array(A, dtype=np.float16))
      A_weighted = (np.array(A_weighted, dtype=np.float32))
      A_nom = (np.array(A_nom, dtype=np.float16))
      A_weighted_nom = (np.array(A_weighted_nom, dtype=np.float32))
      A_red = (np.array(A_red, dtype=np.float16))
      Y_bus = (np.array(Y_bus, dtype=np.complex64))   
             

      return pload, qload, pgen, Y_line, Y_shunt,\
          A, A_weighted, A_nom, A_weighted_nom, A_red, Y_bus

  def data_gen_N0(self, tst_nwk, folder):

      c = False
      c2 = False
      c3 = False
      l = '_'
      l2 = '_'
      l3 = '_'

      pload_0, qload_0, pgen_0, Y_line, Y_shunt,\
          A_0, A_weighted_0, A_nom_0, A_weighted_nom_0, A_red_0, Y_bus_0\
          = self.get_data_pts(tst_nwk, c, c2, c3, l, l2, l3)
          
      node_feat = np.stack((pload_0, qload_0, pgen_0), axis=-1)
      edge_feat = np.stack((A_0, A_weighted_0, A_nom_0, A_weighted_nom_0, A_red_0, Y_bus_0), axis=-1)
      line_feat = np.stack((Y_line, Y_shunt), axis=-1)

      np.savez_compressed(os.path.join(folder, 'node_col_pts'), node_feat)
      np.savez_compressed(os.path.join(folder, 'edge_col_pts'), edge_feat)
      np.savez_compressed(os.path.join(folder, 'line_col_pts'), line_feat)

  def data_gen_N1(self, tst_nwk, lines_n_1, folder):

      #topologies with zero unsupplied busses
      lines = np.array(lines_n_1)

      if lines.shape[0] > 0:

          split = math.ceil(len(lines)/50)

          lines_splt = np.array_split(lines, split)

          for b in range(split):

              #N-1 data
              cv_pf, cv_ang, cp_inj, cq_inj,  cpload, cqload = [], [], [], [], [], []
              cpl_fro_to, cql_fro_to = [], []
              cA, cA_weighted, cA_nom, cA_weighted_nom = [], [], [], []
              cpgen, cqgen = [], []
              cA_red = []
              cY_bus, cln_load = [], []
              Yline, Yshunt = [], []

              lines_batch = lines_splt[b]

              for l in lines_batch:
                  l = l[0]

                  c1 = True
                  c2 = False
                  c3 = False

                  pload, qload, pgen, Y_line, Y_shunt,\
                      A, A_weighted, A_nom, A_weighted_nom, A_red, Y_bus\
                      = self.get_data_pts(tst_nwk, c1, c2, c3, l, l2='_', l3='_')
                  
                  Yline.append(Y_line)
                  Yshunt.append(Y_shunt)
                  cpload.append(pload)
                  cqload.append(qload)
                  cA.append(A)
                  cA_weighted.append(A_weighted)
                  cA_nom.append(A_nom)
                  cA_weighted_nom.append(A_weighted_nom)
                  cpgen.append(pgen)
                  cA_red.append(A_red)
                  cY_bus.append(Y_bus)

              # Bus features
              
              cpload = self.lst_reshape(cpload)
              cqload = self.lst_reshape(cqload)
              cpgen = self.lst_reshape(cpgen)

              #line features
              Yline = self.lst_reshape(Yline)
              Yshunt = self.lst_reshape(Yshunt)

              #Adjacency
              cA = self.lst_reshape(cA)
              cA_weighted = self.lst_reshape(cA_weighted)
              cA_nom = self.lst_reshape(cA_nom)
              cA_weighted_nom = self.lst_reshape(cA_weighted_nom)
              cA_red = self.lst_reshape(cA_red)
              cY_bus = self.lst_reshape(cY_bus)
              

              c_node_feat = np.stack(
                  (cpload, cqload, cpgen), axis=-1)
              c_edge_feat = np.stack(
                  (cA, cA_weighted, cA_nom, cA_weighted_nom, cA_red, cY_bus), axis=-1)
              c_line_feat = np.stack(
                  (Yline, Yshunt), axis=-1)

              print(c_node_feat.shape, c_line_feat.shape)

              np.savez_compressed(os.path.join(
                  folder, str(b)+'_cnode_col_pts'), c_node_feat)
              np.savez_compressed(os.path.join(
                  folder, str(b)+'_cedge_col_pts'), c_edge_feat)
              np.savez_compressed(os.path.join(
                  folder, str(b)+'_cline_col_pts'), c_line_feat)

 
#test_ntw = [pn.case6ww(), pn.case24_ieee_rts(), pn.case57(), pn.case118()]
test_ntw = [pn.case24_ieee_rts()]#Add test network

for nw in test_ntw:
    data_args = {
        'test_nw': nw,  # Test network
        'n_pts': 1000  # data points per topology
    }

    is_rectangular = True

    data = Collocation_Data_Generation(data_args, is_rectangular)
    data.get_data()

