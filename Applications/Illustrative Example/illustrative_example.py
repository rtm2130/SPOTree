#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ILLUSTRATIVE EXAMPLE

Generates a two-road instance of the shortest paths problem. 
Runs SPO Tree (greedy) and CART on this dataset and compares their performance.
Produces plots visualizing predicted costs and normalized SPO loss incurred by SPOT and CART.
To run code below, put SPO_tree_greedy.py into the same folder as this script.
"""

import numpy as np
from SPO_tree_greedy import SPOTree
from decision_problem_solver import*
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()

plt.rcParams.update({'font.size': 12})
figsize = (5.2, 4.3)
np.random.seed(0)
dpi=450

#SIMULATED DATASET FUNCTIONS
def get_costs(X):
  X = X.reshape(-1)
  mat = np.zeros((len(X),4))
  for i in range(len(X)):
    mat[i,0] = (X[i] + 0.8)*5-2.1
    mat[i,1] = (5*X[i]+0.4)**2
  return(mat)
  
def gen_dataset(n):
  x = np.random.rand(n,1) #generate random features in [0,1]
  costs = get_costs(x)
  return(x,costs)

def get_step_func_rep(x, costs):
  change_inds = np.where(costs[1:]-costs[:-1] > 0)[0]
  x_change_points = (x[change_inds.tolist()]+x[(change_inds+1).tolist()])/2.0
  x_min = np.append(np.array(x[0]),x_change_points)
  x_max = np.append(x_change_points,np.array(x[-1]))
  change_inds = change_inds.tolist()
  change_inds.append(len(x)-1)
  y = costs[change_inds]
  return(y, x_min, x_max)

def get_decision_boundary(x, costs):
  tmp = costs[:,1] > costs[:,0]
  if not any(tmp) == True:
    return None
  return(min(x[tmp]))

def plot_costs(plot_x, true_costs, est_costs, color_est_costs, est_name, fname):
  true_costs_0 = true_costs[:,0]
  true_costs_1 = true_costs[:,1]
  est_costs_0 = est_costs[:,0]
  est_costs_1 = est_costs[:,1]
  fig,ax = plt.subplots(1, figsize=figsize)
  line_true_costs_0, = ax.plot(plot_x, true_costs_0, linewidth=2.0, color='grey', linestyle='-', label='Edge 1 Cost (True)')
  line_true_costs_1, = ax.plot(plot_x, true_costs_1, linewidth=2.0, color='grey', linestyle='--', label='Edge 2 Cost (True)')
  
  costs,xmin,xmax = get_step_func_rep(plot_x, est_costs_0)
  line_est_costs_0 = ax.hlines(costs, xmin, xmax, linewidth=2.0, color=color_est_costs, linestyle='-', label='Edge 1 Cost ('+est_name+')')
  costs,xmin,xmax = get_step_func_rep(plot_x, est_costs_1)
  line_est_costs_1 = ax.hlines(costs, xmin, xmax, linewidth=2.0, color=color_est_costs, linestyle='--', label='Edge 2 Cost ('+est_name+')')
  
  plt.xlabel('x')
  plt.ylabel('Edge Cost')  
  #plt.ylim(top=21)
  
  bdry = get_decision_boundary(plot_x, est_costs)
  line_est_bdry = ax.axvline(x=bdry, linewidth=1.5, color=color_est_costs, linestyle=':', label='Decision Boundary ('+est_name+')')
  #_,xmin,_ = get_step_func_rep(plot_x, est_costs_0)
  #xbds = xmin[1:]
  #for xbd in xbds:
  #  plt.axvline(x=xbd, linewidth=1.0, color='grey', linestyle=':')
  
  bdry = get_decision_boundary(plot_x, true_costs)
  line_true_bdry = ax.axvline(x=bdry, linewidth=1.5, color='grey', linestyle=':', label='Decision Boundary (True)')
  
  plt.legend(handles=[line_true_costs_0, line_true_costs_1, line_true_bdry,
                      line_est_costs_0, line_est_costs_1, line_est_bdry], loc='upper left')
  #plt.show()
  plt.savefig(fname, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
  plt.clf()


plot_x = np.linspace(0,1,num=1000).reshape(1000,1)
true_costs = get_costs(plot_x)

#SIMULATED DATA PARAMETERS 
n_train = 10000;
n_valid = 2000;
n_test = 5000;

#GENERATE TRAINING DATA
train_x, train_cost = gen_dataset(n_train)
#GENERATE VALIDATION SET DATA
valid_x, valid_cost = gen_dataset(n_valid)
#GENERATE TESTING DATA
test_x, test_cost = gen_dataset(n_test)

###################################################################
#FIT SPO Tree ALGORITHM
#SPO_weight_param: number between 0 and 1:
#Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
my_tree = SPOTree(max_depth = 1, min_weights_per_node = 20, quant_discret = 0.01, debias_splits=False, SPO_weight_param=1.0, SPO_full_error=True)
my_tree.fit(train_x,train_cost,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
#my_tree.traverse() #prints out the unpruned tree 

#PRUNE DECISION TREE USING VALIDATION SET
my_tree.prune(valid_x, valid_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
#my_tree.traverse() #prints out the pruned tree

#FIND TEST SET SPO LOSS
opt_decision = find_opt_decision(test_cost)['weights']
pred_decision = my_tree.est_decision(test_x)

incurred_test_cost = np.sum([np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))])
opt_test_cost = np.sum(np.sum(test_cost * opt_decision,axis=1))

#percent error:
print("SPO Tree: Test Set Normalized SPO Loss: ")
SPO_error = 100.0*(incurred_test_cost-opt_test_cost)/opt_test_cost
print(str(SPO_error)+" percent error")
est_costs = my_tree.est_cost(plot_x)
plot_costs(plot_x, true_costs, est_costs, 'blue', 'SPOT', 'casestudySPOTdpi'+str(dpi)+'.png')

###################################################################
#FIT MSE Tree ALGORITHM
MSE_tree_depths = [1,2,3,4,5]
MSE_tree_depths_errors = np.zeros(len(MSE_tree_depths))
for max_depth_ind,max_depth in enumerate(MSE_tree_depths):
  #SPO_weight_param: number between 0 and 1:
  #Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
  my_tree = SPOTree(max_depth = max_depth, min_weights_per_node = 20, quant_discret = 0.01, debias_splits=False, SPO_weight_param=0.0, SPO_full_error=True)
  my_tree.fit(train_x,train_cost,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
  #my_tree.traverse() #prints out the unpruned tree 
  
  #PRUNE DECISION TREE USING VALIDATION SET
  my_tree.prune(valid_x, valid_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
  my_tree.traverse() #prints out the pruned tree
  
  #FIND TEST SET SPO LOSS
  opt_decision = find_opt_decision(test_cost)['weights']
  pred_decision = my_tree.est_decision(test_x)
  
  incurred_test_cost = np.sum([np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))])
  opt_test_cost = np.sum(np.sum(test_cost * opt_decision,axis=1))
  
  #percent error:
  print("MSE Tree Depth "+str(max_depth)+": Test Set Normalized SPO Loss: ")
  MSE_tree_depths_errors[max_depth_ind] = 100.0*(incurred_test_cost-opt_test_cost)/opt_test_cost
  print(str(MSE_tree_depths_errors[max_depth_ind])+" percent error")
  est_costs = my_tree.est_cost(plot_x)
  plot_costs(plot_x, true_costs, est_costs, 'orange', 'CART', 'casestudyCART'+str(max_depth)+'dpi'+str(dpi)+'.png')

SPO_tree_depths_errors = [SPO_error]*len(MSE_tree_depths)
fig,ax = plt.subplots(1, figsize=figsize)
ax.plot(MSE_tree_depths, MSE_tree_depths_errors, linewidth=2.0, color='orange', label='CART')
ax.plot(MSE_tree_depths, SPO_tree_depths_errors, linewidth=2.0, color='blue', label='SPOT')
plt.xlabel("Training Depth")
plt.ylabel("Norm. Extra Travel Time (%)")
plt.legend(loc='upper right')
plt.xticks(MSE_tree_depths)
#plt.show()
plt.savefig('casestudyCARTerrorsdpi'+str(dpi)+'.png', format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.clf()
