'''
Runs SPOT (MILP) algorithm on Yahoo News dataset
Outputs decision costs for each test-set instance as pickle file
Note on notation: the paper uses r to denote the binary variables which map training observations to leaves. This code uses z rather than r.
Takes multiple input arguments:
  (1) H: training depth of tree, e.g. "5"
  (2) N_min: min. number of (weighted) observations per leaf, e.g. "100"
  (3) train_x_precision: contextual features x are rounded to train_x_precision before fitting MILP (e.g., 2 = two decimal places)
    higher values of train_x_precision will be more precise but take more computational time
  (4) reg_set_str: sequence of regularization parameters to try (tuned using cross validation), e.g. "0.001-0.01-0.1"
    if "None", fits MILP using no regularizaiton and then prunes using CART pruning procedure (with SPO loss as pruning metric)
  (5) solver_time_limit: MILP solver is terminated after solver_time_limit seconds, returning best-found solution
  (6) decision_problem_seed: seed controlling generation of constraints in article recommendation problem (-1 = no constraints)
  (7) train_size: number of random obs. to extract from the training data. 
    Only useful in limiting the size of the training data (-1 = use full training data)
  (8) quant_discret: Used to fit the SPOT tree used in warm-starting the MILP. 
    Continuous variable split points in the tree is chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
Values of input arguments used in paper:
  (1) H: considered depths of 2, 4, 6
  (2) N_min: "10000"
  (3) train_x_precision: 3
  (4) reg_set_str: "None"
  (5) solver_time_limit: 43200
  (6) decision_problem_seed: ran 9 constraint instances corresponding to seeds of 10, 11, 12, ..., 18
  (7) train_size: -1
  (8) quant_discret: 0.05
'''

import time

import numpy as np
import pickle
from spo_opt_tree_news import*
from SPO2CART import SPO2CART
from SPO_tree_greedy import SPOTree
from decision_problem_solver import*

import sys
##############################################
#seed controlling random subset of training data used (if full training data is not being used)
select_train_seed = 0
########################################
#training parameters
#optimal tree params
H = int(sys.argv[1])#2 #H = max tree depth
N_min = int(sys.argv[2])#4 #N_min = minimum number of observations per leaf node
#higher values of train_x_precision will be more precise but take more computational time
#values >= 8 might cause numerical errors
train_x_precision = int(sys.argv[3])#2
reg_set_str = sys.argv[4]#"None"
if reg_set_str == "None":
  reg_set = None
else:
  reg_set = [int(k) for k in reg_set_str.split('-')]#None
#reg_set = [0.001] #if reg_set = None, uses CART to prune tree
solver_time_limit = int(sys.argv[5])
#ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
decision_problem_seed=int(sys.argv[6]) #if -1, use no constraints in decision problem
train_size=int(sys.argv[7]) #if you want to limit the size of the training data (-1 = no limit)
quant_discret=float(sys.argv[8]) 
########################################
#output filename for alg
fname_out = "SPOopt_news_costs_depth"+str(H)+"_minObs"+str(N_min)+"_prec"+str(train_x_precision)+"_regset"+reg_set_str+"_tLim"+str(solver_time_limit)+"_probSeed"+str(decision_problem_seed)+"_nTrain"+str(train_size)+"_qd"+str(quant_discret)+".pkl";
#############################################################################
#############################################################################
#############################################################################

#generate decision problem
#ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
num_constr = 5 #number of Aw <= b constraints
num_dec = 6 #number of decisions
if decision_problem_seed == -1:
  #no budget constraint case
  A_constr = np.zeros((num_constr,num_dec))
  b_constr = np.ones(num_constr)
  l_constr = np.zeros(num_dec)
  u_constr = np.ones(num_dec)
else:
  np.random.seed(decision_problem_seed)
  A_constr = np.random.exponential(scale=1.0, size=(num_constr,num_dec))
  b_constr = np.ones(num_constr)
  l_constr = np.zeros(num_dec)
  u_constr = np.ones(num_dec)
    
##############################################

thresh = "50"
valid_size = "50.0%"

train_x = np.load('filtered_train_userFeat_'+valid_size+'_'+thresh+'.npy')
valid_x = np.load('filtered_validation_userFeat_'+valid_size+'_'+thresh+'.npy')
test_x = np.load('filtered_test_userFeat_'+valid_size+'_'+thresh+'.npy')

#make negative to turn into minimization problem
train_cost = np.load('filtered_train_clickprob_'+valid_size+'_'+thresh+'.npy')*-1.0
valid_cost = np.load('filtered_validation_clickprob_'+valid_size+'_'+thresh+'.npy')*-1.0
test_cost = np.load('filtered_test_clickprob_'+valid_size+'_'+thresh+'.npy')*-1.0

train_weights = np.load('filtered_train_usernumobserv_'+valid_size+'_'+thresh+'.npy')
valid_weights = np.load('filtered_validation_usernumobserv_'+valid_size+'_'+thresh+'.npy')
test_weights = np.load('filtered_test_usernumobserv_'+valid_size+'_'+thresh+'.npy')

##############################################
#limit size of training data if specified
if train_size != -1 and train_size <= train_x.shape[0] and train_size <= valid_x.shape[0]:
  np.random.seed(select_train_seed)
  sel_inds = np.random.choice(range(train_x.shape[0]), size = train_size, replace=False)
  train_x = train_x[sel_inds]
  train_cost = train_cost[sel_inds]
  train_weights = train_weights[sel_inds]
  sel_inds = np.random.choice(range(valid_x.shape[0]), size = train_size, replace=False)
  valid_x = valid_x[sel_inds]
  valid_cost = valid_cost[sel_inds]
  valid_weights = valid_weights[sel_inds]

#############################################################################
#############################################################################
#############################################################################

start = time.time()

#FIT SPO OPTIMAL TREE
if reg_set is None:
  
  #FIT SPO GREEDY TREE AS INITIAL SOLUTION
  def truncate_train_x(train_x, train_x_precision):
    return(np.around(train_x, decimals=train_x_precision))
  train_x_truncated = truncate_train_x(train_x, train_x_precision)
  #SPO_weight_param: number between 0 and 1:
  #Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
  my_tree = SPOTree(max_depth = H, min_weights_per_node = N_min, quant_discret = quant_discret, debias_splits=False, SPO_weight_param=1.0, SPO_full_error=True)
  my_tree.fit(train_x_truncated,train_cost,weights=train_weights,verbose=False,feats_continuous=True,
              A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr);

  #PRUNE SPO GREEDY TREE USING TRAINING SET (TO GET RID OF REDUNDANT LEAVES)   
  my_tree.prune(train_x_truncated, train_cost, weights_val=train_weights, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
  my_tree.traverse()
  spo_greedy_a, spo_greedy_b, spo_greedy_z = my_tree.get_tree_encoding(x_train=train_x_truncated)
  
  #(OPTIONAL) PRUNE SPO GREEDY TREE USING VALIDATION SET
#      my_tree.prune(valid_x, valid_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
#      spo_greedy_a, spo_greedy_b, spo_greedy_z = my_tree.get_tree_encoding(x_train=train_x_truncated)
#      alpha = my_tree.get_pruning_alpha()
  
  #FIT SPO OPTIMAL TREE USING FOUND INITIAL SOLUTION
  reg_param = 1e-5 #introduce very small amount of regularization to ensure leaves with zero predictive power are aggregated
  spo_dt_a, spo_dt_b, spo_dt_w, spo_dt_y, spo_dt_z, spo_dt_l, spo_dt_d = spo_opt_tree(train_cost,train_x,train_x_precision,reg_param, N_min, H,
                                                                                      weights=train_weights,
                                                                                      a_start=spo_greedy_a, z_start=spo_greedy_z,
                                                                                      Presolve=2, Seed=0, TimeLimit=solver_time_limit,
                                                                                      returnAllOptvars=True,
                                                                                      A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
  
  end = time.time()    
  print "Elapsed time: " + str(end-start)
  
  #(IF NOT USING POSTPRUNING) FIND TEST SET COST    
#      path = decision_path(test_x,spo_dt_a,spo_dt_b)
#      costs_deg[deg][trial_num] = apply_leaf_decision(test_cost,path,spo_dt_w, subtract_optimal=False, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
  
  #PRUNE MILP TREE USING CART PRUNING METHOD ON VALIDATION SET
  spo2cart = SPO2CART(spo_dt_a, spo_dt_b)
  spo2cart.fit(train_x,train_cost,train_x_precision,weights=train_weights,verbose=False,feats_continuous=True,
               A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
  spo2cart.traverse() #prints out the unpruned tree found by MILP
  spo2cart.prune(valid_x, valid_cost, weights_val=valid_weights, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
  spo2cart.traverse() #prints out the pruned tree 
  #FIND TEST SET COST
  pred_decision = spo2cart.est_decision(test_x)
  costs = [np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
  
else:
  spo_dt_a, spo_dt_b, spo_dt_w, _, best_alpha = spo_opt_tunealpha(train_x,train_cost,train_weights,valid_x,valid_cost,valid_weights,train_x_precision,reg_set, N_min, H, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
  print("Best Alpha: " + best_alpha)
  
  end = time.time()    
  print "Elapsed time: " + str(end-start)

  #FIND TEST SET COST    
  path = decision_path(test_x,spo_dt_a,spo_dt_b)
  costs = apply_leaf_decision(test_cost,path,spo_dt_w, subtract_optimal=False, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)

with open(fname_out, 'wb') as output:
  pickle.dump(costs, output, pickle.HIGHEST_PROTOCOL)

print "Average test set cost (max is better): " + str(-1.0*np.mean(costs))
print "Average test set weighted cost (max is better): " + str(-1.0*np.dot(costs,test_weights)/np.sum(test_weights))