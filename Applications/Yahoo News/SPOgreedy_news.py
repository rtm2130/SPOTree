'''
Runs SPOT (greedy) / CART algorithm on Yahoo News dataset
Outputs algorithm decision costs for each test-set instance as pickle file
Also outputs optimal decision costs for each test-set instance as pickle file
Takes multiple input arguments:
  (1) max_depth: training depth of tree, e.g. "5"
  (2) min_weights_per_node: min. number of (weighted) observations per leaf, e.g. "100"
  (3) algtype: set equal to "MSE" (CART) or "SPO" (SPOT greedy)
  (4) decision_problem_seed: seed controlling generation of constraints in article recommendation problem (-1 = no constraints)
  (5) train_size: number of random obs. to extract from the training data. 
  Only useful in limiting the size of the training data (-1 = use full training data)
  (6) quant_discret: continuous variable split points in the tree is chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
Values of input arguments used in paper:
  (1) max_depth: considered depths of 2, 4, 6, 1000
  (2) min_weights_per_node: "10000"
  (3) algtype: "MSE" (CART) or "SPO" (SPOT greedy)
  (4) decision_problem_seed: ran 9 constraint instances corresponding to seeds of 10, 11, 12, ..., 18
  (5) train_size: -1
  (6) quant_discret: 0.05
'''

import time

import numpy as np
import pickle
from decision_problem_solver import*
from SPO_tree_greedy import SPOTree
import sys
    
##############################################
#seed controlling random subset of training data used (if full training data is not being used)
select_train_seed = 0
########################################
#training parameters
max_depth = int(sys.argv[1])#3
min_weights_per_node = int(sys.argv[2])#20
algtype = sys.argv[3] #either "MSE" or "SPO"
#ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
decision_problem_seed=int(sys.argv[4]) #if -1, use no constraints in decision problem
train_size=int(sys.argv[5]) #if you want to limit the size of the training data (-1 = no limit)
quant_discret=float(sys.argv[6]) 
########################################
#output filename for alg
fname_out = algtype+"greedy_news_costs_depth"+str(max_depth)+"_minObs"+str(min_weights_per_node)+"_probSeed"+str(decision_problem_seed)+"_nTrain"+str(train_size)+"_qd"+str(quant_discret)+".pkl";
#output filename for opt costs
fname_out_opt = "Opt_news_costs"+"_probSeed"+str(decision_problem_seed)+".pkl";
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

#FIT ALGORITHM
#SPO_weight_param: number between 0 and 1:
#Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
if algtype == "MSE":
  SPO_weight_param=0.0
elif algtype == "SPO":
  SPO_weight_param=1.0      
my_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = quant_discret, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
my_tree.fit(train_x,train_cost,weights=train_weights,verbose=False,feats_continuous=True,
            A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr); #verbose specifies whether fitting procedure should print progress

my_tree.traverse() #prints out the unpruned tree 

end = time.time()    
print "Elapsed time: " + str(end-start)

##################

start = time.time()

#PRUNE DECISION TREE USING VALIDATION SET
my_tree.prune(valid_x, valid_cost, weights_val=valid_weights, verbose=True, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress

end = time.time()    
print "Elapsed time: " + str(end-start)

my_tree.traverse() #prints out the pruned tree

##################

#FIND TEST SET ALGORITHM COST AND OPTIMAL COST
opt_decision = find_opt_decision(test_cost, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)['weights']
pred_cost = my_tree.est_cost(test_x)
cost_pred_error = [np.mean(abs(pred_cost[i] - test_cost[i])) for i in range(0,pred_cost.shape[0])]
pred_decision = my_tree.est_decision(test_x)
costs = [np.sum(test_cost[i] * pred_decision[i]) for i in range(0,pred_decision.shape[0])]
optcosts = [np.sum(test_cost[i] * opt_decision[i,:]) for i in range(0,opt_decision.shape[0])]

print "Average test set cost (max is better): " + str(-1.0*np.mean(costs))
print "Average test set weighted cost (max is better): " + str(-1.0*np.dot(costs,test_weights)/np.sum(test_weights))
print "Average optimal test set cost (max is better): " + str(-1.0*np.mean(optcosts))
print "Average optimal test set weighted cost (max is better): " + str(-1.0*np.dot(optcosts,test_weights)/np.sum(test_weights))

with open(fname_out, 'wb') as output:
  pickle.dump(costs, output, pickle.HIGHEST_PROTOCOL)
with open(fname_out_opt, 'wb') as output:
  pickle.dump(optcosts, output, pickle.HIGHEST_PROTOCOL)