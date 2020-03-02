'''
Runs random forest algorithm on Yahoo News dataset
This code considers two methods of aggregating individual tree predictions to obtain a forest decision:
  - "mean": averages cost predictions for each tree in the forest; outputs decision associated with average cost
  - "mode": outputs the mode decision recommended by the trees in the forest
Outputs decision costs for each test-set instance as pickle file
Takes multiple input arguments:
  (1) max_depth_set_str: sequence of training depths tuned using cross validation, e.g. "2-4-5"
  (2) min_samples_leaf_set_str: sequence of "min. (weighted) observations per leaf" tuned using cross validation, e.g. "20-50-100"
  (3) n_estimators_set_str: sequence of number of trees in forest tuned using cross validation, e.g. "20-50-100"
  (4) max_features_set_str: sequence of number of features used in feature bagging tuned using cross validation, e.g. "2-3-4"
  (5) algtype: set equal to "MSE" (CART forest) or "SPO" (SPOT forest)
  (6) number of workers to use in parallel processing (i.e., fitting individual trees in the forest in parallel)
  (7) decision_problem_seed: seed controlling generation of constraints in article recommendation problem (-1 = no constraints)
  (8) train_size: number of random obs. to extract from the training data. 
  Only useful in limiting the size of the training data (-1 = use full training data)
  (9) quant_discret: continuous variable split points in the trees are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
Values of input arguments used in paper:
  (1) max_depth_set_str: "1000"
  (2) min_samples_leaf_set_str: "10000"
  (3) n_estimators_set_str: "50"
  (4) max_features_set_str: "2-3-4-5"
  (5) algtype: "MSE" for CART forest, "SPO" for SPOT forest
  (6) number of workers to use in parallel processing: 10
  (7) decision_problem_seed: ran 9 constraint instances corresponding to seeds of 10, 11, 12, ..., 18
  (8) train_size: -1
  (9) quant_discret: 0.05
'''

import time

import numpy as np
import pickle
from gurobipy import*
from decision_problem_solver import*
from SPOForest import SPOForest
import sys

########################################
#SEEDS FOR RANDOM NUMBER GENERATORS
#seed for rngs in random forest
forest_seed = 0
#seed controlling random subset of training data used (if full training data is not being used)
select_train_seed = 0 
########################################
#training parameters
max_depth_set_str = sys.argv[1]
max_depth_set=[int(k) for k in max_depth_set_str.split('-')]#[None]
min_samples_leaf_set_str = sys.argv[2]
min_samples_leaf_set=[int(k) for k in min_samples_leaf_set_str.split('-')]#[5]
n_estimators_set_str = sys.argv[3]
n_estimators_set=[int(k) for k in n_estimators_set_str.split('-')]#[100,500]
max_features_set_str = sys.argv[4]
max_features_set=[int(k) for k in max_features_set_str.split('-')]#[3]
algtype=sys.argv[5] #either "MSE" or "SPO"
#number of workers
if sys.argv[6] == "1":
  run_in_parallel = False
  num_workers = None
else: 
  run_in_parallel = True
  num_workers = int(sys.argv[6])
#ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
decision_problem_seed=int(sys.argv[7]) #if -1, use no constraints in decision problem
train_size=int(sys.argv[8]) #if you want to limit the size of the training data (-1 = no limit)
quant_discret=float(sys.argv[9]) 
########################################
#output filename
fname_out_mean = algtype+"Forest_news_costs_depthSet"+max_depth_set_str+"_minObsSet"+min_samples_leaf_set_str+"_nEstSet"+n_estimators_set_str+"_mFeatSet"+max_features_set_str+"_aMethod"+"mean"+"_probSeed"+str(decision_problem_seed)+"_nTrain"+str(train_size)+"_qd"+str(quant_discret)+".pkl";
fname_out_mode = algtype+"Forest_news_costs_depthSet"+max_depth_set_str+"_minObsSet"+min_samples_leaf_set_str+"_nEstSet"+n_estimators_set_str+"_mFeatSet"+max_features_set_str+"_aMethod"+"mode"+"_probSeed"+str(decision_problem_seed)+"_nTrain"+str(train_size)+"_qd"+str(quant_discret)+".pkl";
#############################################################################
#############################################################################
#############################################################################

#generate decision problem
num_constr = 5 #number of Aw <= b constraints
num_dec = 6 #number of decisions
#ineligible_seeds = [27,28,29,32,39] #considered range = 10-40 inclusive
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

def forest_traintest(train_x,train_cost,train_weights,test_x,test_cost,test_weights,n_estimators,max_depth,min_samples_leaf,max_features,run_in_parallel,num_workers,algtype, A_constr, b_constr, l_constr, u_constr):
    if algtype == "MSE":
      SPO_weight_param=0.0
    elif algtype == "SPO":
      SPO_weight_param=1.0
    regr = SPOForest(n_estimators=n_estimators,run_in_parallel=run_in_parallel,num_workers=num_workers, 
                     max_depth=max_depth, min_weights_per_node=min_samples_leaf, quant_discret=quant_discret, debias_splits=False,
                     max_features=max_features,
                     SPO_weight_param=SPO_weight_param, SPO_full_error=True)
    regr.fit(train_x, train_cost, train_weights, verbose_forest=True, verbose=False, feats_continuous=True, seed=forest_seed,
             A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
    pred_decision_mean = regr.est_decision(test_x, method="mean")
    pred_decision_mode = regr.est_decision(test_x, method="mode")
    alg_costs_mean = [np.sum(test_cost[i] * pred_decision_mean[i]) for i in range(0,len(pred_decision_mean))]
    alg_costs_mode = [np.sum(test_cost[i] * pred_decision_mode[i]) for i in range(0,len(pred_decision_mode))]
    return regr, np.dot(alg_costs_mean,test_weights)/np.sum(test_weights), np.dot(alg_costs_mode,test_weights)/np.sum(test_weights)

def forest_tuneparams(train_x,train_cost,train_weights,valid_x,valid_cost,valid_weights,n_estimators_set,max_depth_set,min_samples_leaf_set,max_features_set,run_in_parallel,num_workers,algtype, A_constr, b_constr, l_constr, u_constr):
    best_err_mean = np.float("inf")
    best_err_mode = np.float("inf")
    for n_estimators in n_estimators_set:
      for max_depth in max_depth_set:
        for min_samples_leaf in min_samples_leaf_set:
          for max_features in max_features_set:
            regr, err_mean, err_mode = forest_traintest(train_x,train_cost,train_weights,valid_x,valid_cost,valid_weights,n_estimators,max_depth,min_samples_leaf,max_features,run_in_parallel,num_workers,algtype, A_constr, b_constr, l_constr, u_constr)
            if err_mean <= best_err_mean:
              best_regr_mean, best_err_mean, best_n_estimators_mean,best_max_depth_mean,best_min_samples_leaf_mean,best_max_features_mean = regr, err_mean, n_estimators,max_depth,min_samples_leaf,max_features
            if err_mode <= best_err_mode:
              best_regr_mode, best_err_mode, best_n_estimators_mode,best_max_depth_mode,best_min_samples_leaf_mode,best_max_features_mode = regr, err_mode, n_estimators,max_depth,min_samples_leaf,max_features
    
    print("Best n_estimators (mean method): " + str(best_n_estimators_mean))
    print("Best max_depth (mean method): " + str(best_max_depth_mean))
    print("Best min_samples_leaf (mean method): " + str(best_min_samples_leaf_mean))
    print("Best max_features (mean method): " + str(best_max_features_mean))
    
    print("Best n_estimators (mode method): " + str(best_n_estimators_mode))
    print("Best max_depth (mode method): " + str(best_max_depth_mode))
    print("Best min_samples_leaf (mode method): " + str(best_min_samples_leaf_mode))
    print("Best max_features (mode method): " + str(best_max_features_mode))
    
    return best_regr_mean, best_err_mean, best_n_estimators_mean,best_max_depth_mean,best_min_samples_leaf_mean,best_max_features_mean, best_regr_mode, best_err_mode, best_n_estimators_mode,best_max_depth_mode,best_min_samples_leaf_mode,best_max_features_mode

start = time.time()

#FIT FOREST
regr_mean,_,_,_,_,_,regr_mode,_,_,_,_,_ = forest_tuneparams(train_x,train_cost,train_weights,valid_x,valid_cost,valid_weights,n_estimators_set,max_depth_set,min_samples_leaf_set,max_features_set, run_in_parallel, num_workers, algtype, A_constr, b_constr, l_constr, u_constr)

end = time.time()    
print "Elapsed time: " + str(end-start)

#FIND TEST SET COST
pred_decision_mean = regr_mean.est_decision(test_x, method="mean")
pred_decision_mode = regr_mode.est_decision(test_x, method="mode")
costs_mean = [np.sum(test_cost[i] * pred_decision_mean[i]) for i in range(0,len(pred_decision_mean))]
costs_mode = [np.sum(test_cost[i] * pred_decision_mode[i]) for i in range(0,len(pred_decision_mode))]

print "Average test set cost (mean method) (max is better): " + str(-1.0*np.mean(costs_mean))
print "Average test set weighted cost (mean method) (max is better): " + str(-1.0*np.dot(costs_mean,test_weights)/np.sum(test_weights))
print "Average test set cost (mode method) (max is better): " + str(-1.0*np.mean(costs_mode))
print "Average test set weighted cost (mode method) (max is better): " + str(-1.0*np.dot(costs_mode,test_weights)/np.sum(test_weights))

with open(fname_out_mean, 'wb') as output:
  pickle.dump(costs_mean, output, pickle.HIGHEST_PROTOCOL)
with open(fname_out_mode, 'wb') as output:
  pickle.dump(costs_mode, output, pickle.HIGHEST_PROTOCOL)
  
# Getting back the objects:
#with open(fname_out, 'rb') as input:
#  costs_deg = pickle.load(input)