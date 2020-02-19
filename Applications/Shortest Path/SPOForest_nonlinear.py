'''
Runs random forest algorithm on shortest path dataset with nonlinear mapping from features to costs ("nonlinear")
This code considers two methods of aggregating individual tree predictions to obtain a forest decision:
  - "mean": averages cost predictions for each tree in the forest; outputs decision associated with average cost
  - "mode": outputs the mode decision recommended by the trees in the forest
Outputs decision costs for each test-set instance as pickle file
Takes multiple input arguments:
  (1) n_train: number of training observations. can take values 200, 10000
  (2) eps: parameter (\bar{\epsilon}) in the paper controlling noise in mapping from features to costs.
    n_train = 200: can take values 0, 0.25
    n_train = 10000: can take values 0, 0.5
  (3) deg_set_str: set of deg parameters to try, e.g. "2-10". 
    deg = parameter "degree" in the paper controlling nonlinearity in mapping from features to costs. 
    can try values in {2,10}
  (4) reps_st, reps_end: we provide 10 total datasets corresponding to different generated B values (matrix mapping features to costs). 
    script will run code on problem instances reps_st to reps_end 
  (5) max_depth_set_str: sequence of training depths tuned using cross validation, e.g. "2-4-5"
  (6) min_samples_leaf_set_str: sequence of "min. (weighted) observations per leaf" tuned using cross validation, e.g. "20-50-100"
  (7) n_estimators_set_str: sequence of number of trees in forest tuned using cross validation, e.g. "20-50-100"
  (8) max_features_set_str: sequence of number of features used in feature bagging tuned using cross validation, e.g. "2-3-4"
  (9) aggr_method: method for aggregating individual tree cost predictions to arrive at forest decisions. Either "mean" or "mode"
  (10) algtype: set equal to "MSE" (CART forest) or "SPO" (SPOT forest)
  (11) number of workers to use in parallel processing (i.e., fitting individual trees in the forest in parallel)
Values of input arguments used in paper:
  (1) n_train: consider values 200, 10000
  (2) eps:
    n_train = 200: considered values 0, 0.25
    n_train = 10000: considered values 0, 0.5
  (3) deg_set_str: "2-10"
  (4) reps_st, reps_end: reps_st = 0, reps_end = 10
  (5) max_depth_set_str: "1000"
  (6) min_samples_leaf_set_str: "20"
  (7) n_estimators_set_str: "100"
  (8) max_features_set_str: "2-3-4-5"
  (9) aggr_method: "mean"
  (10) algtype: "MSE" (CART forest) or "SPO" (SPOT forest)
  (11) number of workers to use in parallel processing: 8
'''

import time

import numpy as np
import pickle
from gurobipy import*
from SPOForest import SPOForest
from decision_problem_solver import*
##############################################
forest_seed = 0 #seed to set random forest rng
##############################################
import sys
#problem parameters
n_train = int(sys.argv[1])#200
eps = float(sys.argv[2])#0
deg_set_str = sys.argv[3]
deg_set=[int(k) for k in deg_set_str.split('-')]#[2,4,6,8,10]
#evaluate algs of dataset replications from rep_st to rep_end
reps_st = int(sys.argv[4])#0 #can be as low as 0
reps_end = int(sys.argv[5])#1 #can be as high as 50
valid_frac = 0.2
########################################
#training parameters
max_depth_set_str = sys.argv[6]
max_depth_set=[int(k) for k in max_depth_set_str.split('-')]#[None]
min_samples_leaf_set_str = sys.argv[7]
min_samples_leaf_set=[int(k) for k in min_samples_leaf_set_str.split('-')]#[5]
n_estimators_set_str = sys.argv[8]
n_estimators_set=[int(k) for k in n_estimators_set_str.split('-')]#[100,500]
max_features_set_str = sys.argv[9]
max_features_set=[int(k) for k in max_features_set_str.split('-')]#[3]
aggr_method=sys.argv[10] #either "mean" or "mode"
algtype=sys.argv[11] #either "MSE" or "SPO"
#number of workers
if sys.argv[12] == "1":
  run_in_parallel = False
  num_workers = None
else: 
  run_in_parallel = True
  num_workers = int(sys.argv[12])
########################################
#output filename
fname_out = algtype+"Forest_nonlin_costs_tr"+str(n_train)+"_eps"+str(eps)+"_degSet"+deg_set_str+"_repsSt"+str(reps_st)+"_repsEnd"+str(reps_end)+"_depthSet"+max_depth_set_str+"_minObsSet"+min_samples_leaf_set_str+"_nEstSet"+n_estimators_set_str+"_mFeatSet"+max_features_set_str+"_aMethod"+aggr_method+".pkl";
#############################################################################
#############################################################################
#############################################################################
#data = pickle.load(open('non_linear_big_data_dim4.p','rb'))
#'non_linear_data_dim4.p' has the following options:
#n_train: 200, 400, 800
#nonlinear degrees: 8, 2, 4, 10, 6
#eps: 0, 0.25, 0.5
#50 replications of the experiment (0-49)
#dataset characteristics: 5 continuous features x, dimension 4 grid c, 1000 test set observations
if n_train == 10000:
  data = pickle.load(open('non_linear_bigdata10000_dim4.p','rb'))
else:
  data = pickle.load(open('non_linear_data_dim4.p','rb'))
n_test = 1000
#############################################################################
#############################################################################
#############################################################################
assert(reps_st >= 0)
assert(reps_end <= 50)
n_reps = reps_end-reps_st

def forest_traintest(train_x,train_cost,test_x,test_cost,n_estimators,max_depth,min_samples_leaf,max_features,run_in_parallel,num_workers,aggr_method, algtype):
    if algtype == "MSE":
      SPO_weight_param=0.0
    elif algtype == "SPO":
      SPO_weight_param=1.0
    regr = SPOForest(n_estimators=n_estimators,run_in_parallel=run_in_parallel,num_workers=num_workers, 
                     max_depth=max_depth, min_weights_per_node=min_samples_leaf, quant_discret=0.01, debias_splits=False,
                     max_features=max_features,
                     SPO_weight_param=SPO_weight_param, SPO_full_error=True)
    regr.fit(train_x, train_cost, verbose_forest=True, verbose=False, feats_continuous=True, seed=forest_seed)
    pred_decision = regr.est_decision(test_x, method=aggr_method)
    return regr, np.mean([np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))])

def forest_tuneparams(train_x,train_cost,valid_x,valid_cost,n_estimators_set,max_depth_set,min_samples_leaf_set,max_features_set,run_in_parallel,num_workers,aggr_method, algtype):
    best_err = np.float("inf")
    for n_estimators in n_estimators_set:
      for max_depth in max_depth_set:
        for min_samples_leaf in min_samples_leaf_set:
          for max_features in max_features_set:
            regr, err = forest_traintest(train_x,train_cost,valid_x,valid_cost,n_estimators,max_depth,min_samples_leaf,max_features,run_in_parallel,num_workers,aggr_method, algtype)
            if err <= best_err:
              best_regr, best_err, best_n_estimators,best_max_depth,best_min_samples_leaf,best_max_features = regr, err, n_estimators,max_depth,min_samples_leaf,max_features
    
    print("Best n_estimators: " + str(best_n_estimators))
    print("Best max_depth: " + str(best_max_depth))
    print("Best min_samples_leaf: " + str(best_min_samples_leaf))
    print("Best max_features: " + str(best_max_features))
    return best_regr, best_err, best_n_estimators,best_max_depth,best_min_samples_leaf,best_max_features

#costs_deg[deg] yields a n_reps*n_test matrix of costs corresponding to the experimental data for deg, i.e.
#costs_deg[deg][i][j] gives the observed cost on test set i (0-49) example j (0-(n_test-1))
costs_deg = {}

for deg in deg_set:
  costs_deg[deg] = np.zeros((n_reps,n_test))
    
  for trial_num in range(reps_st,reps_end):
    train_x,train_cost,test_x,test_cost = data[n_train][deg][eps][trial_num]
    print "Deg "+str(deg)+", Trial Number "+str(trial_num)+" out of " + str(reps_end)
    
    #split up training data into train/valid split
    n_valid = int(np.floor(n_train*valid_frac))
    valid_x = train_x[:n_valid]
    valid_cost = train_cost[:n_valid]
    train_x = train_x[n_valid:]
    train_cost = train_cost[n_valid:]
    
    start = time.time()
    
    #FIT FOREST
    regr,_,_,_,_,_ = forest_tuneparams(train_x,train_cost,valid_x,valid_cost,n_estimators_set,max_depth_set,min_samples_leaf_set,max_features_set, run_in_parallel, num_workers, aggr_method, algtype)
    
    end = time.time()    
    print "Elapsed time: " + str(end-start)

    #FIND TEST SET COST
    pred_decision = regr.est_decision(test_x, method=aggr_method)
    incurred_test_cost = [np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
    
    print "Average test set cost: " + str(np.mean(incurred_test_cost))
    
    costs_deg[deg][trial_num] = incurred_test_cost

    # Saving the objects occasionally:
    if trial_num % 25 == 0:
      with open(fname_out, 'wb') as output:
        pickle.dump(costs_deg, output, pickle.HIGHEST_PROTOCOL)

with open(fname_out, 'wb') as output:
  pickle.dump(costs_deg, output, pickle.HIGHEST_PROTOCOL)
  
# Getting back the objects:
#with open(fname_out, 'rb') as input:
#  costs_deg = pickle.load(input)