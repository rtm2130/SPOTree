'''
Runs SPOT (MILP) algorithm on shortest path dataset with nonlinear mapping from features to costs ("nonlinear")
Outputs decision costs for each test-set instance as pickle file
Note on notation: the paper uses r to denote the binary variables which map training observations to leaves. This code uses z rather than r.
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
  (5) H: training depth of tree, e.g. "5"
  (6) N_min: min. number of (weighted) observations per leaf, e.g. "100"
  (7) train_x_precision: contextual features x are rounded to train_x_precision before fitting MILP (e.g., 2 = two decimal places)
    higher values of train_x_precision will be more precise but take more computational time
  (8) reg_set_str: sequence of regularization parameters to try (tuned using cross validation), e.g. "0.001-0.01-0.1"
    if "None", fits MILP using no regularizaiton and then prunes using CART pruning procedure (with SPO loss as pruning metric)
  (9) solver_time_limit: MILP solver is terminated after solver_time_limit seconds, returning best-found solution
Values of input arguments used in paper:
  (1) n_train: consider values 200, 10000
  (2) eps:
    n_train = 200: considered values 0, 0.25
    n_train = 10000: considered values 0, 0.5
  (3) deg_set_str: "2-10"
  (4) reps_st, reps_end: reps_st = 0, reps_end = 10
  (5) H: 
    n_train = 200: considered depths of 1, 2, 3, 1000
    n_train = 10000: considered depths of 2, 4, 6, 1000
  (6) N_min: 20
  (7) train_x_precision: 2
  (8) reg_set_str: "None"
  (9) solver_time_limit: 16200
'''

import time

import numpy as np
import pickle
from spo_opt_tree_nonlinear import*
from SPO2CART import SPO2CART
from SPO_tree_greedy import SPOTree
from decision_problem_solver import*
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
#optimal tree params
H = int(sys.argv[6])#2 #H = max tree depth
N_min = int(sys.argv[7])#4 #N_min = minimum number of observations per leaf node
#higher values of train_x_precision will be more precise but take more computational time
#values >= 8 might cause numerical errors
train_x_precision = int(sys.argv[8])#2
reg_set_str = sys.argv[9]#"None"
if reg_set_str == "None":
  reg_set = None
else:
  reg_set = [int(k) for k in reg_set_str.split('-')]#None
#reg_set = [0.001] #if reg_set = None, uses CART to prune tree
solver_time_limit = int(sys.argv[10])
########################################
#output filename
fname_out = "SPOopt_nonlin_costs_tr"+str(n_train)+"_eps"+str(eps)+"_degSet"+deg_set_str+"_repsSt"+str(reps_st)+"_repsEnd"+str(reps_end)+"_depth"+str(H)+"_minObs"+str(N_min)+"_prec"+str(train_x_precision)+"_regset"+reg_set_str+"_tLim"+str(solver_time_limit)+".pkl";
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
    
    #FIT SPO OPTIMAL TREE
    if reg_set is None:
      
      #FIT SPO GREEDY TREE AS INITIAL SOLUTION
      def truncate_train_x(train_x, train_x_precision):
        return(np.around(train_x, decimals=train_x_precision))
      train_x_truncated = truncate_train_x(train_x, train_x_precision)
      #SPO_weight_param: number between 0 and 1:
      #Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
      my_tree = SPOTree(max_depth = H, min_weights_per_node = N_min, quant_discret = 0.01, debias_splits=False, SPO_weight_param=1.0, SPO_full_error=True)
      my_tree.fit(train_x_truncated,train_cost,verbose=False,feats_continuous=True);

      #PRUNE SPO GREEDY TREE USING TRAINING SET (TO GET RID OF REDUNDANT LEAVES)   
      my_tree.prune(train_x_truncated, train_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
      spo_greedy_a, spo_greedy_b, spo_greedy_z = my_tree.get_tree_encoding(x_train=train_x_truncated)
      
      #(OPTIONAL) PRUNE SPO GREEDY TREE USING VALIDATION SET
#      my_tree.prune(valid_x, valid_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
#      spo_greedy_a, spo_greedy_b, spo_greedy_z = my_tree.get_tree_encoding(x_train=train_x_truncated)
#      alpha = my_tree.get_pruning_alpha()
      
      #FIT SPO OPTIMAL TREE USING FOUND INITIAL SOLUTION
      reg_param = 1e-4 #introduce very small amount of regularization to ensure leaves with zero predictive power are aggregated
      spo_dt_a, spo_dt_b, spo_dt_w, spo_dt_y, spo_dt_z, spo_dt_l, spo_dt_d = spo_opt_tree(train_cost,train_x,train_x_precision,reg_param, N_min, H,
                                                                                          a_start=spo_greedy_a, z_start=spo_greedy_z,
                                                                                          Presolve=2, Seed=0, TimeLimit=solver_time_limit,
                                                                                          returnAllOptvars=True)
      
      end = time.time()    
      print "Elapsed time: " + str(end-start)
      
      #(IF NOT USING POSTPRUNING) FIND TEST SET COST    
#      path = decision_path(test_x,spo_dt_a,spo_dt_b)
#      costs_deg[deg][trial_num] = apply_leaf_decision(test_cost,path,spo_dt_w, subtract_optimal=False)
      
      #PRUNE MILP TREE USING CART PRUNING METHOD ON VALIDATION SET
      spo2cart = SPO2CART(spo_dt_a, spo_dt_b)
      spo2cart.fit(train_x,train_cost,train_x_precision,verbose=False,feats_continuous=True)
      spo2cart.prune(valid_x, valid_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
      #my_tree.traverse() #prints out the pruned tree 
      #(IF PRUNED) FIND TEST SET COST
      pred_decision = spo2cart.est_decision(test_x)
      costs_deg[deg][trial_num] = [np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
      
    else:
      spo_dt_a, spo_dt_b, spo_dt_w, _, best_alpha = spo_opt_tunealpha(train_x,train_cost,valid_x,valid_cost,train_x_precision,reg_set, N_min, H)
      print("Best Alpha: " + best_alpha)
      
      end = time.time()    
      print "Elapsed time: " + str(end-start)

      #FIND TEST SET COST    
      path = decision_path(test_x,spo_dt_a,spo_dt_b)
      costs_deg[deg][trial_num] = apply_leaf_decision(test_cost,path,spo_dt_w, subtract_optimal=False)
    
    # Saving the objects occasionally:
    if trial_num % 5 == 0:
      with open(fname_out, 'wb') as output:
        pickle.dump(costs_deg, output, pickle.HIGHEST_PROTOCOL)

with open(fname_out, 'wb') as output:
  pickle.dump(costs_deg, output, pickle.HIGHEST_PROTOCOL)
  
# Getting back the objects:
#with open(fname_out, 'rb') as input:
#  costs_deg = pickle.load(input)