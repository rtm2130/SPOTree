"""
SPO RANDOM FOREST IMPLEMENTATION

This code will work for general predict-then-optimize applications. Fits SPO Forest to dataset of feature-cost pairs.

The structure of the decision-making problem of interest should be encoded in a file called decision_problem_solver.py. 
Specifically, this code requires two functions:
  get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector for underlying decision problem)
  find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors 
"""
import numpy as np
from mtp import MTP
from decision_problem_solver import*
from scipy.spatial import distance
from SPO_tree_greedy import SPOTree
from joblib import Parallel, delayed
from collections import Counter

class SPOForest(object):
  '''
  This function initializes the SPO forest
  
  FOREST PARAMETERS:
    
  n_estimators: number of SPO trees in the random forest 
  
  max_features: number of features to consider when looking for the best split in each node
  
  run_in_parallel, num_workers: if run_in_parallel is set to True, enables parallel computing among num_workers threads. 
  If num_workers is not specified, uses the number of cpu cores available. The task of computing each SPO tree in the forest
  is distributed among the available cores. (each tree may only use 1 core and thus this arg is set to None in SPOTree class)
  
  TREE PARAMETERS (DIRECTLY PASSED TO SPOTree CLASS):
  
  max_depth: maximum training depth of each tree in the forest (default = Inf: no depth limit)
  
  min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node for each tree in the forest
  
  quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
  
  SPO_weight_param: splits are decided through loss = SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
    SPO_weight_param = 1.0 -> SPO loss
    SPO_weight_param = 0.0 -> MSE loss (i.e., CART)
  
  SPO_full_error: if SPO error is used, are the full errors computed for split evaluation, 
    i.e. are the alg's decision losses subtracted by the optimal decision losses?
  
  Keep all other parameter values as default
  '''
  def __init__(self, n_estimators=10, run_in_parallel=False, num_workers=None, **kwargs): 
    self.n_estimators = n_estimators
    if (run_in_parallel == False):
      num_workers = 1
    if num_workers is None:
      num_workers = -1 #this uses all available cpu cores
    self.run_in_parallel = run_in_parallel
    self.num_workers = num_workers
    
    self.forest = [None]*n_estimators
    for t in range(n_estimators):
      self.forest[t] = SPOTree(**kwargs)
  
  '''
  This function fits the SPO forest on data (X,C,weights).
  
  X: The feature data used in tree splits. Can either be a pandas data frame or numpy array, with:
    (a) rows of X = observations
    (b) columns of X = features
  C: the cost vectors used in the leaf node models. Must be a numpy array, with:
    (a) rows of C = observations
    (b) columns of C = cost vector components
  weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding weight of observation i
  feats_continuous: If False, all feature are treated as categorical. If True, all feature are treated as continuous.
    feats_continuous can also be a boolean vector of dimension = num_features specifying how to treat each feature
  verbose: if verbose=True, prints out progress in tree fitting procedure
  verbose_forest: if verbose_forest=True, prints out progress in the forest fitting procedure
  seed: seed for rng
  '''
  def fit(self, X, C, weights=None, verbose_forest=False, seed=None, 
          feats_continuous=False, verbose=False, refit_leaves=False,
          **kwargs):
    
    self.decision_kwargs = kwargs
    
    num_obs = C.shape[0]
    
    if weights is None:
      weights = np.ones([num_obs])
    
    if seed is not None:
      np.random.seed(seed)
    tree_seeds = np.random.randint(0, high=2**32-1, size=self.n_estimators)
    
    
    if self.num_workers == 1:
      for t in range(self.n_estimators):
        if verbose_forest == True:
          print("Fitting tree " + str(t+1) + "out of " + str(self.n_estimators))
        np.random.seed(tree_seeds[t]) 
        bootstrap_inds = np.random.choice(range(num_obs), size=num_obs, replace=True)
        Xb = np.copy(X[bootstrap_inds])
        Cb = np.copy(C[bootstrap_inds])
        weights_b = np.copy(weights[bootstrap_inds])
        self.forest[t].fit(Xb, Cb, weights=weights_b, seed=tree_seeds[t], 
                           feats_continuous=feats_continuous, verbose=verbose, refit_leaves=refit_leaves,
                           **kwargs)
    
    else:
      self.forest = Parallel(n_jobs=self.num_workers, max_nbytes=1e5)(delayed(_fit_tree)(t, self.n_estimators, self.forest[t], X, C, weights, verbose_forest, tree_seeds[t], feats_continuous=feats_continuous, verbose=verbose, refit_leaves=refit_leaves, **kwargs) for t in range(self.n_estimators))
       
  
  '''
  Prints all trees in the forest
  Required: call forest fit() method first
  '''
  def traverse(self):
    for t in range(self.n_estimators):
      print("Printing Tree " + str(t+1) + "out of " + str(self.n_estimators))
      self.forest[t].traverse()
      print("\n\n\n")
      
  '''
  Predicts decisions or costs given data Xnew
  Required: call tree fit() method first
  
  method: method for aggregating decisions from each of the individual trees in the forest. Two approaches:
    (1) "mean": averages predicted cost vectors from each tree, then finds decision with respect to average cost vector
    (2) "mode": each tree in the forest estimates an optimal decision; take the most-recommended decision
  
  NOTE: return_loc argument not supported:
  (If return_loc=True, est_decision will also return the leaf node locations for the data, in addition to the decision.)
  '''
  def est_decision(self, Xnew, method="mean"):
    if method == "mean":
      forest_costs = self.est_cost(Xnew)
      forest_decisions = find_opt_decision(forest_costs,**self.decision_kwargs)['weights']
    
    elif method == "mode":
      num_obs = Xnew.shape[0]
      tree_decisions = [None]*self.n_estimators
      for t in range(self.n_estimators):
        tree_decisions[t] = self.forest[t].est_decision(Xnew)
      tree_decisions = np.array(tree_decisions)
      forest_decisions = np.zeros((num_obs,tree_decisions.shape[2]))
      for i in range(num_obs):
        forest_decisions[i] = _get_mode_row(tree_decisions[:,i,:])
    
    return forest_decisions
  
  def est_cost(self, Xnew):
    tree_costs = [None]*self.n_estimators
    for t in range(self.n_estimators):
      tree_costs[t] = self.forest[t].est_cost(Xnew)
    tree_costs = np.array(tree_costs)
    forest_costs = np.mean(tree_costs,axis=0)    
    return forest_costs

'''
Helper methods (ignore)
'''
  
def _fit_tree(t, n_estimators, tree, X, C, weights, verbose_forest, tree_seed, **kwargs):
  if verbose_forest == True:
    print("Fitting tree " + str(t+1) + "out of " + str(n_estimators))
  
  num_obs = C.shape[0]
  np.random.seed(tree_seed) 
  bootstrap_inds = np.random.choice(range(num_obs), size=num_obs, replace=True)
  Xb = np.copy(X[bootstrap_inds])
  Cb = np.copy(C[bootstrap_inds])
  weights_b = np.copy(weights[bootstrap_inds])
  tree.fit(Xb, Cb, weights=weights_b, seed=tree_seed, **kwargs)
  return(tree)

def _get_mode_row(a):
  return(np.array(Counter(map(tuple, a)).most_common()[0][0]))