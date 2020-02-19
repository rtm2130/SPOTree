"""
SPO GREEDY TREE IMPLEMENTATION

This code will work for general predict-then-optimize applications. Fits SPO (greedy) tree to dataset of feature-cost pairs.

The structure of the decision-making problem of interest should be encoded in a file called decision_problem_solver.py. 
Specifically, this code requires two functions:
  get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector for underlying decision problem)
  find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors 

"""
import numpy as np
from mtp import MTP
from decision_problem_solver import*
from scipy.spatial import distance

class SPOTree(object):
  '''
  This function initializes the SPO tree
  
  Parameters:
  
  max_depth: maximum training depth of each tree in the forest (default = Inf: no depth limit)
  
  min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node for each tree in the forest
  
  quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
  
  SPO_weight_param: splits are decided through loss = SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
    SPO_weight_param = 1.0 -> SPO loss
    SPO_weight_param = 0.0 -> MSE loss (i.e., CART)
  
  SPO_full_error: if SPO error is used, are the full errors computed for split evaluation, 
    i.e. are the alg's decision losses subtracted by the optimal decision losses? 
  
  run_in_parallel: if set to True, enables parallel computing among num_workers threads. If num_workers is not
  specified, uses the number of cpu cores available.
  
  max_features: number of features to consider when looking for the best split in each node. Useful when building random forests. Default equal to total num features
  
  Keep all other parameter values as default
  '''
  def __init__(self, **kwargs): 
    self.SPO_weight_param = kwargs["SPO_weight_param"]
    self.SPO_full_error = kwargs["SPO_full_error"]
    self.tree = MTP(**kwargs)
  
  '''
  This function fits the tree on data (X,C,weights).
  
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
  
  Keep all other parameter values as default
  '''
  def fit(self, X, C, 
          weights=None, feats_continuous=False, verbose=False, refit_leaves=False, seed=None,
          **kwargs):
    self.pruned = False
    self.decision_kwargs = kwargs
    num_obs = C.shape[0]
    
    A = np.array(range(num_obs))
    if self.SPO_full_error == True and self.SPO_weight_param != 0.0:
      for i in range(num_obs):
        A[i] = find_opt_decision(C[i,:].reshape(1,-1),**kwargs)['objective'][0]
    
    if self.SPO_weight_param != 0.0 and self.SPO_weight_param != 1.0:
      if self.SPO_full_error == True:
        SPO_loss_bound = -float("inf")
        for i in range(num_obs):
          SPO_loss = -find_opt_decision(-C[i,:].reshape(1,-1),**kwargs)['objective'][0] - A[i]
          if SPO_loss >= SPO_loss_bound:
            SPO_loss_bound = SPO_loss
        
      else:
        c_max = np.max(C,axis=0)
        SPO_loss_bound = -find_opt_decision(-c_max.reshape(1,-1),**kwargs)['objective'][0]
      
      #Upper bound for MSE loss: maximum pairwise difference between any two elements
      dists = distance.cdist(C, C, 'sqeuclidean')
      MSE_loss_bound = np.max(dists)
        
    else:
      SPO_loss_bound = 1.0
      MSE_loss_bound = 1.0
    
    #kwargs["SPO_loss_bound"] = SPO_loss_bound
    #kwargs["MSE_loss_bound"] = MSE_loss_bound
    self.tree.fit(X,A,C,
                  weights=weights, feats_continuous=feats_continuous, verbose=verbose, refit_leaves=refit_leaves, seed=seed,
                  SPO_loss_bound = SPO_loss_bound, MSE_loss_bound = MSE_loss_bound,
                  **kwargs)
  
  '''
  Prints out the tree. 
  Required: call tree fit() method first
  Prints pruned tree if prune() method has been called, else prints unpruned tree
  verbose=True prints additional statistics within each leaf
  '''
  def traverse(self, verbose=False):
    self.tree.traverse(verbose=verbose)
  
  '''
  Prunes the tree. Set verbose=True to track progress
  '''
  def prune(self, Xval, Cval, 
            weights_val=None, one_SE_rule=True,verbose=False,approx_pruning=False):
    num_obs = Cval.shape[0]
    
    Aval = np.array(range(num_obs))
    if self.SPO_full_error == True and self.SPO_weight_param != 0.0:
      for i in range(num_obs):
        Aval[i] = find_opt_decision(Cval[i,:].reshape(1,-1),**self.decision_kwargs)['objective'][0]
    
    self.tree.prune(Xval,Aval,Cval,
                    weights_val=weights_val,one_SE_rule=one_SE_rule,verbose=verbose,approx_pruning=approx_pruning)
    self.pruned = True
    
  
  '''
  Produces decision or cost given data Xnew
  Required: call tree fit() method first
  Uses pruned tree if pruning method has been called, else uses unpruned tree
  Argument alpha controls level of pruning. If not specified, uses alpha trained from the prune() method
  
  As a step in finding the estimated decisions for data (Xnew), this function first finds
  the leaf node locations corresponding to each row of Xnew. It does so by a top-down search
  starting at the root node 0. 
  If return_loc=True, est_decision will also return the leaf node locations for the data, in addition to the decision.
  '''
  def est_decision(self, Xnew, alpha=None, return_loc=False):
    return self.tree.predict(Xnew, np.array(range(0,Xnew.shape[0])), alpha=alpha, return_loc=return_loc)
  
  def est_cost(self, Xnew, alpha=None, return_loc=False):
    return self.tree.predict(Xnew, np.array(range(0,Xnew.shape[0])), alpha=alpha, return_loc=return_loc, get_cost=True)

  '''
  Other methods (ignore)
  '''
  def get_tree_encoding(self, x_train=None):
    return self.tree.get_tree_encoding(x_train=x_train)
  
  def get_pruning_alpha(self):
    if self.pruned == True:
      return self.tree.alpha_best
    else:
      return(0)