"""
Encodes SPOT MILP as the structure of a CART tree in order to apply CART's pruning method
Also supports traverse() which traverses the tree
"""
import numpy as np
from mtp_SPO2CART import MTP_SPO2CART
from decision_problem_solver import*
from scipy.spatial import distance


def truncate_train_x(train_x, train_x_precision):
  return(np.around(train_x, decimals=train_x_precision))

class SPO2CART(object):
  '''
  This function initializes the SPO tree
  
  Parameters:
  
  max_depth: the maximum depth of the pre-pruned tree (default = Inf: no depth limit)

  min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node

  min_depth: the minimum depth of the pre-pruned tree (default: set equal to max_depth)

  min_diff: if depth > min_depth, stop splitting if improvement in fit does not exceed min_diff
  
  binary_splits: if True, use binary splits when building the tree, else consider multiway splits 
  (i.e., when splitting on a variable, split on all unique vals)
  
  debias_splits/frac_debias_set/min_debias_set_size: Additional params when binary_splits = True. If debias_splits = True, then in each node,
  hold out frac_debias_set of the training set (w.r.t. case weights) to evaluate the error of the best splitting point for each feature. 
  Stop bias-correcting when we have insufficient data; i.e. the total weight in the debias set < min_debias_set_size.
    Note: after finding best split point, we then refit the model on all training data and recalculate the training error
  
  quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
  
  run_in_parallel: if set to True, enables parallel computing among num_workers threads. If num_workers is not
  specified, uses the number of cpu cores available.
  '''
  def __init__(self, a,b,**kwargs):
    
    kwargs["SPO_weight_param"] = 1.0
    
    if "SPO_full_error" not in kwargs:
      kwargs["SPO_full_error"] = True
    
    self.SPO_weight_param = kwargs["SPO_weight_param"]
    self.SPO_full_error = kwargs["SPO_full_error"]
    self.tree = MTP_SPO2CART(a,b,**kwargs)
  
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
  '''
  def fit(self, X, C, train_x_precision, 
          weights=None, feats_continuous=True, verbose=False, refit_leaves=False,
          **kwargs):
    self.decision_kwargs = kwargs
    X = truncate_train_x(X, train_x_precision)
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
                  weights=weights, feats_continuous=feats_continuous, verbose=verbose, refit_leaves=refit_leaves, 
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
    
  
  '''
  Produces decision given data Xnew
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