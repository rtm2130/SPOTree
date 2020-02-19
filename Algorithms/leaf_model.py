'''
Helper class for mtp.py

Defines the leaf nodes of the tree, specifically
- the computation of the predicted cost vectors and decisions within the given leaf of the tree
- the SPO/MSE loss from using the predicted decision within the leaf
'''

import numpy as np
from decision_problem_solver import*
#from scipy.spatial import distance

'''
mtp.py depends on the classes and functions below. 
These classes/methods are used to define the model object in each leaf node,
as well as helper functions for certain operations in the tree fitting procedure.

Summary of methods and functions to specify:
  Methods as a part of class LeafModel: fit(), predict(), to_string(), error(), error_pruning()
  Other helper functions: get_sub(), are_Ys_diverse()
  
'''

'''
LeafModel: the model used in each leaf. 
Has five methods: fit, predict, to_string, error, error_pruning

SPO_weight_param: number between 0 and 1:
Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
'''
class LeafModel(object):
  
  #Any additional args passed to mtp's init() function are directly passed here
  def __init__(self,*args,**kwargs):
    self.SPO_weight_param = kwargs["SPO_weight_param"]
    self.SPO_full_error = kwargs["SPO_full_error"]
    return
  
  '''
  This function trains the leaf node model on the data (A,Y,weights).
  
  A and Y can take any form (lists, matrices, vectors, etc.). For our applications, I recommend making Y
  the response data (e.g., choices) and A alternative-specific data (e.g., features, choice sets)
  
  weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding 
    weight of observation/customer i. If you know you will not be using case weights
    in your particular application, you can ignore this input entirely.
  
  Returns 0 or 1.
    0: No errors occurred when fitting leaf node model
    1: An error occurred when fitting the leaf node model (probably due to insufficient data)
  If fit returns 1, then the tree will not consider the split that led to this leaf node model
  
  fit_init is a LeafModel object which represents a previously-trained leaf node model.
  If specified, fit_init is used for initialization when training this current LeafModel object.
  Useful for faster computation when fit_init's coefficients are close to the optimal solution of the new data.
  
  For those interested in defining their own leaf node functions:
    (1) It is not required to use the fit_init argument in your code
    (2) All edge cases must be handled in code below (ex: arguments
        consist of a single entry, weights are all zero, Y has one unique choice, etc.).
        In these cases, either hard-code a model that works with these edge-cases (e.g., 
        if all Ys = 1, predict 1 with probability one), or have the fit function return 1 (error)
    (3) Store the fitted model as an attribute to the self object. You can name the attribute
        anything you want (i.e., it does not have to be self.model_obj and self.model_coef below),
        as long as its consistent with your predict_prob() and to_string() methods
        
  Any additional args passed to mtp's fit() function are directly passed here
  '''
  def fit(self, A, Y, weights, fit_init=None, refit=False, SPO_loss_bound=None, MSE_loss_bound=None, **kwargs):    
    #no need to refit this model since it is already fit to optimality
    #note: change this behavior if debias=TRUE
    if refit == True:
      return(0)
      
    self.SPO_loss_bound = SPO_loss_bound
    self.MSE_loss_bound = MSE_loss_bound
    
    def fast_row_avg(X,weights):
      return (np.matmul(weights,X)/sum(weights)).reshape(-1)
    
    #if no observations are mapped to this leaf, then assign any feasible cost vector here 
    if sum(weights) == 0:
      self.mean_cost = np.ones(get_num_decisions(**kwargs))
    else:
      self.mean_cost = fast_row_avg(Y,weights)
    self.decision = find_opt_decision(self.mean_cost.reshape(1,-1),**kwargs)['weights'].reshape(-1)
    
    return(0)
    
  '''
  This function applies model from fit() to predict choice data given new data A.
  Returns a list/numpy array of choices (one list entry per observation, i.e. l[i] yields prediction for ith obs.).
  Note: make sure to call fit() first before this method.
  
  Any additional args passed to mtp's predict() function are directly passed here
  '''
  def predict(self, A, get_cost=False, *args,**kwargs):
    if get_cost==True:
      #Returns predicted cost corresponding to this leaf node
      return np.array([self.mean_cost]*len(A))
    else:
      #Returns predicted decision corresponding to this leaf node
      return np.array([self.decision]*len(A))
  '''
  This function outputs the errors for each observation in pair (A,Y).  
  Used in training when comparing different tree splits.
  Ex: mean-squared-error between observed data Y and predict(A)
  
  How to pass additional arguments to this function: simply pass these arguments to the init()/fit() functions and store them
  in the self object.
  '''
  def error(self,A,Y):
    def MSEloss(C,Cpred):
      #return distance.cdist(C, Cpred, 'sqeuclidean').reshape(-1)
      MSE = (C**2).sum(axis=1)[:, None] - 2 * C.dot(Cpred.transpose()) + ((Cpred**2).sum(axis=1)[None, :])
      return MSE.reshape(-1)
    
    def SPOloss(C,decision):
      return np.matmul(C,decision).reshape(-1)
    
    if self.SPO_weight_param == 1.0:
      if self.SPO_full_error == True:
        SPO_loss = SPOloss(Y,self.decision) - A
      else:
        SPO_loss = SPOloss(Y,self.decision)
      return SPO_loss
    elif self.SPO_weight_param == 0.0:
      MSE_loss = MSEloss(Y, self.mean_cost.reshape(1,-1))
      return MSE_loss
    else:
      if self.SPO_full_error == True:
        SPO_loss = SPOloss(Y,self.decision) - A
      else:
        SPO_loss = SPOloss(Y,self.decision)
      MSE_loss = MSEloss(Y, self.mean_cost.reshape(1,-1))
      return self.SPO_weight_param*SPO_loss/self.SPO_loss_bound+(1.0-self.SPO_weight_param)*MSE_loss/self.MSE_loss_bound
  
  '''
  This function outputs the errors for each observation in pair (A,Y).  
  Used in pruning to determine the best tree subset.
  Ex: mean-squared-error between observed data Y and predict(A)
  
  How to pass additional arguments to this function: simply pass these arguments to the init()/fit() functions and store them
  in the self object.
  '''
  def error_pruning(self,A,Y):
    return self.error(A,Y)
  
  '''
  This function returns the string representation of the fitted model
  Used in traverse() method, which traverses the tree and prints out all terminal node models
  
  Any additional args passed to mtp's traverse() function are directly passed here
  '''
  def to_string(self,*leafargs,**leafkwargs):
    return "Mean cost vector: \n" + str(self.mean_cost) +"\n"+"decision: \n"+str(self.decision)
    

'''
Given attribute data A, choice data Y, and observation indices data_inds,
extract those observations of A and Y corresponding to data_inds

If only attribute data A is given, returns A.
If only choice data Y is given, returns Y.

Used to partition the data in the tree-fitting procedure
'''
def get_sub(data_inds,A=None,Y=None,is_boolvec=False):
  if A is None:
    return Y[data_inds]
  if Y is None:
    return A[data_inds]
  else:
    return A[data_inds],Y[data_inds]

'''
This function takes as input choice data Y and outputs a boolean corresponding
to whether all of the choices in Y are the same. 

It is used as a test for whether we should make a node a leaf. If are_Ys_diverse(Y)=False,
then the node will become a leaf. Otherwise, if the node passes the other tests (doesn't exceed
max depth, etc), we will consider splitting on the node.
'''
def are_Ys_diverse(Y):
  #return False iff all cost vectors (rows of Y) are the same
  tmp = [len(np.unique(Y[:,j])) for j in range(Y.shape[1])]
  return (np.max(tmp) > 1)

