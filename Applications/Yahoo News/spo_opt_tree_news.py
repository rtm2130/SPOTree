'''
Code for fitting SPOT MILP on Yahoo News dataset.
Note that this code is specifically designed for the Yahoo News application and
will not work for other applications without modifying the constraints. 
Note on notation: the paper uses r to denote the binary variables which map training observations to leaves. This code uses z rather than r.
'''


import numpy as np
import pandas as pd
from gurobipy import*
import pickle
from sklearn.model_selection import KFold
from decision_problem_solver import*

# Helper functions for tree structure
def find_parent_index(t):
  return (t+1)//2 - 1

def find_ancestors(t):
    l= []
    r = []
    if t == 0:
        return
    else:
        while find_parent_index(t) !=0:
            parent = find_parent_index(t)
            if (t+1)% (1+parent) ==1:
                r.append(parent)
            else:
                l.append(parent)
            t = parent
        if t==2:
            r.append(0)
        else:
            l.append(0)
    return[l,r]

#truncate training set features to desired precision    
def truncate_train_x(train_x, train_x_precision):
  return(np.around(train_x, decimals=train_x_precision))
  

#trains an optimal tree model on train_cost, train_x, and reg parameter spo_opt_tree_reg (scalar)
#returns parameter encoding of the opimal tree (a,b,w). (a,b) encode splits, (w) encode leaf decisions
#optimal tree params:
#N_min = minimum number of observations per leaf node
#H = max tree depth
#def spo_opt_tree(train_cost, train_x,spo_opt_tree_reg):
def spo_opt_tree(train_cost, train_x, train_x_precision, spo_opt_tree_reg, N_min, H, 
                 weights=None, 
                 returnAllOptvars=False,
                 a_start=None, b_start=None, w_start=None, y_start=None, z_start=None, l_start=None, d_start=None, 
                 threads=None, MIPGap=None, MIPFocus=None, verbose=False, Seed=None, TimeLimit=None,
                 Presolve=None, ImproveStartTime=None, VarBranch=None, Cuts=None, 
                 tune=False, TuneCriterion=None, TuneJobs=None, TuneTimeLimit=None, TuneTrials=None, tune_foutpref=None,
                 A_constr=None, b_constr=None, l_constr=None, u_constr=None):
    
    ################################333
    #assert(spo_opt_tree_reg >= 1e-4)
    ################################333
    assert(A_constr is not None)
    assert(b_constr is not None)
    assert(l_constr is not None)
    assert(u_constr is not None)
    assert(weights is not None)
    
    #currently only lower bound = 0, upper bound = 1 constraints supported
    assert(len(np.unique(l_constr)) == 1 and np.unique(l_constr)[0] == 0)
    assert(len(np.unique(u_constr)) == 1 and np.unique(u_constr)[0] == 1)
    
    num_constr, D = A_constr.shape
    
    # We label all nodes of the tree by 0, 1, 2, ... 2**(H+1) - 1.
    T_B = 2**H - 1
    T_L = 2**H
  
    n_train, P = train_x.shape
    #truncate x features so eps (below) will not be too small
    train_x = truncate_train_x(train_x, train_x_precision)
    
    assert(np.all(train_x >= 0))
    assert(np.all(train_x <= 1))
    assert(np.all(train_cost <= 0)) #assert nonpositive costs
    assert(np.all(train_cost.shape[0] == train_x.shape[0]))
    # Instantiate optimization model
    # Compute average optimal cost across all training set observations
    # (Although irrelevant for the optimization problem, it helps in interpreting alpha)
    optimal_costs = np.zeros(train_x.shape[0])
    for i in range(train_x.shape[0]):
      optimal_costs[i] = find_opt_decision(train_cost[i,:].reshape(1,-1),A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)['objective'][0]
    if weights is not None:
      sum_optimal_cost = np.dot(optimal_costs, weights)
    else:
      sum_optimal_cost = sum(optimal_costs)
    
    # Compute big M constant
    negM = 0
    for i in range(train_x.shape[0]):
      min_decision_cost = find_opt_decision(train_cost[i,:].reshape(1,-1),A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)['objective'][0]
      if min_decision_cost <= negM:
        negM = min_decision_cost
    
    #M = train_cost.max()*(dim-1)*2
    spo = Model('spo_opt_tree')
    if verbose == False:
      spo.Params.OutputFlag = 0
    #compute epsilon constants
    #eps = np.float("inf")
    #for j in range(train_x.shape[1]):
      #ordered_feat = np.sort(train_x[:,j])
      #diffs = ordered_feat[1:]-ordered_feat[:-1]
      #nonzero_diffs = diffs[diffs > 0]
      #if min(nonzero_diffs) <= eps:
        #eps = min(nonzero_diffs)
    #one_plus_eps = 1 + eps
    
    eps = np.array([np.float("inf")]*train_x.shape[1])
    for j in range(train_x.shape[1]):
      ordered_feat = np.sort(train_x[:,j])
      diffs = ordered_feat[1:]-ordered_feat[:-1]
      nonzero_diffs = diffs[diffs > 0]
      eps[j] = min(nonzero_diffs)
    one_plus_eps_max = 1 + max(eps) 
    
    #run params
    if threads is not None:
      spo.Params.Threads = threads
    if MIPGap is not None:
      spo.Params.MIPGap = MIPGap # default = 1e-4, try 1e-2
    if MIPFocus is not None:
      spo.Params.MIPFocus = MIPFocus
    if Seed is not None:
      spo.Params.Seed = Seed
    if TimeLimit is not None:
      spo.Params.TimeLimit = TimeLimit
    if Presolve is not None:
      spo.Params.Presolve = Presolve
    if ImproveStartTime is not None:
      spo.Params.ImproveStartTime = ImproveStartTime
    if VarBranch is not None:
      spo.Params.VarBranch = VarBranch
    if Cuts is not None:
      spo.Params.Cuts = Cuts
    
    #tune params
    if tune == True and TuneCriterion is not None:
      spo.Params.TuneCriterion = TuneCriterion
    if tune == True and TuneJobs is not None:
      spo.Params.TuneJobs = TuneJobs
    if tune == True and TuneTimeLimit is not None:
      spo.Params.TuneTimeLimit = TuneTimeLimit
    if tune == True and TuneTrials is not None:
      spo.Params.TuneTrials = TuneTrials

    # Add variables
    y = spo.addVars(tuplelist([(i, t) for i in range(n_train) for t in range(T_L)]), lb = -GRB.INFINITY, ub = 0, name = 'y')
    z = spo.addVars(tuplelist([(i, t) for i in range(n_train) for t in range(T_L)]), vtype=GRB.BINARY, name = 'z')
    #z = spo.addVars(tuplelist([(i, t) for i in range(n_train) for t in range(T_L)]), lb = 0, ub = 1, name = 'z')
    w = spo.addVars(tuplelist([(t, j) for t in range(T_L) for j in range(D)]), lb = 0, ub= 1,name = 'w')
    l = spo.addVars(tuplelist([i for i in range(T_L)]), vtype=GRB.BINARY, name = 'l')
    d = spo.addVars(tuplelist([i for i in range(T_B)]), vtype=GRB.BINARY, name = 'd')
    a = spo.addVars(tuplelist([(j,t) for j in range(P) for t in range(T_B)]), vtype=GRB.BINARY, name = 'a')
    #b = spo.addVars(tuplelist([i for i in range(T_B)]), lb = 0, name = 'b')
    b = spo.addVars(tuplelist([i for i in range(T_B)]), lb = 0, ub = 1, name = 'b')
    
    if a_start is not None:
      for i in range(P):
        for j in range(T_B):
          a[i,j].start = a_start[i,j]
    
    if b_start is not None:
      for i in range(T_B):
        b[i].start = b_start[i]
        
    if w_start is not None:
      for i in range(T_L):
        for j in range(D):
          w[i,j].start = w_start[i,j]
    
    if y_start is not None:
      for i in range(n_train):
        for j in range(T_L):
          y[i,j].start = y_start[i,j]
    
    if z_start is not None:
      for i in range(n_train):
        for j in range(T_L):
          z[i,j].start = z_start[i,j]
    
    if l_start is not None:
      for i in range(T_L):
        l[i].start = l_start[i]
    
    if d_start is not None:
      for i in range(T_B):
        d[i].start = d_start[i]
    
    spo.update() #for initial values to be written immediately
    
#    if a_start is not None:
#      for i in range(P):
#        for j in range(T_B):
#          print(a[i,j].start)
#    
#    if b_start is not None:
#      for i in range(T_B):
#        print(b[i].start)
#        
#    if w_start is not None:
#      for i in range(T_L):
#        for j in range(D):
#          print(w[i,j].start)
#    
#    if y_start is not None:
#      for i in range(n_train):
#        for j in range(T_L):
#          print(y[i,j].start)
#    
#    if z_start is not None:
#      for i in range(n_train):
#        for j in range(T_L):
#          print(z[i,j].start)
#    
#    if l_start is not None:
#      for i in range(T_L):
#        print(l[i].start)
#    
#    if d_start is not None:
#      for i in range(T_B):
#        print(d[i].start)
    

    # Add constraints
    # Const
    for i in range(n_train):
        for t in range(T_L):
            #expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for key,j in Edge_dict.items()])
            expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for j in range(D)])
            spo.addConstr(y[i,t] >= expr_constraint)
            spo.addConstr(y[i,t] >= negM * z[i,t])
            # spo.addConstr(y[i,t] >= quicksum(train_cost[i,j] * w[t,j] for key,j in Edge_dict.items())- M * (1 - z[i,t]))

    # # (genreral constraint for feasibility of nominal problem Aw <= B)
    # for t in range(T_L):
    #     for i in range(K):
    #         spo.addConstr(quicksum(A[i,j] * w[t,j] for j in range(D)) <= B[i] )

    # Const
    #constraint for feasibility of decision problem)
    for t in range(T_L):
      spo.addConstrs((quicksum(A_constr[i][j]*w[t,j] for j in range(D)) <= b_constr[i] for i in range(num_constr)))
      spo.addConstr(quicksum(w[t,j] for j in range(D)) == 1)

    # Const
    for i in range(n_train):
        # spo.addConstr(quicksum(z[i,t] for t in range(T_L)) == 1)
        spo.addConstr(LinExpr([(1,z[i,t]) for t in range(T_L)]) == 1)

    # Const
    for i in range(n_train):
        for t in range(T_L):
            spo.addConstr(z[i,t] <= l[t])

    # Const
    for t in range(T_L):
        # spo.addConstr(quicksum(z[i,t] for i in range(n_train))>= N_min * l[t])
        if weights is not None:
          spo.addConstr(LinExpr([(weights[i],z[i,t]) for i in range(n_train)])>= N_min * l[t])
        else:
          spo.addConstr(LinExpr([(1,z[i,t]) for i in range(n_train)])>= N_min * l[t])

    # Const
    for i in range(n_train):
        for t in range(T_L):
            left, right = find_ancestors(t + T_B)
            for m in right:
                # spo.addConstr(quicksum(a[p,m]* train_x[i,p] for p in range(P)) >= b[m]- (1 - z[i,t] ))
                spo.addConstr(LinExpr([(train_x[i,p],a[p,m]) for p in range(P)])  >= b[m]- (1 - z[i,t] ))

    # Const
            for m in left:
                # spo.addConstr(quicksum(a[p,m]* (x[i,p] + eps[p]) for p in range(P))<= b[m] + (1+eps_max)*(1-z[i,t] ))
                #spo.addConstr(quicksum(a[p,m]* train_x[i,p]  for p in range(P)) +0.0001<= b[m] + (1-z[i,t] ))
                #spo.addConstr(LinExpr([(train_x[i,p],a[p,m]) for p in range(P)]) + eps  <= b[m] +  (1+eps)*(1 - z[i,t] ))
                spo.addConstr(LinExpr([(train_x[i,p]+ eps[p],a[p,m]) for p in range(P)]) <= b[m] +  one_plus_eps_max*(1 - z[i,t] ))
                #spo.addConstr(LinExpr([(train_x[i,p],a[p,m]) for p in range(P)]) <= b[m] +  1 - one_plus_eps*z[i,t])

    # Const
    for t in range(T_B):
        # spo.addConstr(quicksum(a[p,t] for p in range(P)) == d[t])
        spo.addConstr(LinExpr([(1,a[p,t]) for p in range(P)]) == d[t])

    # Const
    for t in range(T_B):
        #spo.addConstr(b[t] <= d[t])
        spo.addConstr(b[t] >= 1 - d[t])

    # Const
    for t in range(1,T_B):
        spo.addConstr(d[t] <= d[find_parent_index(t)])
      
    # Const (optional): ensures LP relaxation of problem has y's defined sensibly
    for i in range(n_train):
      spo.addConstr(LinExpr([(1,y[i,t]) for t in range(T_L)]) >= optimal_costs[i])
        #for t in range(T_L):
            #expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for key,j in Edge_dict.items()])
            #expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for j in range(D)])
            #spo.addConstr(expr_constraint >= optimal_costs[i])
            #spo.addConstr(LinExpr([(1,y[i,t]) for t in range(T_L)]) >= optimal_costs[i])

    # Add objective
    # spo.setObjective( quicksum(y[i,t] for i in range(n_train) for t in range(T_L))/n_train + spo_opt_tree_reg* quicksum(d[t] for t in range(T_B) ), GRB.MINIMIZE)
    if weights is not None:
      expr_objective = LinExpr([(weights[i], y[i,t]) for i in range(n_train) for t in range(T_L) ]) - sum_optimal_cost
    else:
      expr_objective = LinExpr([(1, y[i,t]) for i in range(n_train) for t in range(T_L) ]) - sum_optimal_cost
    #expr_objective = LinExpr([(1, y[i,t]) for i in range(n_train) for t in range(T_L) ])
    if spo_opt_tree_reg > 0:
      if weights is not None:
        sum_weights = sum(weights)
        expr_objective.add(LinExpr([(1, d[t]) for t in range(T_B)])*spo_opt_tree_reg*sum_weights)
      else:
        expr_objective.add(LinExpr([(1, d[t]) for t in range(T_B)])*spo_opt_tree_reg*n_train)
    spo.setObjective(expr_objective, GRB.MINIMIZE)
    

    # Solve optimization
    if tune == True:
      spo.tune()
      if tune_foutpref is None:
        tune_foutpref='tune'
      for i in range(spo.tuneResultCount):
        spo.getTuneResult(i)
        spo.write(tune_foutpref+str(i)+'.prm')
    spo.optimize()
    
    #############################33333
#    if spo.status == GRB.OPTIMAL:
#      print("Objective Value:")
#      print(spo.ObjVal)
#      print("Reg term of objective:")
#      if weights is not None:
#        print(spo_opt_tree_reg*sum_weights)
#      else:
#        print(spo_opt_tree_reg*n_train)
#    else:
#      import sys
#      print("Infeasible!")
#      sys.exit("Decision problem infeasible")
    #############################33333
  
    # Get values of objective and variables
    # print('Obj=')
    # print(spo.getObjective().getValue())
    #
    # z_ = np.zeros((n,T_L))
    # z_res = spo.getAttr('X', z)
    # for i,j in z_res:
    #     z_[i,j] = z_res[i,j]
    # print(z_)
    spo_dt_a = np.zeros((P,T_B))
    a_res = spo.getAttr('X', a)
    for i in range(P):
      for j in range(T_B):
        spo_dt_a[i,j] = a_res[i,j]
    #for i,j in a_res:
        #spo_dt_a[i,j] = a_res[i,j]

    spo_dt_b = np.zeros(T_B)
    b_res = spo.getAttr('X', b)
    for i in range(T_B):
      spo_dt_b[i] = b_res[i]
    #for i in b_res:
        #spo_dt_b[i] = b_res[i]

    spo_dt_w = np.zeros((T_L,D))
    w_res = spo.getAttr('X', w)
    for i in range(T_L):
      for j in range(D):
        spo_dt_w[i,j] = w_res[i,j]
    #for i,j in w_res:
        #spo_dt_w[i,j] = w_res[i,j]
    
    spo_dt_y = np.zeros((n_train,T_L))
    y_res = spo.getAttr('X', y)
    for i in range(n_train):
      for j in range(T_L):
        spo_dt_y[i,j] = y_res[i,j]
    spo_dt_z = np.zeros((n_train,T_L))
    z_res = spo.getAttr('X', z)
    for i in range(n_train):
      for j in range(T_L):
        spo_dt_z[i,j] = z_res[i,j]
    spo_dt_l = np.zeros(T_L)
    l_res = spo.getAttr('X', l)
    for i in range(T_L):
      spo_dt_l[i] = l_res[i]
    spo_dt_d = np.zeros(T_B)
    d_res = spo.getAttr('X', d)
    for i in range(T_B):
      spo_dt_d[i] = d_res[i]
    
    if returnAllOptvars == False:
      return spo_dt_a, spo_dt_b, spo_dt_w
    else:
      return spo_dt_a, spo_dt_b, spo_dt_w, spo_dt_y, spo_dt_z, spo_dt_l, spo_dt_d

# Given a tree defined by a and b for all interior nodes, find the path (including the leaf node in which it lies) of observations using its features
def decision_path(x,a,b):
    T_B = len(b)
    if len(x.shape) == 1:
        n = 1
        P = x.size
    else:
        n, P = x.shape
    res = []
    for i in range(n):
        node = 0
        path = [0]
        T_B = a.shape[1]
        while node < T_B:
            if np.dot(x[i,:], a[:,node]) < b[node]:
                node = (node+1)*2 - 1
            else:
                node = (node+1)*2
            path.append(node)
        res.append(path)
    return np.array(res)


# Given the path of an observation (including the leaf node in which it lies), find the predicted total cost for that observation
def apply_leaf_decision(c,path, w, subtract_optimal=False, A_constr=None, b_constr=None, l_constr=None, u_constr=None):
    T_L, D = w.shape
    n = c.shape[0]
    paths = path[:, -1]
    actual_cost = []
    for i in range(n):
        decision_node = paths[i] - T_L +1
        cost_decision = np.dot(c[i,:], w[decision_node,:])
        if subtract_optimal == True:
          cost_optimal = find_opt_decision(c[i,:].reshape(1,-1), A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)['objective'][0]
          actual_cost.append(cost_decision-cost_optimal)
        else:
          actual_cost.append(cost_decision)          
    return np.array(actual_cost)

def spo_opt_traintest(train_x,train_cost,train_weights,test_x,test_cost,test_weights,train_x_precision,spo_opt_tree_reg, N_min, H, A_constr=None, b_constr=None, l_constr=None, u_constr=None):
    spo_dt_a,spo_dt_b, spo_dt_w = spo_opt_tree(train_cost,train_x,train_x_precision,spo_opt_tree_reg, N_min, H, 
                                               weights=train_weights, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
    path = decision_path(test_x,spo_dt_a,spo_dt_b)
    costs = apply_leaf_decision(test_cost,path, spo_dt_w, subtract_optimal=True, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
    return spo_dt_a,spo_dt_b, spo_dt_w, np.dot(costs,test_weights)/np.sum(test_weights)

def spo_opt_tunealpha(train_x,train_cost,train_weights,valid_x,valid_cost,valid_weights,train_x_precision,reg_set, N_min, H, A_constr=None, b_constr=None, l_constr=None, u_constr=None):
    best_err = np.float("inf")
    for alpha in reg_set:
      spo_dt_a,spo_dt_b, spo_dt_w, err = spo_opt_traintest(train_x,train_cost,train_weights,valid_x,valid_cost,valid_weights,train_x_precision,alpha, N_min, H, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)
      if err <= best_err:
        best_spo_dt_a, best_spo_dt_b, best_spo_dt_w, best_err, best_alpha = spo_dt_a,spo_dt_b, spo_dt_w, err, alpha
    return best_spo_dt_a, best_spo_dt_b, best_spo_dt_w, best_err, best_alpha

#def cv_spo_opt_traintest(cost, X, train_x_precision,reg_set, splits = 4):
#    dic = {reg:0 for reg in reg_set}
#    n, n_edges = cost.shape
#    K = X.shape[1]
#    kf = KFold(n_splits = splits)
#    for train, test in kf.split(X):
#        X_train, X_test, cost_train, cost_test = X[train], X[test], cost[train], cost[test]
#        opt_cost = find_opt_decision(cost_test, A_constr=A_constr, b_constr=b_constr, l_constr=l_constr, u_constr=u_constr)['objective']
#        for spo_opt_tree_reg in reg_set:
#            actual_cost = spo_opt_traintest(X_train, cost_train, X_test, cost_test,train_x_precision,spo_opt_tree_reg)
#            dic[spo_opt_tree_reg] += sum(actual_cost - opt_cost)
#    return smallest_dic_value(dic)
#
#def smallest_dic_value(dic):
#    reverse = dict()
#    for key in dic.keys():
#        reverse[dic[key]] = key
#    return reverse[min(reverse.keys())]
