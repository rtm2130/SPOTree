'''
Code for fitting SPOT MILP on shortest paths dataset. 
Note that this code is specifically designed for the shortest paths application and
will not work for other applications without modifying the constraints. 
Note on notation: the paper uses r to denote the binary variables which map training observations to leaves. This code uses z rather than r.
'''


import numpy as np
import pandas as pd
from gurobipy import*
import pickle
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
def spo_opt_tree(train_cost, train_x, train_x_precision, spo_opt_tree_reg, N_min, H, returnAllOptvars=False,
                 a_start=None, b_start=None, w_start=None, y_start=None, z_start=None, l_start=None, d_start=None, 
                 threads=None, MIPGap=None, MIPFocus=None, verbose=False, Seed=None, TimeLimit=None,
                 Presolve=None, ImproveStartTime=None, VarBranch=None, Cuts=None, 
                 tune=False, TuneCriterion=None, TuneJobs=None, TuneTimeLimit=None, TuneTrials=None, tune_foutpref=None):
    assert(spo_opt_tree_reg >= 1e-4)    
    # We label all nodes of the tree by 0, 1, 2, ... 2**(H+1) - 1.
    T_B = 2**H - 1
    T_L = 2**H
    #Edge_list comes from importing shortest_path_solver
    Edges_w_t = tuplelist([(i,j,t) for i,j in Edge_list for t in range(T_L)])
  
    n_train, P = train_x.shape
    #truncate x features so eps (below) will not be too small
    train_x = truncate_train_x(train_x, train_x_precision)
    
    assert(np.all(train_x >= 0))
    assert(np.all(train_x <= 1))
    assert(np.all(train_cost >= 0)) #assert nonnegative costs
    assert(np.all(train_cost.shape[0] == train_x.shape[0]))
    # Instantiate optimization model
    # Compute average optimal cost across all training set observations
    # (Although irrelevant for the optimization problem, it helps in interpreting alpha)
    optimal_costs = np.zeros(train_x.shape[0])
    for i in range(train_x.shape[0]):
      optimal_costs[i] = find_opt_decision(train_cost[i,:].reshape(1,-1))['objective'][0]
    sum_optimal_cost = sum(optimal_costs)
    
    # Compute big M constant
    M = 0
    for i in range(train_x.shape[0]):
      longest_path_cost = -find_opt_decision(-train_cost[i,:].reshape(1,-1))['objective'][0]
      if longest_path_cost >= M:
        M = longest_path_cost
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
    y = spo.addVars(tuplelist([(i, t) for i in range(n_train) for t in range(T_L)]), lb = 0,name = 'y')
    z = spo.addVars(tuplelist([(i, t) for i in range(n_train) for t in range(T_L)]), vtype=GRB.BINARY, name = 'z')
    #z = spo.addVars(tuplelist([(i, t) for i in range(n_train) for t in range(T_L)]), lb = 0, ub = 1, name = 'z')
    w = spo.addVars(tuplelist([(t, j) for t in range(T_L) for j in range(D)]), lb = 0,name = 'w')
    l = spo.addVars(tuplelist([i for i in range(T_L)]), vtype=GRB.BINARY, name = 'l')
    d = spo.addVars(tuplelist([i for i in range(T_B)]), vtype=GRB.BINARY, name = 'd')
    a = spo.addVars(tuplelist([(j,t) for j in range(P) for t in range(T_B)]), vtype=GRB.BINARY, name = 'a')
    #b = spo.addVars(tuplelist([i for i in range(T_B)]), lb = 0, name = 'b')
    b = spo.addVars(tuplelist([i for i in range(T_B)]), ub = 1, name = 'b')
    
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
    # Const 7b
    for i in range(n_train):
        for t in range(T_L):
            #expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for key,j in Edge_dict.items()])
            expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for j in range(D)])
            spo.addConstr(y[i,t] >= expr_constraint - M * (1 - z[i,t]))
            # spo.addConstr(y[i,t] >= quicksum(train_cost[i,j] * w[t,j] for key,j in Edge_dict.items())- M * (1 - z[i,t]))

    # # Const 7c (genreral constraint for feasibility of nominal problem Aw <= B)
    # for t in range(T_L):
    #     for i in range(K):
    #         spo.addConstr(quicksum(A[i,j] * w[t,j] for j in range(D)) <= B[i] )

    # Const 7c (constraint for feasibility of shortest_path problem)
    flow = spo.addVars(Edges_w_t, lb = 0, name = 'flow')
    spo.addConstrs((quicksum(flow[i,j,t] for i,j,t in Edges_w_t.select(i,'*',t)) - quicksum(flow[k,i,t] for k,i,t in Edges_w_t.select('*',i,t)) == 0
                            for i in range(2, dim**2) for t in range(T_L) ))
    spo.addConstrs((quicksum(flow[i,j,t] for i,j,t in Edges_w_t.select(1, '*',t)) == 1 for t in range(T_L)))
    spo.addConstrs((quicksum(flow[i,j,t] for i,j,t in Edges_w_t.select('*', dim**2,t)) == 1 for t in range(T_L)))
    spo.addConstrs( w[t,Edge_dict[(i,j)]] - flow[i,j,t] == 0 for i,j,t in Edges_w_t) # Map shortest path flow to w_t

    # Const 7d
    for i in range(n_train):
        # spo.addConstr(quicksum(z[i,t] for t in range(T_L)) == 1)
        spo.addConstr(LinExpr([(1,z[i,t]) for t in range(T_L)]) == 1)

    # Const 7e
    for i in range(n_train):
        for t in range(T_L):
            spo.addConstr(z[i,t] <= l[t])

    # Const 7f
    for t in range(T_L):
        # spo.addConstr(quicksum(z[i,t] for i in range(n_train))>= N_min * l[t])
        spo.addConstr(LinExpr([(1,z[i,t]) for i in range(n_train)])>= N_min * l[t])

    # Const 7g
    for i in range(n_train):
        for t in range(T_L):
            left, right = find_ancestors(t + T_B)
            for m in right:
                # spo.addConstr(quicksum(a[p,m]* train_x[i,p] for p in range(P)) >= b[m]- (1 - z[i,t] ))
                spo.addConstr(LinExpr([(train_x[i,p],a[p,m]) for p in range(P)])  >= b[m]- (1 - z[i,t] ))

    # Const 7h
            for m in left:
                # spo.addConstr(quicksum(a[p,m]* (x[i,p] + eps[p]) for p in range(P))<= b[m] + (1+eps_max)*(1-z[i,t] ))
                #spo.addConstr(quicksum(a[p,m]* train_x[i,p]  for p in range(P)) +0.0001<= b[m] + (1-z[i,t] ))
                #spo.addConstr(LinExpr([(train_x[i,p],a[p,m]) for p in range(P)]) + eps  <= b[m] +  (1+eps)*(1 - z[i,t] ))
                spo.addConstr(LinExpr([(train_x[i,p]+ eps[p],a[p,m]) for p in range(P)]) <= b[m] +  one_plus_eps_max*(1 - z[i,t] ))
                #spo.addConstr(LinExpr([(train_x[i,p],a[p,m]) for p in range(P)]) <= b[m] +  1 - one_plus_eps*z[i,t])

    # Const 7i
    for t in range(T_B):
        # spo.addConstr(quicksum(a[p,t] for p in range(P)) == d[t])
        spo.addConstr(LinExpr([(1,a[p,t]) for p in range(P)]) == d[t])

    # Const 7j
    for t in range(T_B):
        #spo.addConstr(b[t] <= d[t])
        spo.addConstr(b[t] >= 1 - d[t])

    # Const 7k
    for t in range(1,T_B):
        spo.addConstr(d[t] <= d[find_parent_index(t)])
      
    # Const 7l (optional): ensures LP relaxation of problem has obj >= 0
    for i in range(n_train):
      spo.addConstr(LinExpr([(1,y[i,t]) for t in range(T_L)]) >= optimal_costs[i])
        #for t in range(T_L):
            #expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for key,j in Edge_dict.items()])
            #expr_constraint = LinExpr([ (train_cost[i,j], w[t,j]) for j in range(D)])
            #spo.addConstr(expr_constraint >= optimal_costs[i])
            #spo.addConstr(LinExpr([(1,y[i,t]) for t in range(T_L)]) >= optimal_costs[i])

    # Add objective
    # spo.setObjective( quicksum(y[i,t] for i in range(n_train) for t in range(T_L))/n_train + spo_opt_tree_reg* quicksum(d[t] for t in range(T_B) ), GRB.MINIMIZE)
    expr_objective = LinExpr([(1, y[i,t]) for i in range(n_train) for t in range(T_L) ]) - sum_optimal_cost
    #expr_objective = LinExpr([(1, y[i,t]) for i in range(n_train) for t in range(T_L) ])
    if spo_opt_tree_reg > 0:
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
def apply_leaf_decision(c,path, w, subtract_optimal=False):
    T_L, D = w.shape
    n = c.shape[0]
    paths = path[:, -1]
    actual_cost = []
    for i in range(n):
        decision_node = paths[i] - T_L +1
        cost_decision = np.dot(c[i,:], w[decision_node,:])
        if subtract_optimal == True:
          cost_optimal = find_opt_decision(c[i,:].reshape(1,-1))['objective'][0]
          actual_cost.append(cost_decision-cost_optimal)
        else:
          actual_cost.append(cost_decision)          
    return np.array(actual_cost)

def spo_opt_traintest(train_x,train_cost,test_x,test_cost,train_x_precision,spo_opt_tree_reg, N_min, H):
    spo_dt_a,spo_dt_b, spo_dt_w = spo_opt_tree(train_cost,train_x,train_x_precision,spo_opt_tree_reg, N_min, H)
    path = decision_path(test_x,spo_dt_a,spo_dt_b)
    return spo_dt_a,spo_dt_b, spo_dt_w, np.mean(apply_leaf_decision(test_cost,path, spo_dt_w, subtract_optimal=True))

def spo_opt_tunealpha(train_x,train_cost,valid_x,valid_cost,train_x_precision,reg_set, N_min, H):
    best_err = np.float("inf")
    for alpha in reg_set:
      spo_dt_a,spo_dt_b, spo_dt_w, err = spo_opt_traintest(train_x,train_cost,valid_x,valid_cost,train_x_precision,alpha, N_min, H)
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
#        opt_cost = find_opt_decision(cost_test)['objective']
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
