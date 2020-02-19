'''
Generic file to set up the decision problem (i.e., optimization problem) under consideration
Must have functions: 
  get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector)
  find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors

This particular file sets up a shortest path decision problem over a 4 x 4 grid network, where driver starts in
southwest corner and tries to find shortest path to northeast corner.
'''

from gurobipy import *
import numpy as np

dim = 4 #(creates dim * dim grid, where dim = number of vertices)
Edge_list = [(i,i+1) for i in range(1, dim**2 + 1) if i % dim != 0]
Edge_list += [(i, i + dim) for i in range(1, dim**2 + 1) if i <= dim**2 - dim]
Edge_dict = {} #(assigns each edge to a unique integer from 0 to number-of-edges)
for index, edge in enumerate(Edge_list):
    Edge_dict[edge] = index
D = len(Edge_list) # D = number of decisions

def get_num_decisions():
  return D

Edges = tuplelist(Edge_list)
# Find the optimal total cost for an observation in the context of shortes path
m_shortest_path = Model('shortest_path')
m_shortest_path.Params.OutputFlag = 0
flow = m_shortest_path.addVars(Edges, ub = 1, name = 'flow')
m_shortest_path.addConstrs((quicksum(flow[i,j] for i,j in Edges.select(i,'*')) - quicksum(flow[k, i] for k,i in Edges.select('*',
  i)) == 0 for i in range(2, dim**2)), name = 'inner_nodes')
m_shortest_path.addConstr((quicksum(flow[i,j] for i,j in Edges.select(1, '*')) == 1), name = 'start_node')
m_shortest_path.addConstr((quicksum(flow[i,j] for i,j in Edges.select('*', dim**2)) == 1), name = 'end_node')

def shortest_path(cost):
    # m_shortest_path.setObjective(quicksum(flow[i,j] * cost[Edge_dict[(i,j)]] for i,j in Edges), GRB.MINIMIZE)
    m_shortest_path.setObjective(LinExpr( [ (cost[Edge_dict[(i,j)]],flow[i,j] ) for i,j in Edges]), GRB.MINIMIZE)
    m_shortest_path.optimize()
    return {'weights': m_shortest_path.getAttr('x', flow), 'objective': m_shortest_path.objVal}

def find_opt_decision(cost):
    weights = np.zeros(cost.shape)
    objective = np.zeros(cost.shape[0])
    for i in range(cost.shape[0]):
        temp = shortest_path(cost[i,:])
        for edge in Edges:
            weights[i, Edge_dict[edge]] = temp['weights'][edge]
        objective[i] = temp['objective']
    return {'weights': weights, 'objective':objective}
