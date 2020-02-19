# Algorithms

This folder contains implementations for training (greedy) SPO Trees (SPO_tree_greedy.py) and SPO Forests (SPOForest.py) for general predict-then-optimize problems. The implementation of the SPO Tree MILP approach is tailored to the specific applications of the paper and therefore is provided in the relevant applications folder.

The SPO Tree / SPO Forest classes consist of the following methods:
* \_\_init\_\_(): initializes the algorithm
* fit(): trains the algorithm on data: contextual features X, cost vectors C
* traverse(): prints out the learned tree/forest
* prune(): Not implemented for SPO Forests. Prunes the SPO Tree on a held-out validation set to prevent overfitting. Applies the CART pruning method.
* est_decision(): outputs estimated optimal decisions given new contextual features Xnew
* est_cost(): outputs estimated cost vectors given new contextual features Xnew

Further documentation is provided in the headers of SPO_tree_greedy.py and SPOForest.py.

The structure of the decision-making problem of interest should be encoded in a file called decision_problem_solver.py. This file should contain two functions specified by the practitioner:
* get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector for underlying decision problem)
* find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors.

Any (optional) arguments for these functions may be passed as keyword arguments to the fit() functions of the SPO Tree/Forest classes. An example is given in the Yahoo News application. The shortest path applications provide an additional example of the specification of decision_problem_solver.py.

Code currently only supports Python 2.7 (not Python 3).
Package Dependencies: gurobipy (with valid Gurobi license), numpy, pandas, scipy, joblib
