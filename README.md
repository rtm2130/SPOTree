# SPO Tree

This is a Python code base for training "SPO Trees" (SPOTs) from the paper "Decision Trees for Decision-Making under the Predict-then-Optimize Framework".

The "Algorithms" folder contains implementations for training (greedy) SPO Trees (SPO_tree_greedy.py) and SPO Forests (SPOForest.py) for general predict-then-optimize problems.

The "Applications" folder contains all data + code for reproducing the three numerical experiments (applications) covered in the paper:
* Illustrative Example: A two-road shortest path decision problem.
* Shortest Path: A shortest path decision problem over a 4 x 4 grid network, where driver starts in
southwest corner and tries to find shortest path to northeast corner.
* Yahoo News: A news article recommendation decision problem constructed from the Yahoo! Front Page Today Module dataset.

Code currently only supports Python 2.7 (not Python 3).
Package Dependencies: gurobipy (with valid Gurobi license), numpy, pandas, scipy, joblib
* The Illustrative Example application also depends on matplotlib
