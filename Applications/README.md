# Applications

This folder contains all code for reproducing the three numerical experiments (applications) covered in the paper:
* Illustrative Example: A two-road shortest path decision problem.
* Shortest Path: A shortest path decision problem over a 4 x 4 grid network, where driver starts in
southwest corner and tries to find shortest path to northeast corner.
* Yahoo News: A news article recommendation decision problem constructed from the Yahoo! Front Page Today Module dataset.

Datasets used in the numerical experiments may be found here:
* Shortest Path: https://archive.org/details/spotree_data_shortestpath
* Yahoo News: https://archive.org/details/spotree_data_yahoonews

To reproduce the numerical experiments, merge into a single folder all codes in the Algorithms folder with the codes + data files (unzipped) corresponding to the application of interest. Then, run the relevant application Python script.

The headers of the application scripts contains all experimental parameter settings used in the paper.

Code currently only supports Python 2.7 (not Python 3).
Package Dependencies: gurobipy (with valid Gurobi license), numpy, pandas, scipy, joblib
* The Illustrative Example application also depends on matplotlib
