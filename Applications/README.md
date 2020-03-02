# Applications

This folder contains all code for reproducing the three numerical experiments (applications) covered in the paper:
* Illustrative Example: A two-road shortest path decision problem.
* Shortest Path: A shortest path decision problem over a 4 x 4 grid network, where driver starts in
southwest corner and tries to find shortest path to northeast corner.
* Yahoo News: A news article recommendation decision problem constructed from the Yahoo! Front Page Today Module dataset.

Datasets used in the numerical experiments may be found here:
* Shortest Path: https://archive.org/details/spotree_shortestpathdata
* Yahoo News: The Yahoo! Front Page Today dataset may be found at https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=49. A license must be obtained from Yahoo to access the dataset. The data may be used only for academic research purposes and may not be used for any commercial purposes or by any commercial entity. Preprocessing scripts are included to format the raw dataset to match the one used in our numerical experiments.

To reproduce the numerical experiments, merge into a single folder all codes in the Algorithms folder with the codes + data files (unzipped) corresponding to the application of interest. Then, run the relevant application Python script.

The headers of the application scripts contains all experimental parameter settings used in the paper.

Code currently only supports Python 2.7 (not Python 3).
Package Dependencies: gurobipy (with valid Gurobi license), numpy, pandas, scipy, joblib
* The Illustrative Example application also depends on matplotlib
