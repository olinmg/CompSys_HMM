This folder contains 5 code files.

# Graph.py
Implements the HMM class upon which the other code files operate.
Can be used to generate forward simulation examples, load existing values (as from given csv files), and to do message passing on the corresponding clique tree.
In this file also the visualization code for the simulation can be found.
For further code ussage instructions, we reffer to the bottom of the Graph.py file.

# log_reg_analysis.py
Implements the logistic regression, SVM, and small nn forecasting of C_t based on all observed X(t,i) values.
Also plots the accuracies of the fitted models per t.

# Inference Algorithm.py
File "Inference Algorithm.py" is for part II. Here, the path should be replaced  as well. You can do this in the top of the document to acces the csv files.

# Learning of the parameters.py
File "Learning of the parameters.py" is for part III. This code calls functions from "Graph.py", thus the path in "Graph.py" should be changed in regards to above.

# Gradient_ascent.py
File "Gradient_ascent.py" is for part III. This code also loads a file from a path.

The code doesn't require any other setup. Just note that the different files import functions from Graph.py, and remember to check that numpy, pandas, os, re, math, random and scipy is installed and up to date.
