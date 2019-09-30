# verification_dt_n_rf
This project is related to the master thesis: Verification of Random Forests: A Case Study of ACAS Xu.

## data_generation 
In this folder, 45 nnet files for 45 deep neural networks are stored in the subfolder 'nnet'. 'util.py' and 'reconstruct_table.py' are used to create samples by interpreting neural networks based on nnet files. Sampls are stored as csv files in the subfoler 'tables'.

## paths_generation
 'tree_n_paths.py' reads the data from the csv files, trains a decision tree, and by invoking 'toPath_Con.py' to store paths of the decision tree into a text file. The folder 'paths_dt' contains 45 text file for all 45 neural networks.
 
 'forest_n_paths.py' reads the data from the csv files, trains a random forest with three decision trees, gets all paths of all decision trees of the random forest by invoking 'toPath_Con.py', does Cartersian product of all the paths of all decision trees, and get feasible path combinations by examing all paths of the Cartersian product using Z3. Because of the path explosion problem, only eight neural networks have stored feasible path combinations. They are stored in 'paths_rf' folder.
 
 ## verification
 
'to_solver.py' includes the function to perform property checking on paths or path combinations.

'property_verifiyn.py', where n presents the number of property, which is from 2 to 10, are the verfication codes for decision trees.

'property_verifiym_rf.py', where m presents the number of perperty, which is an elment from (2, 3, 4, 8,9), are the verification codes for random forests. For property 5, 6, 7 and 10, there is no path combinations to verify as a result of path explosion. 

