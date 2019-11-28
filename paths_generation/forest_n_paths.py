# This file trains a random forest based on corresponding CSV file, and stores feasible path combinations of each file.

import pandas as pd
from itertools import product
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from toPath_Con import tree_to_path
from z3 import *
import pickle # using to save and load object


# get data from csv and add column header: f0, ..., f4 are the features,
# l_min is the classification label with lowest score, l_max is the classification label with highest score .
def get_data():
    # get data from the corresponding csv file.
    data = pd.read_csv("../data_generation/tables/table_ACASXU_run2a_1_3_batch_2000.csv",
                       names=['f0', 'f1', 'f2', 'f3', 'f4', 'l_min', 'l_max'])
    return data


# get paths from each decision tree.
def forest_to_path(clf, feature_names):

    forests = []
    trees =[]

    def get_forest(i, tree_tmp):
     if i < clf.n_estimators:
        tree = tree_to_path(clf.estimators_[i], feature_names, clf.classes_)
        i += 1
        if i < clf.n_estimators:              # there are still other unexplored decision trees.
            tree_tmp1 = tree_tmp[:]
            tree_tmp1.append(tree)
            get_forest(i, tree_tmp1)

        else:                                 # the current decision tree is the last one of the random forest.
            tree_tmp2 = tree_tmp[:]
            tree_tmp2.append(tree)
            forests.append(tree_tmp2)

    get_forest(0, trees)
    return forests


def path_combing(forest, features):

    catersian_paths =[]
    for dt in product(*forest):
        catersian_paths.append(list(dt))

    feasible_paths = []
    for a in catersian_paths:
      pre_tmp = []
      post_tmp = []     #get lmin

      for b in a:
        pre_tmp += (b[:-2])
        post_tmp.append (b[-2:]) #list of labels of the combined paths


      #pass to z3, to see whether it is a feasible path
      n = 0
      while n < len(features):
        f_name = features[n]
        locals()[f_name] = Real(f_name)
        n += 1
      lmin = Int('lmin')
      lmax = Int('lmax')

      p = []
      for f in pre_tmp:
          p.append(eval(f))

      #build solver
      s = Solver()
      s. add(p)
      print(s.check())

      if str(s.check()) == "sat":
        print(post_tmp)
#        label = Counter(post_tmp).most_common()[0][0] #get the label with highest frequency
        label = Counter(tuple(x) for x in iter(post_tmp)).most_common()[0][0]
        pre_tmp += label
        feasible_paths.append(pre_tmp) #list with all path combination


    return(feasible_paths)


def main():

    data = get_data()
    print("* data.head()", data.head(), sep="\n", end="\n\n")
    print("* data.tail()", data.tail(), sep="\n", end="\n\n")

    features = list(data.columns[:5])  # get feature names

    # l is the combination of classification labels(l_min, l_max)
    data['l'] = data['l_min'].map(str) + ',' + data['l_max'].map(str)

    x = data[features]  # get feature values
    y = data['l']          # get label values

    rf = RandomForestClassifier(n_estimators=3, max_depth=None) #(15000: 5_1   n_dt: 3, depth:10  acc:90,9%)
    rf = rf.fit(x, y)

    # get paths from each decision tree
    forest_paths = forest_to_path(rf, features)

    # get the feasible path combinations
    combined_paths = path_combing(forest_paths[0], features)

    print(len(combined_paths))
    print(combined_paths)

    # store the combined_paths of certain dataset to a text file.
    with open("paths_rf/combined_paths_1_3.txt", "wb") as fp:
        pickle.dump(combined_paths, fp)

    return [combined_paths, features]


if __name__ == "__main__":
    main()
