# before running this file, data should be extracted as csv file and be stored under ../data_generation/tables/name.csv

import pandas as pd
import numpy as np
import pickle # using to save and load object

from sklearn.tree import _tree
from sklearn import tree


# get data from csv, for example "../data_generation/tables/table_ACASXU_run2a_1_1_batch_2000.csv"
# add column header: f0, ..., f4 are the features,
# l_min is the classification label with lowest score, l_max is the classification label with highest score .

def get_data():
    data = pd.read_csv("../data_generation/tables/table_ACASXU_run2a_1_1_batch_2000.csv",
                       names=['f0', 'f1', 'f2', 'f3', 'f4', 'l_min', 'l_max'])
    return data


def tree_to_path(tree, feature_names):
    paths = []
    path = []
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED  else "undefined!"   # no feature name of this node, means leaf nodes.
        for i in tree_.feature
    ]

    def store_path(node, path_a):

        if tree_.feature[node] != _tree.TREE_UNDEFINED: #if not a leaf node, decision conditions like f0 > 0.5.
            name = feature_name[node]
            threshold = tree_.threshold[node]
            path_temp1 = path_a[:]
            path_temp2 = path_a[:]
            # add the constraint to the left
            con_left = "{} <= {}".format(name, threshold)
            path_temp1.append(con_left)
            store_path(tree_.children_left[node], path_temp1)
            # add the constraint to the right
            con_right = "{} > {}".format(name, threshold)
            path_temp2.append(con_right)
            store_path(tree_.children_right[node], path_temp2)

        else:
            path_temp3 = path_a[:]
            # value of scikit learn decision tree means the current number of each classes
            # (2, 0) means there are two element of the first class; (0, 2) means there two element of the second class.
            # the following maps tree leaf value to the corresponding class name.

            # get index that with nonzero value
            index = np.nonzero(tree_.value[node][0])

            leaf_label = tree.classes_[index][0].split(',')  # got format like ['3.0', '1.0']
            label_min = 'lmin == ' + str(leaf_label[0])
            label_max = 'lmax == ' + str(leaf_label[1])
            # add the leaf to the end of the path.
            path_temp3.append(label_min)
            path_temp3.append(label_max)
            paths.append(path_temp3)

    store_path(0,  path)
    return paths


def main():
    data = get_data()
    print("* data.head()", data.head(), sep="\n", end="\n\n")
    print("* data.tail()", data.tail(), sep="\n", end="\n\n")

    features = list(data.columns[:5])  # get feature names

    # l is the combination of classification labels(l_min, l_max)
    data['l'] = data['l_min'].map(str) + ',' + data['l_max'].map(str)

    x = data[features]  # get feature values
    y = data['l']          # get label values

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    # Get tree size (number of nodes)
    print(clf.tree_.node_count)
    # print class names of the tree (unique leaves).
    print(clf.classes_)

    tree_paths = tree_to_path(clf, features)

    # store the tree to a text file, for example for neural network 1_1 save to txt file "tree_paths_11.txt"
    with open("paths_dt/tree_paths_11.txt", "wb") as fp:
        pickle.dump(tree_paths, fp)

    # features only stored into a text file for once.
    with open("features.txt", "wb") as fp:
        pickle.dump(features, fp)

    # get the total number of paths of a decision tree
    print(len(tree_paths))
    return [tree_paths, features]


if __name__ == "__main__":
    main()
