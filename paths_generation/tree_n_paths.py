# This file trains a decision tree based on the corresponding CSV file, extracts and stores paths of that decision tree into a text file.
# before running this file, data should be extracted as csv file and be stored under ../data_generation/tables/name.csv

import pandas as pd
import pickle # using to save and load object
from sklearn import tree
from toPath_Con import tree_to_path


# get data from csv, for example "../data_generation/tables/table_ACASXU_run2a_1_1_batch_2000.csv"
# add column header: f0, ..., f4 are the features,
# l_min is the classification label with lowest score, l_max is the classification label with highest score .

def get_data():
    data = pd.read_csv("../data_generation/tables/table_ACASXU_run2a_1_1_batch_2000.csv",
                       names=['f0', 'f1', 'f2', 'f3', 'f4', 'l_min', 'l_max'])
    return data


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

    tree_paths = tree_to_path(clf, features, clf.classes_)

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
