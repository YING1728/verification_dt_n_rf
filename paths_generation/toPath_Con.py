from sklearn.tree import _tree
import numpy as np


def tree_to_path(tree, feature_names, classes):
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

            # get index that with nonzero value
            index = np.nonzero(tree_.value[node][0])

            leaf_label = classes[index][0].split(',')  # got format like ['3.0', '1.0']

            label_min = 'lmin == ' + str(leaf_label[0])
            label_max = 'lmax == ' + str(leaf_label[1])
            # add the leaf to the end of the path.
            path_temp3.append(label_min)
            path_temp3.append(label_max)
            paths.append(path_temp3)

    store_path(0,  path)
    return paths