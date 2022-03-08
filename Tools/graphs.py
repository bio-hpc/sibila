#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import tree
import math
from random import randint
# parameters
from sklearn.metrics import roc_curve, auc, roc_auc_score



def create_tree_graph(rf, prefix_out, feature_list, n_estimators):
    return
    # Pull out one tree from the forest
    # for i in rf.estimators_:
    #     print(i)
    tree = rf.estimators_[n_estimators - 1]
    # Export the image to a dot file
    export_graphviz(tree,
                    out_file=prefix_out + '_tree.dot',
                    filled=True,
                    special_characters=True,
                    feature_names=feature_list,
                    rounded=True,
                    precision=1)

    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file(prefix_out + '_tree.dot')

    # Write graph to a png file
    graph.write_png(prefix_out + '_tree.png')
    print(prefix_out + '_tree.dot')

    #create_join_tree_graph(rf, feature_list, prefix_out)


def create_join_tree_graph(rf, feature_list, file_tree):
    if TREE_MULTIGRAPH:
        n_cols = 2
        n_rows = math.ceil(len(rf.estimators_) / n_cols)
        # if len(rf.estimators_) % n_cols == 0:  n_rows += 1
        if len(rf.estimators_) < n_cols:
            n_rows += 1
            n_cols = len(rf.estimators_)

        # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 2), dpi=900)
        fig, axes = plt.subplots(n_rows, n_cols)
        index_y = 0
        for index in range(0, len(rf.estimators_)):
            if index % n_cols == 0 and index != 0:
                index_y += 1
            tree.plot_tree(
                rf.estimators_[index],
                feature_names=feature_list,
                # class_names=cn,
                filled=True,
                ax=axes[index_y][index % n_cols],
            )

            axes[index_y][index % n_cols].set_title(str(index + 1), fontsize=11)
        fig.savefig('{}_{}_trees.svg'.format(file_tree, len(rf.estimators_)))


def grap_histogram(data, labels, out_file):
    """
         Plot the impurity-based feature importances of the forest
    :param data:
    :param labels:
    :param out_file:
    :return:
    """
    plt.clf()
    plt.cla()
    plt.figure()
    plt.margins(0.2)
    plt.title("Feature importances")
    # plt.bar(range(importan.shape[1]), importan[indices],
    #        color="r", yerr=std[indices], align="center")
    plt.bar(range(len(data)), data, color="b", align="center")
    plt.xticks(range(len(data)), labels, rotation=90, fontsize=5)
    plt.xlim([-1, len(data)])

    plt.savefig(out_file, dpi=900)


def graph_tree(model, id_list, claasses, prefix_out):
    """
        :param model: 
        :param id_list: data set columns
        :param claasses:  classification classes
        :param prefix_out:
        :return:        
    """
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model, feature_names=id_list, class_names=claasses, filled=True)
    plt.savefig(prefix_out + "_decistion_tree.png")
