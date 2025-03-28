import joblib
import os
import numpy as np
from sklearn.tree._tree import TREE_LEAF
from sklearn.metrics import accuracy_score
import itertools
import copy
import sys

def main(dataset, FS_name):

    folder_path = f'./DT/{dataset}/{FS_name}'
    floating_path = folder_path+'/float_models'
    pruned_path = folder_path+'/pruning'

    if not os.path.exists(pruned_path):
        os.makedirs(pruned_path)


    feature_sizes = [5, 10, 15, 20, 25, 30]
    models = {}
    for num_f in feature_sizes:
        model_path = f"{floating_path}/{num_f}.joblib"
        dt_model = joblib.load(model_path)
        models[num_f] = dt_model
    X_train = np.load(os.path.join(floating_path, f"X_train_{feature_sizes[-1]}.npy"))
    X_test = np.load(os.path.join(floating_path, f"X_test_{feature_sizes[-1]}.npy"))
    y_train = np.load(os.path.join(floating_path, "y_train.npy"))
    y_test = np.load(os.path.join(floating_path, "y_test.npy"))

    np.save(f"{pruned_path}/X_train_{feature_sizes[-1]}.npy",X_train)
    np.save(f"{pruned_path}/X_test_{feature_sizes[-1]}.npy",X_test)
    np.save(f"{pruned_path}/y_train.npy",y_train)
    np.save(f"{pruned_path}/y_test.npy",y_test)


    def prune_tree(decision_tree, threshold):
        def prune_node(node):
            if decision_tree.tree_.children_left[node] == TREE_LEAF:
                return

            # Prune the subtrees
            left_child = decision_tree.tree_.children_left[node]
            right_child = decision_tree.tree_.children_right[node]
            prune_node(left_child)
            prune_node(right_child)
            def is_leaf(node):
                return (decision_tree.tree_.children_left[node] == TREE_LEAF and decision_tree.tree_.children_right[node] == TREE_LEAF)

            if is_leaf(left_child) and is_leaf(right_child):
                # Compute impurity decrease
                impurity = decision_tree.tree_.impurity[node]
                impurity_left = decision_tree.tree_.impurity[left_child]
                impurity_right = decision_tree.tree_.impurity[right_child]
                n_samples = decision_tree.tree_.n_node_samples[node]
                n_samples_left = decision_tree.tree_.n_node_samples[left_child]
                n_samples_right = decision_tree.tree_.n_node_samples[right_child]

                impurity_decrease = (
                    impurity - (n_samples_left * impurity_left + n_samples_right * impurity_right) / n_samples
                )

                if impurity_decrease < threshold:
                    decision_tree.tree_.children_left[node] = TREE_LEAF
                    decision_tree.tree_.children_right[node] = TREE_LEAF
                    print(f"pruned node {node} with impurity decrease {impurity_decrease}")

        prune_node(0)



    # threshold : impurity decrease
    threshold = 0.5
    pruned_models = {}

    def count_internal_nodes(decision_tree):
        internal_nodes = 0
        for i in range(decision_tree.tree_.node_count):
            if (decision_tree.tree_.children_left[i] != TREE_LEAF or
                decision_tree.tree_.children_right[i] != TREE_LEAF):
                internal_nodes += 1
        return internal_nodes

    for num_f, dt_model in models.items():
        # from sklearn.base import clone
        pruned_dt_model = copy.deepcopy(dt_model)
        n_nodes_before = pruned_dt_model.tree_.node_count
        internal_nodes_before = count_internal_nodes(pruned_dt_model)
        depth_before = pruned_dt_model.tree_.max_depth


        prune_tree(pruned_dt_model, threshold)
        n_nodes_after = pruned_dt_model.tree_.node_count
        internal_nodes_after = count_internal_nodes(pruned_dt_model)
        depth_after = pruned_dt_model.tree_.max_depth
        print(f"Feature Size: {num_f}, Nodes before pruning: {n_nodes_before}, Nodes after pruning: {n_nodes_after}")
        print(f"depth_before{depth_before}")
        print(f"depth_after{depth_after}")
        print(f"internal_nodes_after : {internal_nodes_after}")
        print(f"internal_nodes_before: {internal_nodes_before}")



        X_test_fs = X_test[:, :num_f]
        y_pred = pruned_dt_model.predict(X_test_fs)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Feature Size: {num_f}, Accuracy after pruning: {accuracy:.4f}")
        pruned_model_path = f"{pruned_path}/{num_f}.joblib"
        joblib.dump(pruned_dt_model, pruned_model_path)
        pruned_models[num_f] = pruned_dt_model

    f=open(f"{pruned_path}/pruning_log.txt", 'w')
    stdbackout = sys.stdout
    sys.stdout = f
    print(f"Model;Accuracy;Feature_size;")
    for num_f in feature_sizes:
        X_test_fs = X_test[:, :num_f]
        raw_model = models[num_f]
        pruned_model = pruned_models[num_f]

    #  raw model
        y_pred_raw = raw_model.predict(X_test_fs)
        accuracy_raw = accuracy_score(y_test, y_pred_raw)

        y_pred_pruned = pruned_model.predict(X_test_fs)
        accuracy_pruned = accuracy_score(y_test, y_pred_pruned)

        
        print(f"Raw;{num_f};{accuracy_raw};")
        print(f"Pruned;{num_f};{accuracy_pruned};")
        
    sys.stdout=stdbackout
    f.close()

if __name__=="__main__":

    dataset=['wesad','spd']
    FS_name = ['jmi', 'disr','fisher_score']

    combinations = list(itertools.product(dataset, FS_name))

    for combo in combinations:
        main(combo[0], combo[1])