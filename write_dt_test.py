from sklearn.tree import DecisionTreeClassifier
import numpy as np

def traverse_tree(tree, feature_names, node=0):
    if tree.feature[node] != -2:  # Internal node
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        
        # Generate combinational logic for the internal node
        logic = f"({feature} <= {threshold}) ? "
        left_subtree = traverse_tree(tree, feature_names, tree.children_left[node])
        right_subtree = traverse_tree(tree, feature_names, tree.children_right[node])
        
        return logic + left_subtree + " : " + right_subtree
    else:  # Leaf node
        value = tree.value[node]
        prediction = np.argmax(value)
        return str(prediction)


def decision_tree_to_verilog(decision_tree, feature_names):
    print("module DecisionTree(")
    print("  input", ", ".join([f"{name}" for name in feature_names]) + ",")
    print("  output prediction")
    print(");")
    print()

    tree = decision_tree.tree_
    logic_expression = traverse_tree(tree, feature_names)
    
    print(f"  assign prediction = {logic_expression};")
    print("endmodule")


# Example Usage
if __name__ == "__main__":
    # Dummy data and decision tree classifier for demonstration
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 0, 1]
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Feature names corresponding to X
    feature_names = ['feature1', 'feature2']
    
    # Generate Verilog code
    decision_tree_to_verilog(clf, feature_names)
