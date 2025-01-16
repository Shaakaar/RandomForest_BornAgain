# MIT License

# Copyright(c) 2020 Toni Pacheco

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pathlib
import persistence
from IPython.display import display
import ipywidgets as widgets


def create_objective_selection(show=True):
    select = widgets.Dropdown(
        options=[('Depth', 0), ('NbLeaves', 1), ('Depth > NbLeaves', 2), ('Heuristic', 4)],
        value=4,
        description='Objective:',
    )
    if show:
        display(select)
    return select


def create_depth_selection(show=True):
    select = widgets.IntSlider(
            value=3,
            min=2,
            max=10,
            step=1,
            description='Max depth:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
    if show:
        display(select)
    return select


def create_n_trees_selection(show=True):
    select = widgets.IntSlider(
        value=10,
        min=3,
        max=500,
        step=1,
        description='#Trees:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    if show:
        display(select)
    return select


def load(X, y, dataset, fold, n_trees, F=None, S=None, return_file=False):
    respath = str(pathlib.Path(__file__).parent.absolute()) + '/resources/forests'
    if F or S:
        filename = '{}.F{}.S{}.RF{}.txt'.format(dataset, F, S, fold)
        filename = '{}/{}/F{}.S{}/{}'.format(respath, dataset, F, S, filename)
    else:
        filename = '{}.RF{}.txt'.format(dataset, fold)
        filename = '{}/{}/{}'.format(respath, dataset, filename)

    clf = persistence.classifier_from_file(filename, X, y, pruning=True, num_trees=n_trees)
    if return_file:
        return clf, filename
    return clf





############################################################## 
###     NEW     ###############################################
##############################################################



import os
from sklearn.ensemble import RandomForestClassifier

def create_random_forest(
    X_train, y_train, current_dataset, current_fold, tree_depth, n_trees, return_file=False
):
    """
    Create a random forest classifier and save it to a file in the required format.

    Args:
    - X_train: Training data features
    - y_train: Training data labels
    - current_dataset: Name of the dataset
    - current_fold: Current fold for cross-validation
    - tree_depth: Maximum depth of trees
    - n_trees: Number of trees in the random forest
    - return_file: If True, save the random forest to a file and return the file path

    Returns:
    - random_forest: Trained RandomForestClassifier
    - random_forest_file (if return_file is True): Path to the saved random forest file
    """
    def calculate_depth(children_left, children_right):
        """Calculate depth for each node in the tree."""
        def depth(node):
            if node == -1:
                return 0
            left_depth = depth(children_left[node])
            right_depth = depth(children_right[node])
            return max(left_depth, right_depth) + 1

        node_depth = [0] * len(children_left)
        for i in range(len(children_left)):
            node_depth[i] = depth(i)
        return node_depth

    # Train the random forest
    random_forest = RandomForestClassifier(
        n_estimators=n_trees, random_state=0, max_depth=tree_depth
    )
    random_forest.fit(X_train, y_train)

    # Prepare the output directory and file path
    output_dir = os.path.join("output_new", "RF", current_dataset)
    random_forest_file = os.path.join(
        output_dir, f"{current_dataset}.RF{current_fold}.T{n_trees}.D{tree_depth}.txt"
    )

    if return_file:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write the random forest to the file
        with open(random_forest_file, "w") as f:
            # Write metadata
            f.write(f"DATASET_NAME: {current_dataset}\n")
            f.write("ENSEMBLE: RF\n")
            f.write(f"NB_TREES: {n_trees}\n")
            f.write(f"NB_FEATURES: {X_train.shape[1]}\n")
            f.write(f"NB_CLASSES: {len(set(y_train))}\n")
            f.write(f"MAX_TREE_DEPTH: {tree_depth}\n")
            f.write("Format: node / node type (LN - leave node, IN - internal node) "
                    "left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n\n")

            # Write each tree
            for idx, estimator in enumerate(random_forest.estimators_):
                tree_ = estimator.tree_
                node_depths = calculate_depth(tree_.children_left, tree_.children_right)

                f.write(f"[TREE {idx}]\n")
                f.write(f"NB_NODES: {tree_.node_count}\n")
                for i in range(tree_.node_count):
                    left = tree_.children_left[i]
                    right = tree_.children_right[i]
                    feature = tree_.feature[i]
                    threshold = tree_.threshold[i]
                    depth = node_depths[i]
                    majority_class = int(tree_.value[i][0].argmax())  # Majority class

                    # Determine node type
                    node_type = "LN" if left == -1 and right == -1 else "IN"

                    # Adjust leaf node values
                    if node_type == "LN":
                        left = right = feature = -1
                        threshold = -1.0

                    # Write the node information
                    f.write(
                        f"{i} {node_type} {left} {right} {feature} "
                        f"{threshold:.6f} {depth} {majority_class}\n"
                    )
                f.write("\n")

        return random_forest, random_forest_file

    return random_forest
