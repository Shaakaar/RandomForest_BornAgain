#%%
# ------------------------------------------------------------------
#  CODE: BORN AGAIN ALGORITHM
#  RESULTS: Dynamically being saved to ->
#  src\aggregated_results_partial.csv and src\output_new
#  MinGW: download on https://sourceforge.net/projects/mingw/  ->
#  Add to path: C:...\MinGW\msys\1.0\bin and C:...\MinGW\bin
#  CPLEX: download on https://www.ibm.com/products/ilog-cplex-optimization-studio ->
#  Add to path: C:...\IBM\ILOG\CPLEX_Studio_Community2212\cplex\bin\x64_win64
#  Python version: 3.7
#  scikit-learn version: 0.22.1
# ------------------------------------------------------------------


# Import packages
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.metrics import classification_report
from tqdm import tqdm  
from datetime import datetime  
from pathlib import Path

# Project-specific modules
import datasets as ds
import random_forests as rf
import persistence as tree_io
import visualization as tree_view

#%%

def log_time(message):
    """Print a timestamped log message."""
    print(f"{message}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create directories for output
figure_output_dir = Path("output_new/Figures")
figure_output_dir.mkdir(parents=True, exist_ok=True)

# Parameters
current_dataset_loop = ['COMPAS-ProPublica', 'FICO']
current_fold_loop = [1]
n_trees_loop = [5, 10]
current_obj_loop = [2]
max_tree_depth_loop = [3, 4]

# We'll keep a global CSV path for partial results
PARTIAL_CSV_PATH = "aggregated_results_partial.csv"

###############################################################################
# 1) Function to produce Born-Again .tree files
###############################################################################
def run_born_again(objectives, datasets, folds, n_trees, tree_depth):
    """
    Runs bornAgain.exe for each combo. If Ctrl+C is pressed,
    partial combos are returned. No partial CSV saving here,
    because we only track the combos, not classification metrics.
    """

    # Adjust these paths for your local environment
    output_new_dir = Path(
        "C:/Users/ruben/OneDrive/Documents/RandomForest_BornAgain/BATREES/docs/src/output_new"
        
    )
    born_again_dir = output_new_dir / "Born_Again"
    exe_path = r"C:\Users\ruben\OneDrive\Documents\RandomForest_BornAgain\BATREES\docs\src\born_again_dp\bornAgain.exe"

    # Make sure directories exist
    output_new_dir.mkdir(parents=True, exist_ok=True)
    born_again_dir.mkdir(parents=True, exist_ok=True)

    for dsname in datasets:
        (born_again_dir / dsname).mkdir(parents=True, exist_ok=True)

    # We store successful combos (dataset, fold, trees, depth, objective)
    completed_runs = []

    try:
        total_iterations = (
            len(objectives) * len(datasets) * len(folds) *
            len(n_trees) * len(tree_depth)
        )
        progress = tqdm(
            total=total_iterations,
            desc="Running Born Again Trees",
            bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]"
        )

        for o in objectives:
            for dsname in datasets:
                print("\n" + "=="*50)
                print(f"DATASET RUNNING: {dsname} (BORN AGAIN)")
                print("=="*50 + "\n")

                for fold in folds:
                    for trees_count in n_trees:
                        for depth_val in tree_depth:
                            rf_input = output_new_dir / f"RF/{dsname}/{dsname}.RF{fold}.T{trees_count}.D{depth_val}.txt"
                            ba_output = born_again_dir / f"{dsname}/{dsname}.BA{fold}.O{o}.T{trees_count}.D{depth_val}"

                            cmd = [
                                str(exe_path),
                                str(rf_input),
                                str(ba_output),
                                "-trees", str(trees_count),
                                "-obj", str(o)
                            ]
                            log_time(
                                f"Running BornAgain => dataset={dsname}, fold={fold}, "
                                f"trees={trees_count}, depth={depth_val}, obj={o}"
                            )
                            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            if ret.returncode == 0:
                                completed_runs.append((dsname, fold, trees_count, depth_val, o))
                                print(f"Completed: {' '.join(cmd)}")
                            else:
                                print(f"Error: Command {' '.join(cmd)} returned code {ret.returncode}")
                                print(ret.stderr.decode("utf-8"))

                            progress.update(1)
        progress.close()
        return completed_runs

    except KeyboardInterrupt:
        print("\n" + "=="*50 + "\n" + 
        "Interrupted! Returning partial 'completed_runs' so far" 
        + "\n" + "=="*50 + "\n")
        return completed_runs
    
def calculate_APL(tree, X_test):
    # Calculate average path length
    node_depth = np.zeros(shape=tree.tree_.node_count, dtype=np.int64)
    stack = [(0, 0)]  # (node_id, depth)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]
        
        if left_child != -1:
            stack.append((left_child, depth + 1))
        if right_child != -1:
            stack.append((right_child, depth + 1))

    # Calculate average path length using test set
    leaf_ids = tree.apply(X_test)
    return node_depth[leaf_ids].mean()

def calculate_complexity(nn, apl):
    #  # Get all values for standardization
    # all_nodes = [result["Nodes"] for result in aggregated_results] + [  ]
    # all_apl = [result["Avg Path Length"] for result in aggregated_results] + [ba_avg_path_length]

    # # Step 1: Standardization (Formula 1)
    # def standardize(values):
    #     total = sum(values)
    #     return [x / total for x in values]

    # p_nodes = standardize(all_nodes)
    # p_apl = standardize(all_apl)

    # # Step 2: Calculate entropy values (Formula 2)
    # def calculate_entropy(p_values):
    #     n = len(p_values)
    #     entropy = 0
    #     for p in p_values:
    #         if p != 0:  # Handle p=0 case as mentioned in the paper
    #             entropy += p * np.log(p)
    #     return -entropy / np.log(n)

    # E_nodes = calculate_entropy(p_nodes)
    # E_apl = calculate_entropy(p_apl)

    # # Step 3: Calculate weights (Formula 3)
    # def calculate_weights(entropy_values):
    #     # Calculate 1-E for each metric
    #     diversity_degrees = [1 - e for e in entropy_values]
    #     # Sum of all diversity degrees
    #     total_diversity = sum(diversity_degrees)
    #     # Calculate final weights
    #     return [d / total_diversity for d in diversity_degrees]

    # weights = calculate_weights([E_nodes, E_apl])
    # w_nn, w_apl = weights

    # # Calculate complexity score using current tree's standardized values
    # score_nn = p_nodes[-1]   # Current tree's standardized nodes value
    # score_apl = p_apl[-1]    # Current tree's standardized APL value
    # complexity_score = w_nn * score_nn + w_apl * score_apl
    return 0.5 * nn + 0.5 * apl

def get_branch_attributes(tree, node_id):
    """Get attributes used in a branch from root to node"""
    feature_path = set()
    current_id = node_id
    
    # Traverse up until root
    while current_id != 0:
        # Find parent
        parent_id = -1
        for i, (left, right) in enumerate(zip(tree.children_left, tree.children_right)):
            if left == current_id or right == current_id:
                parent_id = i
                break
        if parent_id == -1:
            break
            
        feature_path.add(tree.feature[parent_id])
        current_id = parent_id
        
    return feature_path

def calculate_DAR(clf, X_test):
    """Calculate Duplicate Attribute Ratio"""
    tree = clf.tree_
    n_samples = X_test.shape[0]
    leaf_ids = clf.apply(X_test)  # Get leaf node for each sample
    
    # Calculate w(θ) for each unique leaf
    leaf_weights = {}  # leaf_id -> weight
    for leaf_id in np.unique(leaf_ids):
        leaf_weights[leaf_id] = np.sum(leaf_ids == leaf_id) / n_samples
    
    # Calculate PDAR(θ) for each leaf
    leaf_pdars = {}  # leaf_id -> pdar
    for leaf_id in leaf_weights.keys():
        attributes = get_branch_attributes(tree, leaf_id)
        if len(attributes) > 0:  # Avoid division by zero
            leaf_pdars[leaf_id] = (len(attributes) - len(set(attributes))) / len(attributes)
        else:
            leaf_pdars[leaf_id] = 0
    
    # Calculate final DAR
    dar = sum(leaf_weights[leaf_id] * leaf_pdars[leaf_id] 
             for leaf_id in leaf_weights.keys())
    
    return dar

def get_subtree_structure(tree, node_id):
    """Get structural representation of subtree rooted at node_id"""
    if tree.children_left[node_id] == -1:  # Leaf node
        return "L"
    
    left_struct = get_subtree_structure(tree, tree.children_left[node_id])
    right_struct = get_subtree_structure(tree, tree.children_right[node_id])
    
    return f"({left_struct},{right_struct})"

def calculate_DSR(clf):
    """Calculate Duplicate Subtree Ratio"""
    tree = clf.tree_
    
    # Get structure for each node
    structures = {}  # structure -> [node_ids]
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] != -1:  # Not a leaf
            struct = get_subtree_structure(tree, node_id)
            if struct not in structures:
                structures[struct] = []
            structures[struct].append(node_id)
    
    # Calculate nums(ω) * nodes(ω) for each structure
    duplicate_nodes = sum(len(nodes) * sum(tree.node_count - tree.children_left[node]
                                         for node in nodes)
                         for struct, nodes in structures.items()
                         if len(nodes) > 1)
    
    # Calculate DSR
    dsr = duplicate_nodes / tree.node_count if tree.node_count > 0 else 0
    
    return dsr

###############################################################################
# 2) Main pipeline
###############################################################################
def run_all_processes():
    """
    1) Create random forests (with partial CSV saving), incl. saving RF figures
    2) run born_again() with interrupt
    3) evaluate partial combos (with partial CSV saving), incl. BornAgain figures
    4) final CSV
    """
    aggregated_results = []
    start_time = datetime.now()
    log_time("Script started")

    ###########################################################################
    # A) Create Random Forests (with partial saving)
    ###########################################################################
    total_rf_iterations = (
        len(current_dataset_loop) * len(current_fold_loop) *
        len(n_trees_loop) * len(max_tree_depth_loop)
    )
    rf_progress = tqdm(
        total=total_rf_iterations,
        desc="Creating Random Forests",
        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]"
    )

    for current_dataset in current_dataset_loop:
        print("\n" + "=="*50)
        print(f"DATASET RUNNING: {current_dataset} (RANDOM FOREST)")
        print("=="*50 + "\n")

        for current_fold in current_fold_loop:
            for n_trees in n_trees_loop:
                for max_tree_depth in max_tree_depth_loop:
                    # Load data
                    df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                    X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
                    X_test,  y_test  = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

                    # Train the RF
                    log_time(
                        f"Creating RandomForest => DS={current_dataset}, "
                        f"fold=x{current_fold}, trees={n_trees}, depth={max_tree_depth}"
                    )
                    rf_model, rf_file = rf.create_random_forest(
                        X_train, y_train,
                        current_dataset, current_fold,
                        max_tree_depth, n_trees,
                        return_file=True
                    )

                    rf_trees = [e.tree_ for e in rf_model.estimators_]
                    
                    # Evaluate
                    rf_test_pred   = rf_model.predict(X_test)
                    rf_train_pred  = rf_model.predict(X_train)
                    report_rf      = classification_report(y_test,  rf_test_pred,   output_dict=True)
                    report_rf_tr   = classification_report(y_train, rf_train_pred, output_dict=True)

                    # Calculate total leaves and nodes for all trees
                    rf_leaves = 0
                    rf_nodes = 0
                    for est in rf_model.estimators_:
                        rf_leaves += est.tree_.n_leaves
                        rf_nodes += est.tree_.node_count

                    # Calculate average path length across all trees
                    total_path_length = 0
                    n_samples = 0
                    for est in rf_model.estimators_:
                        total_path_length += calculate_APL(est, X_test)
                        n_samples += 1
                    

                    avg_path_length = total_path_length / n_samples

                    # Store
                    aggregated_results.append({
                        "Dataset":    current_dataset,
                        "Trees":      n_trees,
                        "Max Depth":  max_tree_depth,
                        "Method":     "RandomForest",
                        "Train Acc":  report_rf_tr["accuracy"],
                        "Train F1":   report_rf_tr["weighted avg"]["f1-score"],
                        "Test Acc":   report_rf["accuracy"],
                        "Test F1":    report_rf["weighted avg"]["f1-score"],
                        "Leaves":     rf_leaves,
                        "Nodes":      rf_nodes,
                        "Avg Path Length": avg_path_length,
                        # "Complexity Score": None,
                        "DAR": None,
                        "DSR": None,
                    })

                    # Optional: save a figure for the first tree if you want
                    final_fig_outputdir = figure_output_dir / current_dataset
                    final_fig_outputdir.mkdir(parents=True, exist_ok=True)
                    if n_trees in [5] and max_tree_depth in [3]:
                        # We'll load the 1st tree for a quick figure
                        random_forest_clf = tree_io.classifier_from_file(rf_file, X_train, y_train, pruning=False)
                        # create a graph
                        rf_graph = tree_view.create_graph(
                            rf_trees,
                            features=ds_infos['features'],
                            classes=ds_infos['classes'],
                            colors=ds_infos['colors']
                        )
                        # pick an objective=4 for naming
                        fig_path = final_fig_outputdir / (
                            f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_RF.png"
                        )
                        with open(fig_path, "wb") as f:
                            f.write(rf_graph.create_png())

                    # Real-time partial CSV
                    aggregated_results_df = pd.DataFrame(aggregated_results)
                    aggregated_results_df["Method"] = pd.Categorical(
                            aggregated_results_df["Method"],
                            categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
                            ordered=True
                        )
                    aggregated_results_df.sort_values(["Dataset", "Trees", "Max Depth", "Method"], inplace=True)
                    aggregated_results_df.to_csv(PARTIAL_CSV_PATH, index=False)

                    rf_progress.update(1)

    rf_progress.close()

    ###########################################################################
    # B) Run BornAgain => returns partial combos if interrupted
    ###########################################################################
    completed_runs = run_born_again(
        current_obj_loop,
        current_dataset_loop,
        current_fold_loop,
        n_trees_loop,
        max_tree_depth_loop
    )

    ###########################################################################
    # C) Evaluate partial BornAgain combos (with partial saving)
    ###########################################################################
    ba_progress_total = (
        len(current_dataset_loop)*len(current_fold_loop)*
        len(n_trees_loop)*len(max_tree_depth_loop)*len(current_obj_loop)
    )
    ba_progress = tqdm(
        total=ba_progress_total,
        desc="Evaluating BornAgain",
        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]"
    )

    for o in current_obj_loop:
        for current_dataset in current_dataset_loop:
            for current_fold in current_fold_loop:
                # load data
                df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
                X_test,  y_test  = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

                for n_trees in n_trees_loop:
                    for max_tree_depth in max_tree_depth_loop:

                        # Check if it was completed
                        if (current_dataset, current_fold, n_trees, max_tree_depth, o) not in completed_runs:
                            print(f"Skipping => DS={current_dataset}, fold={current_fold}, "
                                  f"T={n_trees}, depth={max_tree_depth}, obj={o}: not completed.")
                            ba_progress.update(1)
                            continue

                        # The .tree path
                        ba_base = (
                            f"output_new/Born_Again/{current_dataset}/"
                            f"{current_dataset}.BA{current_fold}.O{o}.T{n_trees}.D{max_tree_depth}"
                        )
                        ba_treefile = ba_base + ".tree"

                        if not os.path.exists(ba_treefile):
                            print(f"Missing .tree file: {ba_treefile}")
                            ba_progress.update(1)
                            continue

                        log_time(f"Evaluating BornAgain => DS={current_dataset}, fold={current_fold}, "
                                 f"trees={n_trees}, depth={max_tree_depth}, obj={o}")

                        # load unpruned
                        born_again_clf = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=False)
                        # load pruned
                        born_again_pruned_clf = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=True)

                        # Evaluate unpruned
                        ba_test_pred  = born_again_clf.predict(X_test)
                        ba_train_pred = born_again_clf.predict(X_train)
                        rep_ba        = classification_report(y_test, ba_test_pred,   output_dict=True)
                        rep_ba_tr     = classification_report(y_train, ba_train_pred, output_dict=True)
                        ba_leaves     = born_again_clf.tree_.n_leaves
                        ba_nodes      = born_again_clf.tree_.node_count
                        ba_avg_path_length = calculate_APL(born_again_clf, X_test)
                        ba_complexity = calculate_complexity(ba_nodes, ba_avg_path_length)

                        # Calculate clarity metrics
                        ba_dar = calculate_DAR(born_again_clf, X_test)
                        ba_dsr = calculate_DSR(born_again_clf)

                        aggregated_results.append({
                            "Dataset":   current_dataset,
                            "Trees":     n_trees,
                            "Max Depth": max_tree_depth,
                            "Method":    "BornAgain",
                            "Train Acc": rep_ba_tr["accuracy"],
                            "Train F1":  rep_ba_tr["weighted avg"]["f1-score"],
                            "Test Acc":  rep_ba["accuracy"],
                            "Test F1":   rep_ba["weighted avg"]["f1-score"],
                            "Leaves":    ba_leaves,
                            "Nodes":     ba_nodes,
                            "Avg Path Length": ba_avg_path_length,
                            # "Complexity Score": ba_complexity,
                            "DAR": ba_dar,
                            "DSR": ba_dsr,
                        })
                        # Evaluate pruned
                        ba_pruned_test_pred  = born_again_pruned_clf.predict(X_test)
                        ba_pruned_train_pred = born_again_pruned_clf.predict(X_train)
                        rep_bap              = classification_report(y_test, ba_pruned_test_pred,   output_dict=True)
                        rep_bap_tr           = classification_report(y_train, ba_pruned_train_pred, output_dict=True)
                        ba_pruned_leaves     = born_again_pruned_clf.tree_.n_leaves
                        ba_pruned_nodes      = born_again_pruned_clf.tree_.node_count
                        ba_pruned_avg_path_length = calculate_APL(born_again_clf, X_test)
                        ba_pruned_complexity = calculate_complexity(ba_pruned_nodes, ba_pruned_avg_path_length)
                        # Calculate clarity metrics
                        ba_pruned_dar = calculate_DAR(born_again_pruned_clf, X_test)
                        ba_pruned_dsr = calculate_DSR(born_again_pruned_clf)


                        aggregated_results.append({
                            "Dataset":   current_dataset,
                            "Trees":     n_trees,
                            "Max Depth": max_tree_depth,
                            "Method":    "BornAgain-Pruned",
                            "Train Acc": rep_bap_tr["accuracy"],
                            "Train F1":  rep_bap_tr["weighted avg"]["f1-score"],
                            "Test Acc":  rep_bap["accuracy"],
                            "Test F1":   rep_bap["weighted avg"]["f1-score"],
                            "Leaves":    ba_pruned_leaves,
                            "Nodes":     ba_pruned_nodes,
                            "Avg Path Length": ba_pruned_avg_path_length,
                            # "Complexity Score": ba_pruned_complexity,
                            "DAR": ba_pruned_dar,
                            "DSR": ba_pruned_dsr,
                        })

                        # Save partial CSV again
                        aggregated_results_df = pd.DataFrame(aggregated_results)
                        aggregated_results_df["Method"] = pd.Categorical(
                            aggregated_results_df["Method"],
                            categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
                            ordered=True
                        )
                        aggregated_results_df.sort_values(["Dataset", "Trees", "Max Depth", "Method"], inplace=True)
                        aggregated_results_df.to_csv(PARTIAL_CSV_PATH, index=False)

                        # (Optional) save BornAgain figure
                        final_fig_outputdir = figure_output_dir / current_dataset
                        final_fig_outputdir.mkdir(parents=True, exist_ok=True)

                        # Pruned figure
                        if n_trees in [5] and max_tree_depth in [3]:
                            pruned_graph = tree_view.create_graph(
                                [born_again_pruned_clf.tree_],
                                features=ds_infos['features'],
                                classes=ds_infos['classes'],
                                colors=ds_infos['colors']
                            )
                            pruned_fig_path = final_fig_outputdir / (
                                f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{o}_Pruned.png"
                            )
                            with open(pruned_fig_path, "wb") as f:
                                f.write(pruned_graph.create_png())

                        if n_trees in [5] and max_tree_depth in [3]:
                            # unpruned figure
                            unpruned_graph = tree_view.create_graph(
                                [born_again_clf.tree_],
                                features=ds_infos['features'],
                                classes=ds_infos['classes'],
                                colors=ds_infos['colors']
                            )
                            unpruned_fig_path = final_fig_outputdir / (
                                f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{o}_Unpruned.png"
                            )
                            with open(unpruned_fig_path, "wb") as f:
                                f.write(unpruned_graph.create_png())

                        ba_progress.update(1)

    ba_progress.close()

    ###########################################################################
    # D) Final Output
    ###########################################################################
    results_df = pd.DataFrame(aggregated_results)
    results_df["Method"] = pd.Categorical(
        results_df["Method"],
        categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
        ordered=True
    )
    results_df.sort_values(["Dataset", "Trees", "Max Depth", "Method"], inplace=True)
    log_time("Saving final CSV => aggregated_results.csv")
    results_df.to_csv("aggregated_results.csv", index=False)

    end_time = datetime.now()
    print("\n" + "="*50, "\nFINAL RESULTS:\n", results_df, "\n" + "="*50, "\n")
    log_time("Script completed")

    total_runtime = end_time - start_time
    print("="*50)
    print(f"Total Run Time: {str(total_runtime)}")
    print("="*50)
    return results_df

if __name__ == "__main__":
    df = run_all_processes()


# %%
