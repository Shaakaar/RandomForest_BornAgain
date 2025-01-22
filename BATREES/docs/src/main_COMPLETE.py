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
from collections import Counter
import time

# Project-specific modules
import datasets as ds
import random_forests as rf
import persistence as tree_io
import visualization as tree_view



def log_time(message):
    """Print a timestamped log message."""
    print(f"{message}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create directories for output
figure_output_dir = Path("output_new/Figures")
figure_output_dir.mkdir(parents=True, exist_ok=True)

# Parameters
current_dataset_loop = ['HTRU2']
current_fold_loop = [1]
n_trees_loop = [5, 10]
current_obj_loop = [4]
max_tree_depth_loop = [3, 4, 10]

# We'll keep a global CSV path for partial results
PARTIAL_CSV_PATH = "aggregated_results_partial.csv"

###############################################################################
# 1) Creating Functions to Create and Evaluate BornAgain Trees 
###############################################################################
def run_born_again(objectives, datasets, folds, n_trees, tree_depth, 
                   aggregated_results,  # pass in your current list 
                   PARTIAL_CSV_PATH     # path for partial saving
                  ):
    """
    For each combination, calls bornAgain.exe, measures runtime, 
    AND immediately evaluates the resulting .tree (unpruned & pruned).
    Appends new rows to `aggregated_results` and does partial CSV writing.
    
    Returns:
      completed_runs: list of (dataset, fold, trees_count, depth_val, obj)
      run_times: dict keyed by that 5‐tuple => float runtime (seconds).
    """

    output_new_dir = Path("C:/School/CASE_BA/RF_BA/BATREES/docs/src/output_new")
    born_again_dir = output_new_dir / "Born_Again"
    exe_path = r"C:\School\Case_BA\RF_BA\BATREES\docs\src\born_again_dp\bornAgain.exe"

    output_new_dir.mkdir(parents=True, exist_ok=True)
    born_again_dir.mkdir(parents=True, exist_ok=True)

    total_runtime_bornAgain = 0
    total_runtime_eval = 0

    print("\n" + "=="*50 + "\n")
    print("CREATING BornAgain...")
    print("\n" + "=="*50 + "\n")
    

    for dsname in datasets:
        (born_again_dir / dsname).mkdir(parents=True, exist_ok=True)

    completed_runs = []
    run_times = {}  # (dataset, fold, trees, depth, obj) -> float

    # We’ll have a progress bar for total combos
    total_iterations = (
        len(objectives)*len(datasets)*len(folds)*len(n_trees)*len(tree_depth)
    )
    progress = tqdm(
        total=total_iterations,
        desc="Running and Evaluating BornAgain",
        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]"
    )
    
    t_start_ba = time.time() # Start Time BornAgain

    try:
        
        for o in objectives:
            for dsname in datasets:
                print("\n" + "=="*50)
                print(f"DATASET RUNNING: {dsname} (BORN AGAIN)")
                print("=="*50 + "\n")

                for fold in folds:
                    # We load data once per fold
                    df_train, df_test, ds_infos = ds.load(dsname, fold)
                    X_train = df_train.iloc[:,:-1].values
                    y_train = df_train.iloc[:,-1].values
                    X_test  = df_test.iloc[:,:-1].values
                    y_test  = df_test.iloc[:,-1].values

                    for trees_count in n_trees:
                        for depth_val in tree_depth:
                            # 1) Build paths
                            rf_input = (output_new_dir 
                                        / f"RF/{dsname}/{dsname}.RF{fold}.T{trees_count}.D{depth_val}.txt")
                            ba_output = (born_again_dir 
                                         / f"{dsname}/{dsname}.BA{fold}.O{o}.T{trees_count}.D{depth_val}")

                            cmd = [
                                str(exe_path),
                                str(rf_input),
                                str(ba_output),
                                "-trees", str(trees_count),
                                "-obj",   str(o)
                            ]

                            print('\n' + '--'*50)
                            log_time(
                                f"Running BornAgain => dataset={dsname}, fold={fold}, "
                                f"trees={trees_count}, depth={depth_val}, obj={o}"
                            )

                            start_t = time.time()
                            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            end_t = time.time()
                            log_time('DONE')
                            ba_runtime = end_t - start_t
                            print(f'Runtime: {round(ba_runtime, 3)}s')
                            if ret.returncode != 0:
                                # If failed, we skip aggregator logic
                                print(f"Error in bornAgain.exe => {cmd}")
                                print(ret.stderr.decode("utf-8"))
                            else:
                                # Mark this combo as completed
                                combo = (dsname, fold, trees_count, depth_val, o)
                                completed_runs.append(combo)
                                run_times[combo] = ba_runtime

                                # 3) Evaluate unpruned & pruned immediately
                                t_start_eval = time.time()
                                log_time('Evaluating')
                                treefile = str(ba_output) + ".tree"
                                if not os.path.exists(treefile):
                                    print(f"** Missing .tree file => {treefile}")
                                else:
                                    # Unpruned
                                    unpruned_clf = tree_io.classifier_from_file(treefile, 
                                                                                X_train, y_train,
                                                                                pruning=False)
                                    ba_test_pred   = unpruned_clf.predict(X_test)
                                    ba_train_pred  = unpruned_clf.predict(X_train)
                                    rep_ba         = classification_report(y_test,   ba_test_pred,  output_dict=True)
                                    rep_ba_tr      = classification_report(y_train,  ba_train_pred, output_dict=True)
                                    ba_leaves      = unpruned_clf.tree_.n_leaves
                                    ba_nodes       = unpruned_clf.tree_.node_count
                                    ba_apl         = calculate_APL(unpruned_clf, X_test)
                                    ba_dar         = calculate_DAR(unpruned_clf, X_test)
                                    ba_dsr         = calculate_DSR(unpruned_clf)
                                    ba_zhou        = zhou_interpretability(unpruned_clf)

                                    aggregated_results.append({
                                        "Dataset":   dsname,
                                        "Trees":     trees_count,
                                        "Max Depth": depth_val,
                                        "Method":    "BornAgain",
                                        "Train Acc": round(rep_ba_tr["accuracy"], 3),
                                        "Train F1":  round(rep_ba_tr["weighted avg"]["f1-score"], 3),
                                        "Test Acc":  round(rep_ba["accuracy"], 3),
                                        "Test F1":   round(rep_ba["weighted avg"]["f1-score"], 3),
                                        "Leaves":    ba_leaves,
                                        "Nodes":     ba_nodes,
                                        "Avg Path Length": round(ba_apl, 3),
                                        "DAR":       round(ba_dar, 3),
                                        "DSR":       round(ba_dsr, 3),
                                        "Zhou score":round(ba_zhou, 3),
                                        "Runtime(s)":   round(ba_runtime, 3),
                                        "Runtime(m)": round(ba_runtime/60,3)
                                    })

                                    # Pruned
                                    pruned_clf = tree_io.classifier_from_file(treefile,
                                                                              X_train, y_train,
                                                                              pruning=True)
                                    p_test_pred   = pruned_clf.predict(X_test)
                                    p_train_pred  = pruned_clf.predict(X_train)
                                    rep_bap       = classification_report(y_test,  p_test_pred,   output_dict=True)
                                    rep_bap_tr    = classification_report(y_train, p_train_pred,  output_dict=True)
                                    p_leaves      = pruned_clf.tree_.n_leaves
                                    p_nodes       = pruned_clf.tree_.node_count
                                    p_apl         = calculate_APL(pruned_clf, X_test)
                                    p_dar         = calculate_DAR(pruned_clf, X_test)
                                    p_dsr         = calculate_DSR(pruned_clf)
                                    p_zhou        = zhou_interpretability(pruned_clf)

                                    aggregated_results.append({
                                        "Dataset":   dsname,
                                        "Trees":     trees_count,
                                        "Max Depth": depth_val,
                                        "Method":    "BornAgain-Pruned",
                                        "Train Acc": round(rep_bap_tr["accuracy"], 3),
                                        "Train F1":  round(rep_bap_tr["weighted avg"]["f1-score"], 3),
                                        "Test Acc":  round(rep_bap["accuracy"], 3),
                                        "Test F1":   round(rep_bap["weighted avg"]["f1-score"], 3),
                                        "Leaves":    p_leaves,
                                        "Nodes":     p_nodes,
                                        "Avg Path Length": round(p_apl, 3),
                                        "DAR":       round(p_dar, 3),
                                        "DSR":       round(p_dsr, 3),
                                        "Zhou score":round(p_zhou, 3),
                                        "Runtime(s)":   round(ba_runtime, 3),  # same time as bornAgain
                                        "Runtime(m)": round(ba_runtime/60, 3)
                                    })

                                    # 4) Partial CSV
                                    df_part = pd.DataFrame(aggregated_results)
                                    df_part["Method"] = pd.Categorical(
                                        df_part["Method"],
                                        categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
                                        ordered=True
                                    )
                                    df_part.sort_values(["Dataset","Trees","Max Depth","Method"], inplace=True)
                                    df_part.to_csv(PARTIAL_CSV_PATH, index=False)
                                    t_end_eval = time.time()
                                    eval_runtime = t_end_eval - t_start_eval
                                    log_time('DONE')
                                    print(f'Evaluation Runtime: {round(eval_runtime,3)}s')
                                    print('--'*50)


                                    total_runtime_bornAgain += ba_runtime
                                    total_runtime_eval += eval_runtime
                            progress.update(1)
    # If Interrupted, partial results are returned
    except KeyboardInterrupt:
        print("\n" + "=="*50 + "\nInterrupted => returning partial data.\n" + "=="*50 + "\n")

    t_end_ba = time.time() # End Time BornAgain
    total_runtime_2 = t_end_ba - t_start_ba # Total Runtime BornAgain
    progress.close()

    print('\n' + '=='*50)
    print(f'Total Runtime BornAgain: {round(total_runtime_bornAgain,3)}s')
    print(f'Total Runtime Evaluation: {round(total_runtime_eval,3)}s')
    print(f'Total Runtime (BA): {round(total_runtime_2,3)}s')
    print('=='*50 + '\n')

    return completed_runs, run_times  # aggregated_results is being mutated in-place

    

    
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
    """Get attributes used in a branch from root to node
    
    Args:
        tree: Decision tree classifier
        node_id: ID of the current node
    
    Returns:
        List of feature indices used in the path from root to node
    """
    feature_path = []
    current_id = node_id
    
    # Traverse up the tree until we reach the root (node_id == 0)
    while current_id != 0:
        # Get parent node
        parent_id = -1
        for i, child in enumerate(tree.tree_.children_left):
            if child == current_id:
                parent_id = i
                break
        if parent_id == -1:
            for i, child in enumerate(tree.tree_.children_right):
                if child == current_id:
                    parent_id = i
                    break
                    
        # Add feature used in the split
        feature_path.append(tree.tree_.feature[parent_id])
        current_id = parent_id
        
    # Add root node feature
    feature_path.append(tree.tree_.feature[0])
    return feature_path

def calculate_DAR(clf, X_test):
    """Calculate Duplicate Attribute Ratio
    
    Args:
        clf: Trained decision tree classifier
        X_test: Test samples
        
    Returns:
        float: Duplicate Attribute Ratio
    """
    n_samples = X_test.shape[0]
    tree = clf.tree_
    
    # Get leaf node for each test sample
    leaf_nodes = clf.apply(X_test)
    
    # Calculate DAR for each sample's path
    total_dar = 0
    
    for i, leaf_node in enumerate(leaf_nodes):
        # Get features in path to leaf
        path_features = get_branch_attributes(clf, leaf_node)
        
        # Count duplicate features
        feature_counts = Counter(path_features)
        n_duplicates = sum(count - 1 for count in feature_counts.values() if count > 1)
        
        # Calculate PDAR for this branch (eq 9.4)
        n_nodes = len(path_features)
        pdar = n_duplicates / n_nodes if n_nodes > 0 else 0
        
        # Calculate sample weight (eq 9.3)
        w = 1.0 / n_samples
        
        # Add weighted PDAR to total
        total_dar += w * pdar
    
    return total_dar

def zhou_interpretability(clf):
    """Calculate Zhou's interpretability measure
    
    Formula: I = -0.33 · number of leaves - 0.25 · average depth + 0.13 · maximum depth + 0.59
    
    Args:
        clf: Trained decision tree classifier
        
    Returns:
        float: Interpretability score
    """
    tree = clf.tree_
    
    # Calculate number of leaves
    n_leaves = tree.n_leaves
    
    # Calculate depths of all nodes
    depths = np.zeros(tree.node_count, dtype=np.int32)
    stack = [(0, 0)]  # (node_id, depth)
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        
        if left != -1:
            stack.append((left, depth + 1))
        if right != -1:
            stack.append((right, depth + 1))
    
    # Calculate maximum depth
    max_depth = np.max(depths)
    
    # Calculate average depth (only of leaf nodes)
    leaf_depths = [depths[i] for i in range(tree.node_count) if tree.children_left[i] == -1]
    avg_depth = np.mean(leaf_depths)
    
    # Apply Zhou's formula
    interpretability = (-0.33 * n_leaves - 
                       0.25 * avg_depth + 
                       0.13 * max_depth + 
                       0.59)
    
    return interpretability

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

def Visualize_Tree(method, current_dataset, current_fold, n_trees, max_tree_depth, X_train, y_train, ds_infos, ba_progress, obj):
        

        figure_output_dir = Path("output_new/Figures")
        figure_output_dir.mkdir(parents=True, exist_ok=True)


        log_time(f"Creating Figure for {method} => DS={current_dataset}, fold={current_fold}, "
                    f"trees={n_trees}, depth={max_tree_depth}, obj={obj}")

        
        final_fig_outputdir = figure_output_dir / current_dataset
        final_fig_outputdir.mkdir(parents=True, exist_ok=True)

        # Random Forest
        if method == 'RandomForest':

            # Path to RandomForest file
            random_forest_file = f"output_new/RF/{current_dataset}/{current_dataset}.RF{current_fold}.T{n_trees}.D{max_tree_depth}.txt"

            rf_model = tree_io.classifier_from_file(random_forest_file, X_train, y_train, pruning=False)
            
            rf_trees = [e.tree_ for e in rf_model.estimators_]
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

            
            
        # Born Again
        elif method == 'BornAgain':
                        
                # Path to BornAgain file
                ba_base = (
                    f"output_new/Born_Again/{current_dataset}/"
                    f"{current_dataset}.BA{current_fold}.O{obj}.T{n_trees}.D{max_tree_depth}"
                )
                ba_treefile = ba_base + ".tree"
                        
                # Load pruned
                born_again_pruned_clf = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=True)
                pruned_graph = tree_view.create_graph(
                    [born_again_pruned_clf.tree_],
                    features=ds_infos['features'],
                    classes=ds_infos['classes'],
                    colors=ds_infos['colors']
                )
                pruned_fig_path = final_fig_outputdir / (
                    f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{obj}_Pruned.png"
                )
                with open(pruned_fig_path, "wb") as f:
                    f.write(pruned_graph.create_png())

                
        # BornAgain_pruned 
        elif method == 'BornAgain_pruned':
                
                # Path to BornAgain_pruned file
                ba_base = (
                    f"output_new/Born_Again/{current_dataset}/"
                    f"{current_dataset}.BA{current_fold}.O{obj}.T{n_trees}.D{max_tree_depth}"
                )
                ba_treefile = ba_base + ".tree"

                # Load unpruned
                born_again_clf = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=False)

                unpruned_graph = tree_view.create_graph(
                    [born_again_clf.tree_],
                    features=ds_infos['features'],
                    classes=ds_infos['classes'],
                    colors=ds_infos['colors']
                )
                unpruned_fig_path = final_fig_outputdir / (
                    f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{obj}_Unpruned.png"
                )
                with open(unpruned_fig_path, "wb") as f:
                    f.write(unpruned_graph.create_png())


###############################################################################
# 2) Main pipeline
###############################################################################
def run_all_processes():
    total_runtime_eval = 0
    total_runtime_rf = 0
    aggregated_results = []
    start_time = datetime.now()
    log_time("Script started")

    print("\n" + "=="*50 + "\n")
    print("CREATING RandomForests...")
    print("\n" + "=="*50 + "\n")

    # (A) Create Random Forests
    total_rf_iterations = (
        len(current_dataset_loop)*len(current_fold_loop)*len(n_trees_loop)*len(max_tree_depth_loop)
    )
    rf_progress = tqdm(
        total=total_rf_iterations,
        desc="Creating and Evaluating Random Forests",
        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]",
        dynamic_ncols=True
    )

    t_start_rf = time.time() # Start Time RandomForests
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

                    print('--'*50)
                    log_time(
                        f"Creating RandomForest => DS={current_dataset}, "
                        f"fold=x{current_fold}, trees={n_trees}, depth={max_tree_depth}"
                    )
                    t0 = time.time()
                    rf_model, rf_file = rf.create_random_forest(
                        X_train, y_train,
                        current_dataset, current_fold,
                        max_tree_depth, n_trees,
                        return_file=True
                    )
                    t1 = time.time()
                    log_time('DONE')
                    rf_runtime = t1 - t0
                    print(f'Runtime: {round(rf_runtime,3)}s')

                    # Evaluate
                    t_start_eval = time.time()
                    log_time('Evaluating')
                    rf_test_pred   = rf_model.predict(X_test)
                    rf_train_pred  = rf_model.predict(X_train)
                    report_rf      = classification_report(y_test,  rf_test_pred,   output_dict=True)
                    report_rf_tr   = classification_report(y_train, rf_train_pred, output_dict=True)

                    rf_leaves=0; rf_nodes=0; rf_apl=0; rf_dar=0; rf_dsr=0; rf_zhou=0
                    for est in rf_model.estimators_:
                        rf_leaves += est.tree_.n_leaves
                        rf_nodes  += est.tree_.node_count
                        rf_apl    += calculate_APL(est, X_test)
                        rf_dar    += calculate_DAR(est, X_test)
                        rf_dsr    += calculate_DSR(est)
                        rf_zhou   += zhou_interpretability(est)

                    aggregated_results.append({
                        "Dataset":   current_dataset,
                        "Trees":     n_trees,
                        "Max Depth": max_tree_depth,
                        "Method":    "RandomForest",
                        "Train Acc": round(report_rf_tr["accuracy"], 3),
                        "Train F1":  round(report_rf_tr["weighted avg"]["f1-score"], 3),
                        "Test Acc":  round(report_rf["accuracy"], 3),
                        "Test F1":   round(report_rf["weighted avg"]["f1-score"], 3),
                        "Leaves":    rf_leaves,
                        "Nodes":     rf_nodes,
                        "Avg Path Length": round(rf_apl, 3),
                        "DAR":       round(rf_dar, 3),
                        "DSR":       round(rf_dsr, 3),
                        "Zhou score": round(rf_zhou, 3),
                        "Runtime(s)":   round(rf_runtime, 3),
                        "Runtime(m)":   round(rf_runtime/60, 3)
                    })


                    # Optional: save a figure for the first tree if you want
                    rf_trees = [e.tree_ for e in rf_model.estimators_]

                    final_fig_outputdir = figure_output_dir / current_dataset
                    final_fig_outputdir.mkdir(parents=True, exist_ok=True)
                    if n_trees in [0] and max_tree_depth in [0]:
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


                    # partial CSV
                    df_partial = pd.DataFrame(aggregated_results)
                    df_partial["Method"] = pd.Categorical(
                        df_partial["Method"],
                        categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
                        ordered=True
                    )
                    df_partial.sort_values(["Dataset","Trees","Max Depth","Method"], inplace=True)
                    df_partial.to_csv(PARTIAL_CSV_PATH, index=False)
                    t_end_eval = time.time()
                    t_eval = t_end_eval - t_start_eval
                    log_time('DONE')
                    print(f'Evaluation Runtime: {round(t_eval, 3)}s')
                    print('--'*50)


                    rf_progress.update(1)
                    
                    
                    total_runtime_eval += t_eval
                    total_runtime_rf += rf_runtime
    t_end_rf = time.time() # End Time Random Forests
    rf_progress.close()
    total_runtime_1 = t_end_rf - t_start_rf # Total Runtime RandomForests

    print('\n' + '=='*50)
    print(f'Total Runtime Creating RandomForests: {round(total_runtime_rf,3)}s')
    print(f'Total Runtime Evaluation: {round(total_runtime_eval,3)}s')
    print(f'Total Runtime (RF): {round(total_runtime_1,3)}s')
    print('=='*50 + '\n')

#==============================================================================
# Running BornAgain 
#==============================================================================
    # (B) Run BornAgain => measure external exe time
    completed_runs, run_times = run_born_again(
        current_obj_loop,
        current_dataset_loop,
        current_fold_loop,
        n_trees_loop,
        max_tree_depth_loop,
        aggregated_results,      # pass the same list
        PARTIAL_CSV_PATH
    )


    # C) Final Output
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
    print("\n" + "="*50, "\nFINAL RESULTS:\n" + "="*50 , results_df, "\n")
    log_time("Script completed")

    total_runtime = end_time - start_time
    print("="*50)
    print(f"Total Run Time (RF + BA): {str(total_runtime)}")
    print("="*50)
    return results_df, total_runtime

if __name__ == "__main__":
    results_df, total_runtime = run_all_processes()

#

###################################################################################
# Visualizing the Trees
###################################################################################

#==================================================================================
# Customize which trees are to be visualized
# Leave these as comments in case you want to visualize all trees
# !!! Specify inside the loop which trees should be visualized for each method !!!
#==================================================================================
#current_dataset_loop = ['HTRU2']
#current_fold_loop = [1]
#n_trees_loop = [5, 10]
#current_obj_loop = [4]
#max_tree_depth_loop = [3, 4, 10]

methods = ['RandomForest',  'BornAgain', 'BornAgain_pruned']

print("\n" + "=="*50 + "\n")
print("VISUALIZING TREES...")
print("\n" + "=="*50 + "\n")

ba_progress_total = (
    len(current_dataset_loop)*len(current_fold_loop)*len(n_trees_loop)
    *len(max_tree_depth_loop)*len(current_obj_loop)*len(methods)
)
ba_progress = tqdm(
    total=ba_progress_total,
    desc="Visualizing Trees",
    bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]"
    )

total_start = datetime.now() # Start time visualizing
for obj in current_obj_loop:
    for current_dataset in current_dataset_loop:
        print("\n" + "=="*50)
        print(f"DATASET RUNNING: {current_dataset})")
        print("=="*50 + "\n")
        t_start = time.time() # Start time current dataset

        for current_fold in current_fold_loop:
            # load data
            df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
            X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
            X_test,  y_test  = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

            for n_trees in n_trees_loop:
                for max_tree_depth in max_tree_depth_loop:
                    for method in methods:
                        #===============================================================
                        # Customize which trees should be visualized for each method
                        #===============================================================
                        if method == 'RandomForest' and n_trees not in [5,10,50,100,250,500] or max_tree_depth not in [3,4,7]:
                            continue
                        if method == 'BornAgain' and n_trees not in [5,10,50,100,250,500] or max_tree_depth not in [3,4,7]:
                            continue
                        if method == 'BornAgain_pruned' and n_trees not in [5,10,50,100,250,500] or max_tree_depth not in [3,4,7]:
                            continue
                        
                        Visualize_Tree(method=method, 
                                       current_dataset=current_dataset,
                                       current_fold=current_fold, 
                                       n_trees=n_trees,
                                       max_tree_depth=max_tree_depth, 
                                       X_train=X_train,
                                       y_train=y_train, 
                                       ds_infos=ds_infos, 
                                       ba_progress=ba_progress,
                                       obj=obj)
                        
                        ba_progress.update(1)
        t_end = time.time() # End time current dataset
        runtime_dataset = t_end - t_start
        print('\n' + '--'*50)
        print(f'Runtime Dataset: {round(runtime_dataset,3)}s')
        print('--' * 50 + '\n')
                        
total_end = datetime.now() # End time visualizing
total_runtime_3 = total_end - total_start
ba_progress.close()

print('\n' + '=='*50)
print(f'Total Runtime (Visualizing): {total_runtime_3}')
print('=='*50)

Complete_runtime = total_runtime + total_runtime_3
print('\n' + '=='*50)
print(f'COMPLETE TOTAL RUNTIME (RF + BA + Visualizing): {Complete_runtime}')
print('=='*50)


#%%

print('Total seconds RF:', np.sum(results_df[results_df['Method'] == 'RandomForest']['Runtime(s)']))
print('Total minutes RF:', np.sum(results_df[results_df['Method'] == 'RandomForest']['Runtime(m)']))
print('\n')
print('Total seconds BA:', np.sum(results_df[results_df['Method'] == 'BornAgain']['Runtime(s)']))
print('Total minutes BA:', np.sum(results_df[results_df['Method'] == 'BornAgain']['Runtime(m)']))




