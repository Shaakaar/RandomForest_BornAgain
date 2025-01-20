#%% 
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

# Import your project-specific modules
import datasets as ds
import random_forests as rf
import persistence as tree_io
import visualization as tree_view

# Helper function for logging timestamps
def log_time(message):
    print(f"{message}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Create directories for output
figure_output_dir = Path("output_new/Figures")
figure_output_dir.mkdir(parents=True, exist_ok=True)

# Parameters for the loops
current_dataset_loop = ['COMPAS-ProPublica', 'FICO']
current_fold_loop = [1]
n_trees_loop = [5, 10]
current_obj_loop = [4]
max_tree_depth_loop = [3, 4]

###############################################################################
# 1. This function calls the .exe to produce Born-Again .tree files
###############################################################################
def run_born_again(objective, datasets, folds, n_trees, tree_depth):
    """
    Runs the bornAgain.exe for each combination of:
    - objective in objective
    - dataset in datasets
    - fold in folds
    - trees in n_trees
    - depth in tree_depth

    The .tree files will appear in output_new/Born_Again/<dataset>/
    """
    output_new_dir = Path("C:/School/Case Study Econometrics and Data Science/RandomForest_BornAgain-master/RandomForest_BornAgain-master/BATREES/BA-Trees-master/docs/src/output_new")
    born_again_dir = output_new_dir / "Born_Again"
    exe_path = r"C:\School\Case Study Econometrics and Data Science\RandomForest_BornAgain-master\RandomForest_BornAgain-master\BATREES\BA-Trees-master\docs\src\born_again_dp\bornAgain.exe"

    output_new_dir.mkdir(parents=True, exist_ok=True)
    born_again_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create subfolders for each dataset
    for dsname in datasets:
        dataset_dir = born_again_dir / dsname
        dataset_dir.mkdir(parents=True, exist_ok=True)

    # For progress tracking
    total_iterations = len(objective) * len(datasets) * len(folds) * len(n_trees) * len(tree_depth)
    progress = tqdm(total=total_iterations, desc="Running Born Again Trees", bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    for o in objective:
        for dataset in datasets:
            print('\n')
            print('=='*50 + '\n' + f'DATASET RUNNING: {dataset} (BORN AGAIN)' + '\n' + '=='*50)
            for fold in folds:
                for trees in n_trees:
                    for depth in tree_depth:
                        # Example: COMPAS-ProPublica.BA1.O4.T10.D4.tree
                        rf_input = output_new_dir / f"RF/{dataset}/{dataset}.RF{fold}.T{trees}.D{depth}.txt"
                        ba_output = born_again_dir / f"{dataset}/{dataset}.BA{fold}.O{o}.T{trees}.D{depth}"

                        cmd = [
                            str(exe_path),
                            str(rf_input),
                            str(ba_output),
                            "-trees",
                            str(trees),
                            "-obj",
                            str(o)
                        ]
                        log_time(f"Running Born Again: Dataset={dataset}, Fold={fold}, Trees={trees}, Depth={depth}, Objective={o}")
                        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if ret.returncode != 0:
                            print(f"Error: Command {' '.join(cmd)} failed with code {ret.returncode}")
                            print(ret.stderr.decode("utf-8"))
                        else:
                            print(f"Completed: {' '.join(cmd)}")
                        progress.update(1)
    progress.close()

###############################################################################
# 2. The main pipeline
###############################################################################
def run_all_processes():
    """
    1) Loops over datasets/folds/n_trees/depth to create random forests
    2) Then calls run_born_again(...) to produce .tree files
    3) Then re-loads the BornAgain trees + does classification & stats
    4) Aggregates results into a DataFrame
    """
    aggregated_results = []

    start_time = datetime.now()

    print("=="*50 + '\n')
    log_time("Script started") 
    print('\n' + '=='*50 + '\n')

    ###########################################################################
    # 2.1: Create Random Forests with rf.create_random_forest(), measure their performance
    ###########################################################################
    total_rf_iterations = len(current_dataset_loop) * len(current_fold_loop) * len(n_trees_loop) * len(max_tree_depth_loop)
    rf_progress = tqdm(total=total_rf_iterations, desc="Creating Random Forests", bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    # We'll store some basic info about the random forests in aggregated_results
    for current_dataset in current_dataset_loop:
        print('\n')
        print('=='*50 + '\n' + f'DATASET RUNNING: {current_dataset} (RANDOM FOREST)' + '\n' + '=='*50)
        for current_fold in current_fold_loop:
            for n_trees in n_trees_loop:
                for max_tree_depth in max_tree_depth_loop:
                    print(f"Processing Dataset: {current_dataset}, Fold: {current_fold}, Trees: {n_trees}, Depth: {max_tree_depth}")

                    # Load data
                    df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                    X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
                    X_test, y_test = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

                    # Create Random Forest
                    log_time(f"Creating Random Forest for Dataset: {current_dataset}, Fold: {current_fold}")
                    rf_model, rf_file = rf.create_random_forest(
                        X_train, y_train, current_dataset, current_fold, max_tree_depth, n_trees, return_file=True
                    )

                    rf_trees = [e.tree_ for e in rf_model.estimators_]

                    # Evaluate Random Forest
                    rf_test_pred = rf_model.predict(X_test)
                    rf_train_pred = rf_model.predict(X_train)
                    report_rf = classification_report(y_test, rf_test_pred, output_dict=True)
                    report_rf_train = classification_report(y_train, rf_train_pred, output_dict=True)

                    # Number of leaves in the entire forest
                    rf_leaves = np.sum([est.tree_.n_leaves for est in rf_model.estimators_])

                    # Append results: "RandomForest" row
                    aggregated_results.append({
                        "Dataset": current_dataset,
                        "Trees": n_trees,
                        "Max Depth": max_tree_depth,
                        "Method": "RandomForest",
                        "Train Acc": report_rf_train['accuracy'],
                        "Train F1": report_rf_train['weighted avg']['f1-score'],
                        "Test Acc": report_rf['accuracy'],
                        "Test F1": report_rf['weighted avg']['f1-score'],
                        "Leaves": rf_leaves
                    })

                    # (Optional) save graphs
                    # e.g. random forest graph
                    final_fig_outputdir = figure_output_dir / current_dataset
                    final_fig_outputdir.mkdir(parents=True, exist_ok=True)

                    random_forest = tree_io.classifier_from_file(rf_file, X_train, y_train, pruning=False)
                    if n_trees in [5,10,100] and max_tree_depth in [3]:
                        rf_graph = tree_view.create_graph(
                            rf_trees,
                            features=ds_infos['features'],
                            classes=ds_infos['classes'],
                            colors=ds_infos['colors']
                        )
                        
                        rf_path = final_fig_outputdir / f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_RF.png"
                            
                        with open(rf_path, "wb") as f:
                            f.write(rf_graph.create_png())

                    rf_progress.update(1)

    rf_progress.close()



    ###########################################################################
    # 2.2: Run BornAgain to produce .tree files
    ###########################################################################
    run_born_again(current_obj_loop, current_dataset_loop, current_fold_loop, n_trees_loop, max_tree_depth_loop)

    ###########################################################################
    # 2.3: Load BornAgain trees from .tree files, measure performance
    ###########################################################################
    # We'll read the same data again & evaluate:
    # The path will be: "output_new/Born_Again/{dataset}/{dataset}.BA{fold}.O{o}.T{trees}.D{depth}.tree"

    # Another loop to do BornAgain & BornAgain-Pruned results
    # (We re-run the loop so we can re-load data & the .tree)
    ba_progress_total = len(current_dataset_loop) * len(current_fold_loop) * len(n_trees_loop) * len(max_tree_depth_loop) * len(current_obj_loop)
    ba_progress = tqdm(total=ba_progress_total, desc="Evaluating Born-Again", bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    for o in current_obj_loop:
        for current_dataset in current_dataset_loop:
            for current_fold in current_fold_loop:
                # Load data once per fold
                df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
                X_test, y_test = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

                for n_trees in n_trees_loop:
                    for max_tree_depth in max_tree_depth_loop:
                        # The .tree file we expect
                        ba_base = f"output_new/Born_Again/{current_dataset}/{current_dataset}.BA{current_fold}.O{o}.T{n_trees}.D{max_tree_depth}"
                        ba_treefile = ba_base + ".tree"

                        if not os.path.exists(ba_treefile):
                            # If the .tree file doesn't exist, skip
                            print(f"BornAgain file not found: {ba_treefile}")
                            ba_progress.update(1)
                            continue

                        # Load Born-Again unpruned
                        born_again = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=False)
                        # Load Born-Again pruned
                        born_again_pruned = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=True)

                        # Evaluate Born-Again
                        ba_test_pred = born_again.predict(X_test)
                        ba_train_pred = born_again.predict(X_train)
                        report_ba = classification_report(y_test, ba_test_pred, output_dict=True)
                        report_ba_train = classification_report(y_train, ba_train_pred, output_dict=True)
                        ba_leaves = born_again.tree_.n_leaves

                        aggregated_results.append({
                            "Dataset": current_dataset,
                            "Trees": n_trees,
                            "Max Depth": max_tree_depth,
                            "Method": "BornAgain",
                            "Train Acc": report_ba_train['accuracy'],
                            "Train F1": report_ba_train['weighted avg']['f1-score'],
                            "Test Acc": report_ba['accuracy'],
                            "Test F1": report_ba['weighted avg']['f1-score'],
                            "Leaves": ba_leaves
                        })

                        # Evaluate Born-Again Pruned
                        ba_pruned_test_pred = born_again_pruned.predict(X_test)
                        ba_pruned_train_pred = born_again_pruned.predict(X_train)
                        report_ba_pruned = classification_report(y_test, ba_pruned_test_pred, output_dict=True)
                        report_ba_pruned_train = classification_report(y_train, ba_pruned_train_pred, output_dict=True)
                        ba_pruned_leaves = born_again_pruned.tree_.n_leaves

                        aggregated_results.append({
                            "Dataset": current_dataset,
                            "Trees": n_trees,
                            "Max Depth": max_tree_depth,
                            "Method": "BornAgain-Pruned",
                            "Train Acc": report_ba_pruned_train['accuracy'],
                            "Train F1": report_ba_pruned_train['weighted avg']['f1-score'],
                            "Test Acc": report_ba_pruned['accuracy'],
                            "Test F1": report_ba_pruned['weighted avg']['f1-score'],
                            "Leaves": ba_pruned_leaves
                        })

                        # (Optional) save graphs
                        # pruned graph
                        final_fig_outputdir = figure_output_dir / current_dataset
                        final_fig_outputdir.mkdir(parents=True, exist_ok=True)

                        # Visualize pruned
                        pruned_graph = tree_view.create_graph(
                            [born_again_pruned.tree_],
                            features=ds_infos['features'],
                            classes=ds_infos['classes'],
                            colors=ds_infos['colors']
                        )
                        pruned_output_path = final_fig_outputdir / f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{o}_Pruned.png"
                        with open(pruned_output_path, "wb") as f:
                            f.write(pruned_graph.create_png())

                        # Visualize unpruned if you want
                        if n_trees == 5:
                            unpruned_graph = tree_view.create_graph(
                                [born_again.tree_],
                                features=ds_infos['features'],
                                classes=ds_infos['classes'],
                                colors=ds_infos['colors']
                            )
                            unpruned_output_path = final_fig_outputdir / f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{o}_Unpruned.png"
                            with open(unpruned_output_path, "wb") as f:
                                f.write(unpruned_graph.create_png())

                        ba_progress.update(1)
    ba_progress.close()

    ###########################################################################
    # 2.4: Convert all results to a DataFrame
    ###########################################################################
    results_df = pd.DataFrame(aggregated_results)

    # Convert method to a categorical with a fixed order
    results_df["Method"] = pd.Categorical(
        results_df["Method"],
        categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
        ordered=True
    )

    # Sort by the columns
    results_df = results_df.sort_values(["Dataset", "Trees", "Max Depth", "Method"])
    log_time("Saving results to CSV")
    results_df.to_csv("aggregated_results.csv", index=False)
    end_time = datetime.now()

    print('\n' + "=="*50 + '\n')
    print("FINAL RESULTS:" + '\n')
    print(results_df)
    print('\n' + "=="*50 + '\n')

    print('\n' + "=="*50 + '\n')
    log_time("Script completed")
    print('\n' + '=='*50 + '\n')

    total_runtime = end_time - start_time
    print("=" * 50)
    print(f"Total Run Time: {str(total_runtime)}")
    print("=" * 50)

    return results_df




###############################################################################
# Execute the script
###############################################################################
if __name__ == "__main__":
    results_df = run_all_processes()
