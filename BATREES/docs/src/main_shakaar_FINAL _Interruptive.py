#%% 
# ------------------------------------------------------------------
#  CODE: BornAgain Algorithm 
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

def log_time(message):
    """Print a timestamped log message."""
    print(f"{message}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create directories for output
figure_output_dir = Path("output_new/Figures")
figure_output_dir.mkdir(parents=True, exist_ok=True)

# Parameters for the loops
current_dataset_loop = ['FICO', 'HTRU2']
current_fold_loop = [1]
n_trees_loop = [5, 10]
current_obj_loop = [4]
max_tree_depth_loop = [3, 4, 20]

def run_born_again(objective, datasets, folds, n_trees, tree_depth):
    """
    Runs bornAgain.exe for each combo. If Ctrl+C is pressed mid-run,
    partial combos are returned.
    """
    output_new_dir = Path(
        "C:/School/Case Study Econometrics and Data Science/"
        "RandomForest_BornAgain-master/"
        "BATREES/docs/src/output_new"
    )
    born_again_dir = output_new_dir / "Born_Again"
    exe_path = r"C:\School\Case Study Econometrics and Data Science\RandomForest_BornAgain-master\BATREES\docs\src\born_again_dp\bornAgain.exe"

    output_new_dir.mkdir(parents=True, exist_ok=True)
    born_again_dir.mkdir(parents=True, exist_ok=True)

    for dsname in datasets:
        (born_again_dir / dsname).mkdir(parents=True, exist_ok=True)

    completed_runs = []  # list of (dataset, fold, trees, depth, obj) that finished

    try:
        total_iterations = (len(objective)*len(datasets)*len(folds)*len(n_trees)*len(tree_depth))
        progress = tqdm(total=total_iterations, desc="Running Born Again Trees",
                        bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

        for o in objective:
            for dataset in datasets:
                print("\n" + "=="*50)
                print(f"DATASET RUNNING: {dataset} (BORN AGAIN)")
                print("=="*50 + "\n")
                for fold in folds:
                    for trees in n_trees:
                        for depth in tree_depth:
                            rf_input = output_new_dir / f"RF/{dataset}/{dataset}.RF{fold}.T{trees}.D{depth}.txt"
                            ba_output = born_again_dir / f"{dataset}/{dataset}.BA{fold}.O{o}.T{trees}.D{depth}"

                            cmd = [
                                str(exe_path),
                                str(rf_input),
                                str(ba_output),
                                "-trees", str(trees),
                                "-obj", str(o)
                            ]
                            log_time(
                                f"Running BornAgain => dataset={dataset}, fold={fold}, "
                                f"trees={trees}, depth={depth}, obj={o}"
                            )
                            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            if ret.returncode != 0:
                                print(f"Error: Command {' '.join(cmd)} failed with code {ret.returncode}")
                                print(ret.stderr.decode("utf-8"))
                            else:
                                completed_runs.append((dataset, fold, trees, depth, o))
                                print(f"Completed: {' '.join(cmd)}")
                            progress.update(1)
        progress.close()
        return completed_runs

    except KeyboardInterrupt:
        print("\n" + "="*50 + "\n" + 
              "Interrupted! Returning partial 'completed_runs' so far." 
              + "\n" + "="*50 + "\n")
        return completed_runs


def run_all_processes():
    """
    1) Create random forests, gather results
    2) Run born again (with interrupt support)
    3) Evaluate partial born-again combos
    4) Save final results
    """
    aggregated_results = []

    start_time = datetime.now()
    print("="*100, "\n")
    log_time("Script started")
    print("\n" + "="*100, "\n")

    total_rf_iterations = (len(current_dataset_loop)*len(current_fold_loop)*len(n_trees_loop)*len(max_tree_depth_loop))
    rf_progress = tqdm(total=total_rf_iterations, desc="Creating Random Forests",
                       bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    # -------------------------------
    # (A) Create Random Forests
    # -------------------------------
    for current_dataset in current_dataset_loop:
        print("\n" + "=="*50)
        print(f"DATASET RUNNING: {current_dataset} (RANDOM FOREST)")
        print("=="*50 + "\n")

        for current_fold in current_fold_loop:
            for n_trees in n_trees_loop:
                for max_tree_depth in max_tree_depth_loop:
                    print(f"Processing: DS={current_dataset}, Fold={current_fold}, "
                          f"T={n_trees}, Depth={max_tree_depth}")

                    df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                    X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
                    X_test,  y_test  = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

                    log_time(f"Creating Random Forest (DS={current_dataset}, fold={current_fold})")
                    rf_model, rf_file = rf.create_random_forest(
                        X_train, y_train,
                        current_dataset, current_fold,
                        max_tree_depth, n_trees,
                        return_file=True
                    )
                    rf_trees = [e.tree_ for e in rf_model.estimators_]

                    rf_test_pred   = rf_model.predict(X_test)
                    rf_train_pred  = rf_model.predict(X_train)
                    report_rf      = classification_report(y_test, rf_test_pred, output_dict=True)
                    report_rf_train= classification_report(y_train, rf_train_pred, output_dict=True)

                    rf_leaves = np.sum([est.tree_.n_leaves for est in rf_model.estimators_])

                    aggregated_results.append({
                        "Dataset": current_dataset,
                        "Trees": n_trees,
                        "Max Depth": max_tree_depth,
                        "Method": "RandomForest",
                        "Train Acc": report_rf_train["accuracy"],
                        "Train F1":  report_rf_train["weighted avg"]["f1-score"],
                        "Test Acc":  report_rf["accuracy"],
                        "Test F1":   report_rf["weighted avg"]["f1-score"],
                        "Leaves":    rf_leaves
                    })

                    # (Optional) save graph
                    final_fig_outputdir = figure_output_dir / current_dataset
                    final_fig_outputdir.mkdir(parents=True, exist_ok=True)
                    random_forest_clf = tree_io.classifier_from_file(rf_file, X_train, y_train, pruning=False)
                    if n_trees in [5,10,100] and max_tree_depth in [3]:
                        rf_graph = tree_view.create_graph(
                            rf_trees,
                            features=ds_infos['features'],
                            classes=ds_infos['classes'],
                            colors=ds_infos['colors']
                        )
                        # just pick an obj=4 for naming if you want
                        path_str = f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_RF.png"
                        rf_path = final_fig_outputdir / path_str
                        with open(rf_path, "wb") as f:
                            f.write(rf_graph.create_png())

                    rf_progress.update(1)

    rf_progress.close()

    # -------------------------------
    # (B) Run BornAgain w/ interrupt
    # -------------------------------
    completed_runs = run_born_again(
        current_obj_loop,
        current_dataset_loop,
        current_fold_loop,
        n_trees_loop,
        max_tree_depth_loop
    )

    # -------------------------------
    # (C) Evaluate partial BornAgain
    # -------------------------------
    ba_progress_total = (len(current_dataset_loop)*len(current_fold_loop)*len(n_trees_loop)*len(max_tree_depth_loop)*len(current_obj_loop))
    ba_progress = tqdm(total=ba_progress_total, desc="Evaluating Born-Again",
                       bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]")

    for o in current_obj_loop:
        for current_dataset in current_dataset_loop:
            for current_fold in current_fold_loop:
                # load data for that fold
                df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
                X_test,  y_test  = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

                for n_trees in n_trees_loop:
                    for max_tree_depth in max_tree_depth_loop:
                        # Check if (dataset, fold, n_trees, max_tree_depth, o)
                        # is in completed_runs, else skip
                        if (current_dataset, current_fold, n_trees, max_tree_depth, o) not in completed_runs:
                            print(f"Skipping {current_dataset},fold={current_fold},T={n_trees},D={max_tree_depth},o={o} => not completed.")
                            ba_progress.update(1)
                            continue

                        ba_base = (
                            f"output_new/Born_Again/{current_dataset}/"
                            f"{current_dataset}.BA{current_fold}.O{o}.T{n_trees}.D{max_tree_depth}"
                        )
                        ba_treefile = ba_base + ".tree"

                        if not os.path.exists(ba_treefile):
                            print(f"File missing: {ba_treefile}")
                            ba_progress.update(1)
                            continue

                        log_time(f"Evaluating BornAgain => DS={current_dataset},F={current_fold},T={n_trees},D={max_tree_depth},O={o}")

                        # load unpruned
                        born_again_clf = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=False)
                        # load pruned
                        born_again_pruned_clf = tree_io.classifier_from_file(ba_treefile, X_train, y_train, pruning=True)

                        # Evaluate unpruned
                        ba_test_pred  = born_again_clf.predict(X_test)
                        ba_train_pred = born_again_clf.predict(X_train)
                        rep_ba        = classification_report(y_test, ba_test_pred, output_dict=True)
                        rep_ba_train  = classification_report(y_train, ba_train_pred, output_dict=True)
                        ba_leaves     = born_again_clf.tree_.n_leaves

                        aggregated_results.append({
                            "Dataset": current_dataset,
                            "Trees": n_trees,
                            "Max Depth": max_tree_depth,
                            "Method": "BornAgain",
                            "Train Acc": rep_ba_train["accuracy"],
                            "Train F1":  rep_ba_train["weighted avg"]["f1-score"],
                            "Test Acc":  rep_ba["accuracy"],
                            "Test F1":   rep_ba["weighted avg"]["f1-score"],
                            "Leaves":    ba_leaves
                        })

                        # Evaluate pruned
                        ba_pruned_test_pred  = born_again_pruned_clf.predict(X_test)
                        ba_pruned_train_pred = born_again_pruned_clf.predict(X_train)
                        rep_bap             = classification_report(y_test, ba_pruned_test_pred, output_dict=True)
                        rep_bap_train       = classification_report(y_train, ba_pruned_train_pred, output_dict=True)
                        ba_pruned_leaves    = born_again_pruned_clf.tree_.n_leaves

                        aggregated_results.append({
                            "Dataset": current_dataset,
                            "Trees": n_trees,
                            "Max Depth": max_tree_depth,
                            "Method": "BornAgain-Pruned",
                            "Train Acc": rep_bap_train["accuracy"],
                            "Train F1":  rep_bap_train["weighted avg"]["f1-score"],
                            "Test Acc":  rep_bap["accuracy"],
                            "Test F1":   rep_bap["weighted avg"]["f1-score"],
                            "Leaves":    ba_pruned_leaves
                        })

                        # optional: save pruned/unpruned figures
                        final_fig_outputdir = figure_output_dir / current_dataset
                        final_fig_outputdir.mkdir(parents=True, exist_ok=True)

                        # pruned
                        pruned_graph = tree_view.create_graph(
                            [born_again_pruned_clf.tree_],
                            features=ds_infos['features'],
                            classes=ds_infos['classes'],
                            colors=ds_infos['colors']
                        )
                        pruned_path = final_fig_outputdir / (
                            f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{o}_Pruned.png"
                        )
                        with open(pruned_path, "wb") as f:
                            f.write(pruned_graph.create_png())

                        if n_trees == 5:
                            unpruned_graph = tree_view.create_graph(
                                [born_again_clf.tree_],
                                features=ds_infos['features'],
                                classes=ds_infos['classes'],
                                colors=ds_infos['colors']
                            )
                            unpruned_path = final_fig_outputdir / (
                                f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{o}_Unpruned.png"
                            )
                            with open(unpruned_path, "wb") as f:
                                f.write(unpruned_graph.create_png())

                        ba_progress.update(1)
    ba_progress.close()

    # produce final DataFrame
    results_df = pd.DataFrame(aggregated_results)
    results_df["Method"] = pd.Categorical(
        results_df["Method"],
        categories=["RandomForest", "BornAgain", "BornAgain-Pruned"],
        ordered=True
    )
    results_df.sort_values(["Dataset", "Trees", "Max Depth", "Method"], inplace=True)

    log_time("Saving final CSV")
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
