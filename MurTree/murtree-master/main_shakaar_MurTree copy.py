#%% =============================================================================
# main_murtree_pipeline.py
# =============================================================================

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import datasets as ds
import random_forests as rf
import visualization as tree_view

# Path to MurTree
MURTREE_PATH = r"C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master"
MURTREE_EXE = r"C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master\murtree-master\bin\Debug\murtree.exe"


# Function to create distillation dataset for MurTree
def create_distillation_dataset(X, rf_predictions, feature_names, output_csv):
    data = np.column_stack([X, rf_predictions])
    columns = list(feature_names) + ["RF_label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"[MurTree Integration] Distillation dataset saved at {output_csv}")


# Function to parse MurTree output
def parse_murtree_output(mur_output_file):
    try:
        with open(mur_output_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Error: Output file {mur_output_file} is empty.")
            return None

        # Extract metrics or any relevant information
        # Example parsing: Assume first line is the runtime, second line is misclassifications, etc.
        runtime = float(lines[0].strip())  # First line: runtime in seconds
        misclassifications = int(lines[3].strip())  # Fourth line: misclassification score
        
        print(f"[MurTree Integration] Runtime: {runtime}s, Misclassifications: {misclassifications}")
        
        # Placeholder for returning a model or relevant output
        return {"runtime": runtime, "misclassifications": misclassifications}

    except FileNotFoundError:
        print(f"Error: MurTree output file {mur_output_file} not found.")
        return None


# Function to run MurTree and evaluate
def run_murtree_distillation(current_dataset, current_fold, n_trees, max_tree_depth):
    try:
        # Load dataset
        df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
        X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
        X_test, y_test = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values
        feature_names = ds_infos.get('features', [f"feat_{i}" for i in range(X_train.shape[1])])

        # Train random forest
        random_forest, rf_file = rf.create_random_forest(X_train, y_train, current_dataset, current_fold, max_tree_depth, n_trees, return_file=True)
        rf_train_pred = random_forest.predict(X_train)
        rf_test_pred = random_forest.predict(X_test)
        rf_leaves = sum(tree.tree_.n_leaves for tree in random_forest.estimators_)

        # Create distillation dataset
        os.makedirs("output_new", exist_ok=True)
        distill_csv = f"output_new/distill_{current_dataset}_fold{current_fold}_T{n_trees}_D{max_tree_depth}.csv"
        create_distillation_dataset(X_train, rf_train_pred, feature_names, distill_csv)

        # Run MurTree
        mur_output_file = distill_csv.replace(".csv", "_model.txt")
        mur_cmd = [
            MURTREE_EXE,
            "-file", distill_csv,  # Path to the distillation dataset
            "-max-depth", str(max_tree_depth),  # Tree depth
            "-max-num-nodes", str(max_tree_depth + 1),  # Number of nodes (depth + 1 for safety)
            "-time", "600",  # Runtime in seconds
            "-result-file", mur_output_file,  # Where results will be saved
            "-verbose"  # Enable detailed logging
        ]

        print(f"[MurTree Integration] Running: {' '.join(mur_cmd)}")
        ret = subprocess.run(mur_cmd, capture_output=True, text=True)
        print(f"[MurTree Integration] MurTree stdout:\n{ret.stdout}")
        print(f"[MurTree Integration] MurTree stderr:\n{ret.stderr}")
        if ret.returncode != 0:
            print(f"Error: MurTree command failed with code {ret.returncode}")
            return None


        # Parse MurTree output
        single_tree_model = parse_murtree_output(mur_output_file)

        if not single_tree_model:
            return None

        # Evaluate single tree
        st_train_pred = single_tree_model.predict(X_train)
        st_test_pred = single_tree_model.predict(X_test)
        st_leaves = single_tree_model.n_leaves

        # Metrics
        metrics = {
            "Dataset": current_dataset,
            "Fold": current_fold,
            "Trees": n_trees,
            "Max Depth": max_tree_depth,
            "RF_TrainAcc": classification_report(y_train, rf_train_pred, output_dict=True)['accuracy'],
            "RF_TestAcc": classification_report(y_test, rf_test_pred, output_dict=True)['accuracy'],
            "RF_Leaves": rf_leaves,
            "ST_TrainAcc": classification_report(y_train, st_train_pred, output_dict=True)['accuracy'],
            "ST_TestAcc": classification_report(y_test, st_test_pred, output_dict=True)['accuracy'],
            "ST_Leaves": st_leaves
        }

        # Visualize single tree
        try:
            mur_graph = tree_view.create_graph(
                [single_tree_model],  # Replace with parsed structure
                features=feature_names,
                classes=ds_infos['classes'],
                colors=ds_infos['colors']
            )
            output_path = f"output_new/visualization_{current_dataset}_fold{current_fold}_T{n_trees}_D{max_tree_depth}.png"
            mur_graph.write_png(output_path)
            print(f"[MurTree Integration] Tree visualization saved at {output_path}")
        except Exception as e:
            print(f"Visualization failed: {e}")

        return metrics

    except Exception as e:
        print(f"Error in distillation pipeline: {e}")
        return None


# Function to run pipeline for all datasets
def run_all_datasets_murtree():
    datasets = ["AGE"]
    folds = [1, 2, 3]
    n_trees_list = [5, 10]
    max_depths = [10]

    results = []
    for dsname in datasets:
        for fold in folds:
            for n_trees in n_trees_list:
                for max_depth in max_depths:
                    print(f"Running MurTree for Dataset={dsname}, Fold={fold}, Trees={n_trees}, Depth={max_depth}")
                    metrics = run_murtree_distillation(dsname, fold, n_trees, max_depth)
                    if metrics:
                        results.append(metrics)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("output_new/murtree_results.csv", index=False)
    print("Results saved to output_new/murtree_results.csv")

    # Return results_df for visibility
    return results_df



results_df = run_all_datasets_murtree()

