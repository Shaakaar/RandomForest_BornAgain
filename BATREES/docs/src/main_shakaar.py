#%%
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from sklearn.metrics import classification_report
source_path = os.path.abspath('../src')
output_path = os.path.abspath('../output')
sys.path.append(source_path)
import datasets as ds
import random_forests as rf
import persistence as tree_io
import visualization as tree_view
from pathlib import Path


selected_n_obj = rf.create_objective_selection()
selected_n_tree = rf.create_n_trees_selection()
selected_kfold = ds.create_kfold_selection()
selected_datasets = ds.create_dataset_selection() 
selected_cplex = ds.create_cplex_linking_selection()
selected_depth = rf.create_depth_selection()

# Loading Parameters...
current_obj = selected_n_obj.value
current_dataset=ds.dataset_names[selected_datasets.index]
current_fold = selected_kfold.value
n_trees = selected_n_tree.value
using_cplex = selected_cplex.value
max_tree_depth = 4

#%%


current_obj_loop = [4]
current_dataset_loop = ['Breast-Cancer-Wisconsin', 'COMPAS-ProPublica', 'FICO',
                        'HTRU2', 'Prima-Diabetes']
current_fold_loop = [1]
n_trees_loop = [5, 10, 250, 500]
using_cplex_loop = False
max_tree_depth_loop = [3, 4, 10]



for current_dataset in current_dataset_loop:
    for current_fold in current_fold_loop:
        for n_trees in n_trees_loop:
            for max_tree_depth in max_tree_depth_loop:
                print('Selected parameters:\n')
                print('  Fold:', current_fold)
                print('  Objective:', selected_n_obj.label)
                print('  No. of trees:', n_trees)
                print('  Dataset:', current_dataset)
                print('  Using CPLEX:', using_cplex)

                # Loading data 
                df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                X_train, y_train = df_train.iloc[:,:-1].values, df_train.iloc[:,-1].values
                X_test, y_test = df_test.iloc[:,:-1].values, df_test.iloc[:,-1].values


                # Directly create the random forest instead of loading from a file
                random_forest, random_forest_file = rf.create_random_forest(
                    X_train, y_train, current_dataset, current_fold, max_tree_depth, n_trees, return_file=True
                )

                # Extract individual decision trees from the random forest
                rf_trees = [e.tree_ for e in random_forest.estimators_]


                if 0 == os.system('make --directory=../src/born_again_dp {} > buildlog.txt'.format('withCPLEX=1' if using_cplex else '')):
                    print('Dynamic Program was successful built.')
                else:
                    print('Error while compiling the program with the make commend. Please verify that a suitable compiler is available.')
                    os.system('make --directory=../src/born_again_dp')



                #display(Image(tree_view.create_graph(rf_trees, features=ds_infos['features'], classes=ds_infos['classes'], colors=ds_infos['colors']).create_png()))



def run_all_datasets(objective, datasets, folds, n_trees, cplex, tree_depth):
    # 1. Create the directories if they don't exist yet
    output_new_dir = Path("C:/Users/shaka//Downloads/RandomForest_BornAgain-master/RandomForest_BornAgain-master/BATREES/BA-Trees-master/docs/src/output_new")
    born_again_dir = output_new_dir / "Born_Again"
    
    output_new_dir.mkdir(parents=True, exist_ok=True)
    born_again_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create subfolders for each dataset
    for n in datasets:
        dataset_dir = born_again_dir / n
        dataset_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define the loops
    objectives = objective
    folds = folds
    trees_list = n_trees
    tree_depth = tree_depth

    exe_path = r"C:\Users\shaka\Downloads\RandomForest_BornAgain-master\RandomForest_BornAgain-master\BATREES\BA-Trees-master\docs\src\born_again_dp\bornAgain.exe"

    # 4. For each combination, run the bornAgain executable
    for o in objectives:
        for n in datasets:
            for u in folds:
                for t in trees_list:
                    for d in tree_depth:
                        rf_input = f"C:/Users/shaka/Downloads/RandomForest_BornAgain-master/RandomForest_BornAgain-master/BATREES/BA-Trees-master/docs/src/output_new/RF/{n}/{n}.RF{u}.T{t}.D{d}.txt"
                        ba_output = f"C:/Users/shaka/Downloads/RandomForest_BornAgain-master/RandomForest_BornAgain-master/BATREES/BA-Trees-master/docs/src/output_new/Born_Again/{n}/{n}.BA{u}.O{o}.T{t}.D{d}"

                        # Build the command
                        cmd = [
                            exe_path,  # or "./bornAgain.exe", or an absolute path
                            rf_input,
                            ba_output,
                            "-trees",
                            str(t),
                            "-obj",
                            str(o)
                        ]

                        print(f"Running: {' '.join(cmd)}")
                        ret = subprocess.run(cmd)
                        if ret.returncode != 0:
                            print(f"Error: Command {cmd} failed with code {ret.returncode}")
                            # Optionally break or handle error

run_all_datasets(current_obj_loop, current_dataset_loop, current_fold_loop, n_trees_loop, using_cplex_loop, max_tree_depth_loop)


# Create directories for saving figures
figure_output_dir = Path("output_new/Figures")
os.makedirs(figure_output_dir, exist_ok=True)

for n in current_dataset_loop:
    final_fig_outputdir = figure_output_dir/n
    final_fig_outputdir.mkdir(parents = True, exist_ok=True)

# Initialize results dictionary
aggregated_results = []

for current_obj in current_obj_loop:
    for current_dataset in current_dataset_loop:
        for n_trees in n_trees_loop:
            for max_tree_depth in max_tree_depth_loop:

                final_fig_outputdir = f"{figure_output_dir}/{current_dataset}"
                # Temporary storage for fold-specific results
                fold_metrics = {
                    "RandomForest": {"Train Acc": [], "Train F1": [], "Test Acc": [], "Test F1": [], "Leaves": []},
                    "BornAgain": {"Train Acc": [], "Train F1": [], "Test Acc": [], "Test F1": [], "Leaves": []},
                    "BornAgain-Pruned": {"Train Acc": [], "Train F1": [], "Test Acc": [], "Test F1": [], "Leaves": []},
                }

                for current_fold in current_fold_loop:
                    
                    # Loading data 
                    df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
                    X_train, y_train = df_train.iloc[:,:-1].values, df_train.iloc[:,-1].values
                    X_test, y_test = df_test.iloc[:,:-1].values, df_test.iloc[:,-1].values

                    # Define paths
                    born_again_file = f"output_new/Born_Again/{current_dataset}/{current_dataset}.BA{current_fold}.O{current_obj}.T{n_trees}.D{max_tree_depth}"
                    random_forest_file = f"output_new/RF/{current_dataset}/{current_dataset}.RF{current_fold}.T{n_trees}.D{max_tree_depth}.txt"

                    # Load Random Forest
                    random_forest = tree_io.classifier_from_file(random_forest_file, X_train, y_train, pruning=False)

                    # Load Born-Again Trees
                    born_again = tree_io.classifier_from_file(born_again_file + ".tree", X_train, y_train, pruning=False)
                    born_again_pruned = tree_io.classifier_from_file(born_again_file + ".tree", X_train, y_train, pruning=True)

                    # Visualize the pruned Born-Again tree
                    pruned_graph = tree_view.create_graph(
                        [born_again_pruned.tree_],  # Pass the tree object
                        features=ds_infos['features'],  # Feature names
                        classes=ds_infos['classes'],  # Class labels
                        colors=ds_infos['colors']  # Colors for visualization
                    )

                    # Save pruned graph as PNG
                    pruned_output_path = os.path.join(
                        final_fig_outputdir,
                        f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{current_obj}_Pruned.png"
                    )
                    with open(pruned_output_path, "wb") as f:
                        f.write(pruned_graph.create_png())
                    
                    if n_trees == 5:
                        #Save unpruned graph (optional)
                        unpruned_graph = tree_view.create_graph(
                            [born_again.tree_],
                            features=ds_infos['features'],
                            classes=ds_infos['classes'],
                            colors=ds_infos['colors']
                        )
                        unpruned_output_path = os.path.join(
                            final_fig_outputdir,
                            f"{current_dataset}_F{current_fold}_T{n_trees}_D{max_tree_depth}_O{current_obj}_Unpruned.png"
                        )
                        with open(unpruned_output_path, "wb") as f:
                            f.write(unpruned_graph.create_png())

                    # Calculate leaves
                    total_rf_leaves = np.sum([tree.tree_.n_leaves for tree in random_forest.estimators_])
                    ba_leaves = born_again.tree_.n_leaves
                    ba_pruned_leaves = born_again_pruned.tree_.n_leaves

                    # Evaluate Random Forest
                    rf_test_pred = random_forest.predict(X_test)
                    rf_train_pred = random_forest.predict(X_train)
                    report_rf = classification_report(y_test, rf_test_pred, output_dict=True)
                    report_rf_train = classification_report(y_train, rf_train_pred, output_dict=True)

                    # Evaluate Born-Again Tree
                    ba_test_pred = born_again.predict(X_test)
                    ba_train_pred = born_again.predict(X_train)
                    report_ba = classification_report(y_test, ba_test_pred, output_dict=True)
                    report_ba_train = classification_report(y_train, ba_train_pred, output_dict=True)

                    # Evaluate Pruned Born-Again Tree
                    ba_pruned_test_pred = born_again_pruned.predict(X_test)
                    ba_pruned_train_pred = born_again_pruned.predict(X_train)
                    report_ba_pruned = classification_report(y_train, ba_pruned_train_pred, output_dict=True)
                    report_ba_pruned_train = classification_report(y_train, ba_pruned_train_pred, output_dict=True)

                    # Store fold-specific results
                    fold_metrics["RandomForest"]["Train Acc"].append(report_rf_train['accuracy'])
                    fold_metrics["RandomForest"]["Train F1"].append(report_rf_train['weighted avg']['f1-score'])
                    fold_metrics["RandomForest"]["Test Acc"].append(report_rf['accuracy'])
                    fold_metrics["RandomForest"]["Test F1"].append(report_rf['weighted avg']['f1-score'])
                    fold_metrics["RandomForest"]["Leaves"].append(total_rf_leaves)

                    fold_metrics["BornAgain"]["Train Acc"].append(report_ba_train['accuracy'])
                    fold_metrics["BornAgain"]["Train F1"].append(report_ba_train['weighted avg']['f1-score'])
                    fold_metrics["BornAgain"]["Test Acc"].append(report_ba['accuracy'])
                    fold_metrics["BornAgain"]["Test F1"].append(report_ba['weighted avg']['f1-score'])
                    fold_metrics["BornAgain"]["Leaves"].append(ba_leaves)

                    fold_metrics["BornAgain-Pruned"]["Train Acc"].append(report_ba_pruned_train['accuracy'])
                    fold_metrics["BornAgain-Pruned"]["Train F1"].append(report_ba_pruned_train['weighted avg']['f1-score'])
                    fold_metrics["BornAgain-Pruned"]["Test Acc"].append(report_ba_pruned['accuracy'])
                    fold_metrics["BornAgain-Pruned"]["Test F1"].append(report_ba_pruned['weighted avg']['f1-score'])
                    fold_metrics["BornAgain-Pruned"]["Leaves"].append(ba_pruned_leaves)

                # Compute mean across folds
                for method in fold_metrics:
                    aggregated_results.append({
                        "Dataset": current_dataset,
                        "Trees": n_trees,
                        "Max Depth": max_tree_depth,
                        "Method": method,
                        "Train Acc": np.mean(fold_metrics[method]["Train Acc"]),
                        "Train F1": np.mean(fold_metrics[method]["Train F1"]),
                        "Test Acc": np.mean(fold_metrics[method]["Test Acc"]),
                        "Test F1": np.mean(fold_metrics[method]["Test F1"]),
                        "Leaves": np.mean(fold_metrics[method]["Leaves"]),
                    })

# Convert aggregated results to a DataFrame
results_df = pd.DataFrame(aggregated_results)



    # %%
print(results_df)
