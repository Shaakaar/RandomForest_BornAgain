


import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from pathlib import Path




# Number of folds for cross-validation
n_splits = 10  # If n_splits = 5, running 1 folder is equal to an 80/20 train/test split

# Different sub-sample sizes you want to create from the full dataset
subsample_sizes = []  # adapt as you wish

datasets = [
    {
        "dataset_path": r"C:\School\Case_BA\RF_BA\BATREES\docs\src\resources\datasets\FICO\FICO.full.csv",
        "dataset_name": "FICO_NEW"
    },
    # You can add more datasets if you like...
]

for dataset_info in datasets:
    dataset_path = dataset_info["dataset_path"]
    dataset_name = dataset_info["dataset_name"]

    # Path for output of all splits
    output_base_dir = Path(f'C:/School/Case_BA/RF_BA/BATREES/docs/src/resources/datasets/')
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Processing dataset: {dataset_name}")
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}'. Error: {e}")
        continue

    # Dynamically add full dataset size to `subsample_sizes`
    full_dataset_size = len(data)
    if full_dataset_size not in subsample_sizes:
        subsample_sizes.append(full_dataset_size)

    # Let's assume last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # For each sub-sample size:
    for sub_size in subsample_sizes:
        if sub_size > len(data):
            print('\n' + '=='*50 + '\n' + f"Requested sub_size={sub_size} but dataset only has {len(data)} rows. Skipping..." + '\n' + '=='*50 + '\n')
            continue

        print(f"\nCreating sub-sample of size {sub_size} for {dataset_name}...")

        if sub_size == len(data):
            # Use the entire dataset as is
            sub_data = data
        else:
            # Fraction for stratified sampling
            fraction = sub_size / len(data)

            # Sub-sample the data using train_test_split
            sub_data, _ = train_test_split(
                data,
                train_size=fraction,
                stratify=y,  # ensures class distribution is preserved
                random_state=42
            )

        # 2) Create sub-directory for this sub-sample
        sub_dir = f"{output_base_dir}/{dataset_name}_sub{sub_size}"
        os.makedirs(sub_dir, exist_ok=True)

        # 3) We do StratifiedKFold on the sub_data
        X_sub = sub_data.iloc[:, :-1].values
        y_sub = sub_data.iloc[:, -1].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # For each fold, we make .train{fold}.csv and .test{fold}.csv
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_sub, y_sub), start=1):
            train_data = sub_data.iloc[train_idx]
            test_data = sub_data.iloc[test_idx]

            train_path = f"{sub_dir}/{dataset_name}_sub{sub_size}.train{fold}.csv"
            test_path = f"{sub_dir}/{dataset_name}_sub{sub_size}.test{fold}.csv"

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

        print(f" => Created {n_splits}-fold splits for sub-sample of size {sub_size}, saved in '{sub_dir}'.")
#%%












#%%

##########################################################################
# REMOVE FEATURES + ADJUST DATASET SIZE
##########################################################################

import random
import pandas as pd


dataset_path = r"C:\School\Case_BA\RF_BA\BATREES\docs\src\resources\datasets\FICO\FICO.full.csv"

# Load dataset
data = pd.read_csv(dataset_path)

# Get columns excluding the last column ("Class")
columns = data.columns[:-1].tolist()  # Exclude the last column
class_column = data.columns[-1]       # Save the last column name separately

# Randomly drop 2 columns (excluding "Class")
columns_to_drop = random.sample(columns, 5)
data_dropped = data.drop(columns=columns_to_drop)

# Ensure "Class" is still the last column
assert data_dropped.columns[-1] == class_column, "Class column should always remain as the last column!!"

print(f"Dropped columns: {columns_to_drop}")



import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from pathlib import Path


# Number of folds for cross-validation
n_splits = 10  # If n_splits = 5, running 1 folder is equal to an 80/20 train/test split

# Different sub-sample sizes you want to create from the full dataset
subsample_sizes = []  # adapt as you wish

datasets = [
    {
        "dataset_name": "FICO_FEATURES"
    }
    # You can add more datasets if you like...
]

for dataset_info in datasets:
    dataset_name = dataset_info["dataset_name"]

    # Path for output of all splits
    output_base_dir = Path(f'C:/School/Case_BA/RF_BA/BATREES/docs/src/resources/datasets/')
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Processing dataset: {dataset_name}")
    data = data_dropped

    # Dynamically add full dataset size to `subsample_sizes`
    full_dataset_size = len(data)
    if full_dataset_size not in subsample_sizes:
        subsample_sizes.append(full_dataset_size)

    # Let's assume last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # For each sub-sample size:
    for sub_size in subsample_sizes:
        if sub_size > len(data):
            print('\n' + '=='*50 + '\n' + f"Requested sub_size={sub_size} but dataset only has {len(data)} rows. Skipping..." + '\n' + '=='*50 + '\n')
            continue

        print(f"\nCreating sub-sample of size {sub_size} for {dataset_name}...")

        if sub_size == len(data):
            # Use the entire dataset as is
            sub_data = data
        else:
            # Fraction for stratified sampling
            fraction = sub_size / len(data)

            # Sub-sample the data using train_test_split
            sub_data, _ = train_test_split(
                data,
                train_size=fraction,
                stratify=y,  # ensures class distribution is preserved
                random_state=42
            )

        # 2) Create sub-directory for this sub-sample
        sub_dir = f"{output_base_dir}/{dataset_name}_sub{sub_size}"
        os.makedirs(sub_dir, exist_ok=True)

        # 3) We do StratifiedKFold on the sub_data
        X_sub = sub_data.iloc[:, :-1].values
        y_sub = sub_data.iloc[:, -1].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # For each fold, we make .train{fold}.csv and .test{fold}.csv
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_sub, y_sub), start=1):
            train_data = sub_data.iloc[train_idx]
            test_data = sub_data.iloc[test_idx]

            train_path = f"{sub_dir}/{dataset_name}_sub{sub_size}.train{fold}.csv"
            test_path = f"{sub_dir}/{dataset_name}_sub{sub_size}.test{fold}.csv"

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

        print(f" => Created {n_splits}-fold splits for sub-sample of size {sub_size}, saved in '{sub_dir}'.")
    


