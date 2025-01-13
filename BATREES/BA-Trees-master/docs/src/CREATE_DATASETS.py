#%%

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

datasets = [
    {
        "dataset_path": "C:/Users/shaka/OneDrive/Email attachments/Documenten/Data Science Project/BATREES/BA-Trees-master/docs/src/resources/datasets/AGE/age_predictions_cleaned.csv",
        "dataset_name": "AGE"
    },
    #{
     #   "dataset_path": "C:/Users/shaka/OneDrive/Email attachments/Documenten/Data Science Project/BATREES/BA-Trees-master/docs/src/resources/datasets/AnotherDataset/dataset2.csv",
      #  "dataset_name": "Another-Dataset"
    #},
    
]


for dataset_info in datasets:
    dataset_path = dataset_info["dataset_path"]
    dataset_name = dataset_info["dataset_name"]

    # Load dataset
    print(f"Processing dataset: {dataset_name}")
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}'. Error: {e}")
        continue
    data = pd.read_csv(dataset_path)
        
    # Assuming the last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Create output directory for the dataset
    os.makedirs(f"{output_dir}/{dataset_name}", exist_ok=True)

    # 10-fold stratified splitting
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        # Split the data
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Save the splits
        train_path = f"{output_dir}/{dataset_name}/{dataset_name}.train{fold}.csv"
        test_path = f"{output_dir}/{dataset_name}/{dataset_name}.test{fold}.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

    print(f"Dataset '{dataset_name}' split into 10 folds and saved in {output_dir}/{dataset_name}")

#%%
