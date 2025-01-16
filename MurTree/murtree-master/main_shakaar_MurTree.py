#%%
# =============================================================================
# main_murtree_pipeline.py
# =============================================================================
import pybind11
import pandas as pd 
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#%%





#%%
# Path to MurTree
MURTREE_PATH = r"C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master"
MURTREE_EXE = r"C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master\murtree-master\bin\Debug\murtree.exe"


path_to_data = r'C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master\murtree-master\resources\datasets\AGE\age_predictions_cleaned.csv'
data = pd.read_csv(path_to_data)


X = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_predict = rf.predict(X_test)

acc = accuracy_score(y_test, y_predict)
print(acc)

data_binary = pd.read_csv(r'C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master\murtree-master\resources\datasets\AGE\binaryyyy.csv')

sys.path.append(r"C:\Users\shaka\OneDrive\Email attachments\Documenten\Data Science Project\MurTree-Master\murtree-master\bin\Debug\murtree_python_module.pyd")

#%%
import murtree_python_module as mp

mp.murtree('C:/Users/shaka/OneDrive/Email attachments/Documenten/Data Science Project/binary_dataset.txt', 4, 15, 600)
#%%

import pandas as pd
import numpy as np

# Generate a dataset with 5 columns where the last column is the target
# All variables are binary (0 or 1)
np.random.seed(42)  # For reproducibility
data = np.random.randint(0, 2, size=(100, 5))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Target'])

# Save to a CSV file
file_path = "C:/Users/shaka/OneDrive/Email attachments/Documenten/Data Science Project/binary_dataset.csv"
df.to_csv(file_path, index=False)

file_path
