# skill LCA 3-  All 12 Experiments into One model
## Introduction
This repository contains the code and results for an experiment involving **data visualization and statistical analysis** using Python. The objective is to explore a dataset by performing summary statistics and visualizations such as histograms, box plots, and correlation heatmaps. These techniques are core parts of Exploratory Data Analysis (EDA) in data science.

## Steps-

### 1. Import Libraries:
We import:
- `pandas` for handling datasets,
- `matplotlib.pyplot` and `seaborn` for creating plots,
- `numpy` for numerical operations,
- and `warnings` to ignore unnecessary warnings during execution.

### 2. Load Dataset:
The dataset is loaded using `pd.read_csv()` and stored in a DataFrame for manipulation.

### 3. Data Overview:
We use `.head()` and `.info()` to understand the dataset's structure, types, and non-null counts.

### 4. Data Cleaning:
- Unnecessary columns are dropped.
- Missing values are identified.
- Nulls in important features are filled using mean or mode values.

### 5. Visualization:

#### a) Histograms:
Histograms are plotted to understand the distribution of numerical columns.

#### b) Box Plots:
Box plots help us identify outliers and the spread of key numeric variables.

#### c) Count Plots:
These plots are used to visualize categorical distributions.

#### d) Correlation Heatmap:
This shows the correlation between features, helping identify relationships.

### 6. Data Insights:
- Null values are checked and handled.
- Summary statistics are generated using `.describe()`.

## Concepts Used:

- **Data Loading**: Reading CSV file into a pandas DataFrame.
- **Missing Data Handling**: Using `.isnull().sum()` and `.fillna()` techniques.
- **Data Visualization**:
  - Histogram for distributions
  - Boxplot for spread and outliers
  - Countplot for categorical counts
  - Heatmap for feature correlation
- **Statistical Summary**: Descriptive stats using `.describe()`.
-etc 
## Code:
### Libraries used and Merging and Loading the dataset  

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay, roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  
```
### File paths
```
csv_files = [
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_1.csv',
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_2.csv',
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_3.csv',
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_4.csv'
]
```
### Step 2: Parameters
```
chunk_size = 100000 

data_chunks = []
```
### Step 3: Read each file in chunks and append to the list
```
for file in csv_files:
    print(f"Processing file: {file}")
    try:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            data_chunks.append(chunk)
    except Exception as e:
        print(f"Error reading {file}: {e}")
```
### Step 4: Concatenate all chunks 
```
combined_df = pd.concat(data_chunks, ignore_index=True, sort=True)
```
### Step 5: Save to a new CSV
```
output_file = '/content/drive/MyDrive/combined_dataset.csv'
combined_df.to_csv(output_file, index=False)
print(f"Done! Combined dataset saved as '{output_file}'")
```
### Step 6: Sanity check
```
print("\nFinal dataset info:")
print(f"Total rows: {combined_df.shape[0]}")
print(f"Total columns: {combined_df.shape[1]}")
print(f"Column names: {combined_df.columns.tolist()}")
```
### Calculate percentage of attack and non-attack records
```
attack_percentages = combined_df['attack'].value_counts(normalize=True) * 100
attack_percentages.index = ['Non-Attack' if val == 0 else 'Attack' for val in attack_percentages.index]
print(attack_percentages)
```
### Visualize the distribution of attack vs non-attack records
```
counts = combined_df['attack'].value_counts()
labels = ['Non-Attack' if val == 0 else 'Attack' for val in counts.index]
percentages = (counts / counts.sum()) * 100

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(labels, counts, color=['blue', 'green'])
plt.title("Count of Attack vs Non-Attack")
plt.ylabel("Number of Records")

plt.subplot(1, 2, 2)
plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=['yellow', 'red'])
plt.title("Percentage of Attack vs Non-Attack")

plt.tight_layout()
plt.show()
```
### Feature selection
```
features = ['pkts', 'bytes', 'dur'] 
X = combined_df[features]
y = combined_df['attack']
```
### Step 7: Apply SMOTE to Oversample the Minority Class (Training Set Only)
```
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```
### Split the data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
```
### Scale the features 
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
### Train a Random Forest Model
```
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
```
### Predictions and classification report
```
y_pred = rf.predict(X_test_scaled)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))
```
### XGBoost Model for comparison
```
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
```
### Train Accuracy
```
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
```
### Test Accuracy
```
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("XGBoost Accuracy:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
```
### Confusion Matrix
```
def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    labels = ["Normal", "Attack"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    plt.show()

plot_confusion_matrix(y_test, y_pred, model_name="Random Forest")
```
### ROC Curve
```
def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

plot_roc_curve(model, X_test, y_test, model_name="XGBoost")
```
## Output:
 ![Screenshot 2025-04-28 105643](https://github.com/user-attachments/assets/bdf9086c-2ab3-49ed-a8dd-cb9e76da7e84)
![download (13)](https://github.com/user-attachments/assets/c4f2ba6e-0452-423d-b2a0-31d40b5923ee)
Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99    733609
           1       1.00      0.99      0.99    733609

    accuracy                           0.99   1467218
   macro avg       0.99      0.99      0.99   1467218
weighted avg       0.99      0.99      0.99   1467218

/usr/local/lib/python3.11/dist-packages/xgboost/core.py:158: UserWarning: [04:01:50] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)
XGBoost Accuracy:
Train Accuracy: 0.9934
Test Accuracy:  0.9932
![download (14)](https://github.com/user-attachments/assets/2dea5560-f099-4f5f-b0fa-fc0736ed6701)
![download (15)](https://github.com/user-attachments/assets/42052eba-d8d3-4a75-8fc7-7e504e778495)

