# Simple Disease Classifier Using Multi-Layer Perceptron

This repository contains a simple machine learning pipeline built as a homework assignment to classify diseases based on symptom patterns using a Multi-Layer Perceptron (MLP) neural network.

## Project Overview
The assignment focuses on processing structured training and testing datasets containing clinical symptoms, cleaning extraneous database column artifacts, encoding categorical diagnostic string targets, scaling binary symptom metrics, and evaluating an MLP neural network classifier.

## Technical Features
* **Data Cleaning:** Automatically detects and drops the trailing empty delimiter column (`Unnamed: 133`) that commonly appears as a formatting artifact in standard benchmark disease datasets.
* **Feature Preprocessing:** Utilizes `LabelEncoder` to transform text-based disease targets into categorical numerical IDs and applies `StandardScaler` to normalize features for stable neural network optimization.
* **Architecture Design:** Implements a Multi-Layer Perceptron using `scikit-learn` with a 3-layer deep architecture configuration: `(64, 32, 16)`.

---

## Dataset Structure

The assignment utilizes standard structured benchmark data partitioned into two core files within the project directory:
* **`Training.csv`**: Used to fit the neural network weights and bias parameters.
* **`Testing.csv`**: Comprises 42 test cases containing 132 individual binary clinical symptom tracking columns (such as `itching`, `skin_rash`, `continuous_sneezing`) and 1 target `prognosis` label column to calculate final accuracy.

---

## Code Workflow

### 1. Data Loading & Artifact Filtering
The script reads the CSV input streams and drops empty artifact boundaries to avoid shape mismatches during tensor computation:
```python
for df in [train_df, test_df]:
    if "Unnamed: 133" in df.columns:
        if df["Unnamed: 133"].isnull().all():
            df.drop(columns=["Unnamed: 133"], inplace=True)
