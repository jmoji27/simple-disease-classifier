#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:16:43 2026

@author: jmoji
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load
train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Clean
for df in [train_df, test_df]:
    if "Unnamed: 133" in df.columns:
        if df["Unnamed: 133"].isnull().all():
            df.drop(columns=["Unnamed: 133"], inplace=True)

# Split
X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]

X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

# Encode
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=300)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc * 100 , "%")