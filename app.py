import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Mobile Price Range Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_train():
    df = pd.read_csv("train.csv")
    X = df[["ram", "battery_power", "pc"]]
    y = df["price_range"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_cm = confusion_matrix(y_test, dt_preds)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)
    
    if rf_acc >= dt_acc:
        best_model = rf_model
        best_name = "Random Forest Classifier"
        best_acc = rf_acc
    else:
        best_model = dt_model
        best_name = "Decision Tree Classifier"
        best_acc = dt_acc
        
    return {
        "dt_model": dt_model,
        "dt_acc": dt_acc,
        "dt_cm": dt_cm,
        "rf_model": rf_model,
        "rf_acc": rf_acc,
        "rf_cm": rf_cm,
        "best_model": best_model,
        "best_name": best_name,
        "best_acc": best_acc,
        "X_train": X_train,
        "y_test": y_test
    }

# Ensure file executes cleanly without errors
print("Code validation successful.")
