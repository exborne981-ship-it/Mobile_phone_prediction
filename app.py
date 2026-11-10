import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="Mobile Price Range Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom UI Styles
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Data Loading and Model Training with Caching & Error Handling
@st.cache_data
def load_data_and_train():
    try:
        df = pd.read_csv("train.csv")
    except FileNotFoundError:
        return None, "Dataset file `train.csv` not found in the application directory. Please upload `train.csv` alongside `app.py`."
    
    # Required features and target
    features = ["ram", "battery_power", "pc"]
    target = "price_range"
    
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns in `train.csv`: {missing_cols}"
        
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=6)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_cm = confusion_matrix(y_test, dt_preds)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)
    
    # Automatic Model Selection
    if rf_acc >= dt_acc:
        best_model = rf_model
        best_name = "Random Forest Classifier"
        best_acc = rf_acc
    else:
        best_model = dt_model
        best_name = "Decision Tree Classifier"
        best_acc = dt_acc
        
    metrics = {
        "dt_acc": dt_acc,
        "dt_cm": dt_cm,
        "rf_acc": rf_acc,
        "rf_cm": rf_cm,
        "best_model": best_model,
        "best_name": best_name,
        "best_acc": best_acc
    }
    
    return metrics, None

# Run data loading and capture errors gracefully
result_data, error_message = load_data_and_train()

if error_message:
    st.error(f"⚠️ **Initialization Error:** {error_message}")
    st.stop()

# 4. Header Section
st.markdown('<div class="main-title">📱 Mobile Price Range Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A machine learning web application to classify mobile phone price tiers based on hardware specifications.</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_compare, tab_about = st.tabs(["🔮 Predict Price", "📊 Model Comparison", "📖 Project Documentation"])

# Label Mappings for Target Categories
price_labels = {
    0: "Low (Budget)",
    1: "Medium (Mid-range)",
    2: "High (Flagship)",
    3: "Very High (Ultra-premium)"
}

# --- TAB 1: PREDICTION DASHBOARD ---
with tab_predict:
    st.subheader("Input Hardware Specifications")
    st.markdown("Provide the device parameters below to instantly classify the expected price category.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ram = st.number_input(
                "RAM (MB)", 
                min_value=256, 
                max_value=8192, 
                value=2048, 
                step=128,
                help="Random Access Memory in Megabytes (e.g., 512 to 8192)"
            )
            
        with col2:
            battery_power = st.number_input(
                "Battery Capacity (mAh)", 
                min_value=500, 
                max_value=7000, 
                value=2000, 
                step=100,
                help="Total energy capacity of the battery in mAh"
            )
            
        with col3:
            pc = st.number_input(
                "Primary Camera (MP)", 
                min_value=0, 
                max_value=108, 
                value=12, 
                step=1,
                help="Primary Camera resolution in Megapixels"
            )
            
        submitted = st.form_submit_button("Predict Price Category", type="primary", use_container_width=True)
        
    if submitted:
        input_df = pd.DataFrame([[ram, battery_power, pc]], columns=["ram", "battery_power", "pc"])
        prediction_idx = result_data["best_model"].predict(input_df)[0]
        predicted_category = price_labels[prediction_idx]
        
        st.markdown("---")
        st.success(f"### Predicted Price Tier: **{predicted_category}**")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Active Deployed Algorithm", value=result_data["best_name"])
        with res_col2:
            st.metric(label="Algorithm Test Accuracy", value=f"{result_data['best_acc'] * 100:.2f}%")

# --- TAB 2: MODEL COMPARISON ---
with tab_compare:
    st.subheader("Supervised Classifier Performance Comparison")
    st.info(f"🏆 **Automatically Selected Model:** `{result_data['best_name']}` based on superior test accuracy.")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Decision Tree Accuracy", value=f"{result_data['dt_acc'] * 100:.2f}%")
    with metric_col2:
        st.metric(label="Random Forest Accuracy", value=f"{result_data['rf_acc'] * 100:.2f}%")
        
    st.markdown("### Confusion Matrices")
    class_names = ["Low", "Medium", "High", "Very High"]
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.markdown("**Decision Tree Confusion Matrix**")
        fig_dt, ax_dt = plt.subplots(figsize=(5, 4))
        sns.heatmap(result_data["dt_cm"], annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax_dt)
        ax_dt.set_xlabel("Predicted Label")
        ax_dt.set_ylabel("True Label")
        st.pyplot(fig_dt)
        
    with fig_col2:
        st.markdown("**Random Forest Confusion Matrix**")
        fig_rf, ax_rf = plt.subplots(figsize=(5, 4))
        sns.heatmap(result_data["rf_cm"], annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names, ax=ax_rf)
        ax_rf.set_xlabel("Predicted Label")
        ax_rf.set_ylabel("True Label")
        st.pyplot(fig_rf)

# --- TAB 3: DOCUMENTATION & METHODOLOGY ---
with tab_about:
    st.markdown("""
    ### Project Overview
    The **Mobile Price Range Prediction** application is a machine learning tool designed to evaluate smartphone hardware specifications and categorize devices into four distinct price brackets ranging from Budget to Ultra-Premium.
    
    ### Methodology & Pipeline
    1. **Data Ingestion & Filtering:** Loads historical observation records from `train.csv` and isolates three primary technical specifications:
       - `ram`: Random Access Memory (MB)
       - `battery_power`: Total energy capacity (mAh)
       - `pc`: Primary camera resolution (MP)
    2. **Data Splitting:** Divides the dataset into 80% training and 20% stratified testing sets to ensure balanced class evaluation.
    3. **Model Training & Automated Selection:** Trains both a **Decision Tree Classifier** and an ensemble **Random Forest Classifier**, evaluates their classification accuracies on unseen test folds, and automatically deploys the higher-performing model for real-time predictions.
    
    ### Technology Stack
    - **Frontend / UI:** Streamlit
    - **Machine Learning:** Scikit-Learn
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn
    """)
