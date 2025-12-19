import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_training import (
    split_and_scale,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest
)
from src.prediction import predict_placement
from src.feature_importance import show_feature_importance

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="PlacementOracle AI", layout="wide")
st.title("🎓 PlacementOracle AI")
st.subheader("Student Placement Prediction & Insight System")

st.markdown("""
This application predicts whether a student will get placed
and explains **which factors influence placement decisions**.
""")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_dataset():
    return load_data("data/placement.csv")

df_raw = load_dataset()

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview", "Model Training", "Placement Prediction"]
)

# =========================================================
# 1️⃣ DATASET OVERVIEW
# =========================================================
if menu == "Dataset Overview":
    st.header("📊 Dataset Overview")

    st.write("### Preview of Dataset")
    st.dataframe(df_raw.head())

    st.write("### Dataset Shape")
    st.write(df_raw.shape)

    st.write("### Missing Values")
    st.write(df_raw.isnull().sum())

    st.write("### Placement Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df_raw, ax=ax)
    st.pyplot(fig)

# =========================================================
# 2️⃣ MODEL TRAINING
# =========================================================
elif menu == "Model Training":
    st.header("⚙️ Model Training & Evaluation")

    df = preprocess_data(df_raw)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    if st.button("Train Model"):
        if model_choice == "Logistic Regression":
            model = train_logistic_regression(X_train, y_train)
        elif model_choice == "Decision Tree":
            model = train_decision_tree(X_train, y_train)
        else:
            model = train_random_forest(X_train, y_train)

        st.success(f"{model_choice} trained successfully!")

        from sklearn.metrics import accuracy_score

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write("### Model Accuracy")
        st.write(f"{acc * 100:.2f}%")

        if model_choice == "Random Forest":
            st.write("### Feature Importance")
            feature_names = df.drop("status", axis=1).columns
            show_feature_importance(model, feature_names)
            st.pyplot(plt.gcf())

# =========================================================
# 3️⃣ PLACEMENT PREDICTION
# =========================================================
else:
    st.header("🎯 Placement Prediction")

    df = preprocess_data(df_raw)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    model = train_random_forest(X_train, y_train)
    feature_names = df.drop("status", axis=1).columns

    st.write("### Enter Student Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc_p = st.slider("10th Percentage", 40, 100, 75)
    hsc_p = st.slider("12th Percentage", 40, 100, 70)
    degree_p = st.slider("Degree Percentage", 40, 100, 72)
    mba_p = st.slider("MBA Percentage", 40, 100, 70)
    workex = st.selectbox("Work Experience", ["Yes", "No"])
    etest_p = st.slider("Employability Test Score", 40, 100, 65)
    hsc_s = st.selectbox("12th Stream", ["Science", "Commerce", "Arts"])
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])

    input_data = {
        "gender": 1 if gender == "Male" else 0,
        "ssc_p": ssc_p,
        "ssc_b": 1,
        "hsc_p": hsc_p,
        "hsc_b": 1,
        "hsc_s": hsc_s,
        "degree_p": degree_p,
        "degree_t": degree_t,
        "workex": 1 if workex == "Yes" else 0,
        "etest_p": etest_p,
        "specialisation": 1,
        "mba_p": mba_p
    }

    if st.button("Predict Placement"):
        pred, prob = predict_placement(
            model,
            scaler,
            input_data,
            feature_names
        )

        if pred == 1:
            st.success(f"🎉 Student is likely to be PLACED ({prob*100:.2f}%)")
        else:
            st.error(f"❌ Student is NOT likely to be placed ({prob*100:.2f}%)")
