# ===============================
# Customer Churn Prediction App
# ===============================

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# PAGE CONFIG (FIRST LINE)
# ===============================
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

# ===============================
# LOAD PIPELINE
# ===============================
PIPELINE_PATH = r"C:\Users\Agnel Sharon Jerald\Machine learning\churn_complete_pipeline.pkl"

with open(PIPELINE_PATH, "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_order = artifacts["feature_order"]

# ===============================
# LOAD DATA (FOR METRICS ONLY)
# ===============================
DATA_PATH = r"C:\Users\Agnel Sharon Jerald\Machine learning\Churn.csv"
df = pd.read_csv(DATA_PATH)

# ===============================
# FIX COLUMN NAMES
# ===============================
df.rename(columns={
    "account.length": "account_length",
    "voice.plan": "voice_plan",
    "voice.messages": "voice_messages",
    "intl.plan": "intl_plan",
    "intl.calls": "intl_calls",
    "intl.charge": "intl_charge",
    "day.calls": "day_calls",
    "eve.calls": "eve_calls",
    "night.calls": "night_calls",
    "customer.calls": "customer_calls",
    "churn": "Churn"
}, inplace=True)

# ===============================
# CREATE Total_Charge COLUMN (if missing)
# ===============================
if "Total_Charge" not in df.columns:
    df["Total_Charge"] = 0  # If original day/eve/night charges are gone

# ===============================
# ENCODE YES / NO COLUMNS
# ===============================
df["voice_plan"] = df["voice_plan"].map({"yes": 1, "no": 0})
df["intl_plan"] = df["intl_plan"].map({"yes": 1, "no": 0})
df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})

# ===============================
# DARK THEME
# ===============================
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
h1, h2, h3 { color: #00E5FF; }
.stButton>button {
    background-color: #00E5FF;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs([
    "📌 Overview / EDA",
    "📈 Model Evaluation",
    "🔍 Churn Prediction"
])

# ===============================
# 📌 OVERVIEW / EDA
# ===============================
with tab1:
    st.title("Customer Churn Prediction System")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
    col3.metric("Retention Rate", f"{(1-df['Churn'].mean())*100:.2f}%")

    st.markdown("""
    ### Business Impact
    - Predict customers likely to churn
    - Improve retention strategies
    - Increase revenue
    """)

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Voice Plan vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="voice_plan", hue="Churn", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("International Plan vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="intl_plan", hue="Churn", data=df, ax=ax)
    st.pyplot(fig)

# ===============================
# 📈 MODEL EVALUATION
# ===============================
with tab2:
    st.title("Model Evaluation")

    # Select only feature_order columns
    X = df.reindex(columns=feature_order, fill_value=0)
    y = df["Churn"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y, y_pred)
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

# ===============================
# 🔍 CHURN PREDICTION
# ===============================
with tab3:
    st.title("Predict Customer Churn")
    st.sidebar.header("Customer Details")

    user_input = {}
    for feature in feature_order:
        if feature in ["voice_plan", "intl_plan"]:
            user_input[feature] = st.sidebar.selectbox(
                feature.replace("_", " ").title(),
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        else:
            user_input[feature] = st.sidebar.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                value=0.0
            )

    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_order]
    input_scaled = scaler.transform(input_df)

    if st.button("Predict Churn"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ High Churn Risk — {prob:.2%}")
        else:
            st.success(f"✅ Low Churn Risk — {prob:.2%}")
