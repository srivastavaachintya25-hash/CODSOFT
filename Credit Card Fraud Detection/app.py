import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

# Load model
model = joblib.load("models/model.pkl")

# Upload section
st.subheader("📂 Upload Transaction CSV")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Remove target column if present
    if "Class" in df.columns:
        df_input = df.drop("Class", axis=1)
    else:
        df_input = df

    # Detect button
    if st.button("🔍 Detect Fraud"):

        # Predictions
        predictions = model.predict(df_input)
        df["Prediction"] = predictions

        # Metrics
        total = len(df)
        fraud_count = (predictions == 1).sum()
        normal_count = (predictions == 0).sum()

        st.subheader("📈 Results")
        st.dataframe(df, use_container_width=True)

        # Metrics display
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Fraud Transactions", fraud_count)
        col3.metric("Normal Transactions", normal_count)

        # Alert
        if fraud_count > 0:
            st.error("🚨 Fraudulent transactions detected!")
        else:
            st.success("✅ No fraud detected")

        # 📊 Chart (FIXED POSITION)
        st.subheader("📊 Fraud Distribution")

        labels = ["Normal", "Fraud"]
        values = [normal_count, fraud_count]

        fig, ax = plt.subplots(figsize=(3, 2.5))  
        ax.bar(labels, values)

        ax.set_title("Fraud vs Normal", fontsize=10)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout()

        # Center chart
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig)