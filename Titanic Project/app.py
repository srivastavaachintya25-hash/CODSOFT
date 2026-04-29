import streamlit as st
import pickle
import numpy as np

# Load model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
family = st.slider("Family Size", 1, 10, 1)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert inputs
sex = 1 if sex == "Male" else 0
embarked_map = {"S": 2, "C": 0, "Q": 1}
embarked = embarked_map[embarked]

input_data = np.array([[pclass, sex, age, fare, family, embarked]])


        
if st.button("Predict", key="predict_btn"):
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    # 🔥 KEEP THIS INSIDE
    proba = model.predict_proba(input_data)[0][1]
    st.write(f"Survival Probability: {round(proba * 100, 2)}%")

    st.write("Prediction based on ML model trained on Titanic dataset.")
    st.write("This app predicts survival chances based on passenger details using a trained ML model.")