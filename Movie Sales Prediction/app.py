import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(page_title="Sales Prediction App", page_icon="📈")

# Title
st.title("📈 Sales Prediction App")
st.write("Predict sales based on TV, Radio, and Newspaper advertising budgets.")

# Load Data
data = pd.read_csv("advertising.csv")

# Features & Target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write(f"Model Accuracy (R² Score): {accuracy:.2f}")

# User Inputs
tv = st.number_input("Enter TV Advertising Budget", min_value=0.0, value=150.0)
radio = st.number_input("Enter Radio Advertising Budget", min_value=0.0, value=25.0)
newspaper = st.number_input("Enter Newspaper Advertising Budget", min_value=0.0, value=30.0)

# Predict Button
if st.button("Predict Sales"):
    new_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=['TV', 'Radio', 'Newspaper']
    )

    prediction = model.predict(new_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")

# Dataset Preview
if st.checkbox("Show Dataset"):
    st.subheader("Advertising Dataset")
    st.write(data.head())

# Visualization
st.subheader("TV Advertising vs Sales")
fig, ax = plt.subplots()
ax.scatter(data["TV"], data["Sales"])
ax.set_xlabel("TV Advertising Budget")
ax.set_ylabel("Sales")
ax.set_title("TV Ads vs Sales")
st.pyplot(fig)