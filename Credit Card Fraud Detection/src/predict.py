import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Take sample WITH column names
sample = df.drop("Class", axis=1).iloc[[0]]  # double brackets

# Predict
prediction = model.predict(sample)

# Output
if prediction[0] == 1:
    print("Fraud Transaction Detected!")
else:
    print("Genuine Transaction")