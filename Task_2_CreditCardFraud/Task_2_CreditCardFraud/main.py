import pandas as pd

# Load dataset
data = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# Check class distribution
print("\nClass Distribution:")
print(data['Class'].value_counts())
