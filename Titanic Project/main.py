# Core libraries for data handling
import pandas as pd
import numpy as np

# Visualization libraries (for understanding patterns)
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print("Starting Titanic Survival Prediction Pipeline...")
print("Libraries imported successfully!")

data = pd.read_csv("data/Titanic-Dataset.csv")
print("\nDataset Loaded Successfully!\n")
print(data.head())
print("\nDataset Shape:", data.shape)


print("\nDataset Info:\n")
print(data.info())

print("\nStatistical Summary:\n")
print(data.describe())

print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nQuick Insight:")
print("Cabin has too many missing values, may drop it later.")


data['Age'] = data['Age'].fillna(data['Age'].median())

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data.drop(columns=['Cabin'], inplace=True)

print("\nAfter Cleaning Missing Values:\n")
print(data.isnull().sum())

print("\nData cleaned: Filled Age & Embarked, dropped Cabin.")
print("\nUpdated Columns:", data.columns)


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data['Sex'] = encoder.fit_transform(data['Sex'])
data['Embarked'] = encoder.fit_transform(data['Embarked'])

print("\nAfter Encoding:\n")
print(data[['Sex', 'Embarked']].head())

print("\nCategorical columns converted to numerical successfully.")

# Feature Engineering
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

print("\nFamilySize feature created successfully!")


features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked']

X = data[features]   
y = data['Survived'] 

print("\nSelected Features:\n", features)
print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget variable: Survived (0 = No, 1 = Yes)")



X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("\nData Split Completed!")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nModel will train on 80% data and test on 20% unseen data.")



model = RandomForestClassifier(
    n_estimators=120,   
    max_depth=5,        
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

print("Random Forest model trained with controlled depth to avoid overfitting.")


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)

print("\nModel Evaluation Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nModel successfully predicts passenger survival based on selected features.")

import pickle

# Save the trained model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as titanic_model.pkl")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:\n", cm)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

print("\nNew Feature Added: FamilySize")

features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked']