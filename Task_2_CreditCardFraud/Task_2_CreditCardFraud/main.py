import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("creditcard.csv")

print("Dataset Loaded!")

# Features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Model trained!")

# Predictions
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
