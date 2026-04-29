# SALES PREDICTION USING SKLEARN

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Load Dataset
data = pd.read_csv("advertising.csv")


# Step 3: Display Dataset
print("First 5 Rows of Dataset:")
print(data.head())


# Step 4: Features (Input) and Target (Output)
X = data[['TV', 'Radio', 'Newspaper']]   # Independent Variables
y = data['Sales']                        # Dependent Variable


# Step 5: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 7: Make Predictions
y_pred = model.predict(X_test)


# Step 8: Model Evaluation
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Step 9: Coefficients and Intercept
print("\nModel Coefficients:")
print("TV Coefficient:", model.coef_[0])
print("Radio Coefficient:", model.coef_[1])
print("Newspaper Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)


# Step 10: Predict New Sales
new_data = pd.DataFrame(
    [[150, 25, 30]],
    columns=['TV', 'Radio', 'Newspaper']
)

predicted_sales = model.predict(new_data)

print("\nPredicted Sales for:")
print("TV = 150, Radio = 25, Newspaper = 30")
print("Predicted Sales:", predicted_sales[0])


# Step 11: Visualization
plt.scatter(data["TV"], data["Sales"])
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("TV Ads vs Sales")
plt.show()