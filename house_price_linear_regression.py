import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
df = pd.read_csv("train.csv")

# Create a combined "Bathrooms" feature
df["Bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]

# Select the features and target
features = ["GrLivArea", "BedroomAbvGr", "Bathrooms"]
target = "SalePrice"

df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat}: {coef:,.2f}")
print(f"  Intercept: {model.intercept_:,.2f}")

print("\nEvaluation on Test Set:")
print(f"  RMSE: {rmse:,.2f}")
print(f"  MAE: {mae:,.2f}")
print(f"  R^2: {r2:.4f}")

# Example: predict price for a specific house
example = pd.DataFrame({
    "GrLivArea": [1800],
    "BedroomAbvGr": [3],
    "Bathrooms": [2.0]
})
example_pred = model.predict(example)[0]
print(f"\nPredicted price for 1800 sq ft, 3 bed, 2 bath: ${example_pred:,.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.savefig("actual_vs_predicted.png")
plt.show()
