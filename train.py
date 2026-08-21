import pandas as pd

# Load dataset
data = pd.read_csv("housing.csv")

print("=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Check missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Fill missing values in total_bedrooms with median
data["total_bedrooms"] = data["total_bedrooms"].fillna(
    data["total_bedrooms"].median()
)

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", data.duplicated().sum())

print("\nDataset Shape:", data.shape)

print("\nData Cleaning Completed Successfully!")
# Convert categorical column into numerical columns
data = pd.get_dummies(
    data,
    columns=["ocean_proximity"],
    drop_first=True
)

print("\nAfter Encoding:")
print(data.head())

print("\nNew Dataset Shape:", data.shape)
from sklearn.model_selection import train_test_split

# Separate features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n" + "=" * 50)
print("TRAIN TEST SPLIT")
# ==================================================
# MODEL TRAINING
# ==================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# ==================================================
# MODEL PREDICTION
# ==================================================

y_pred = model.predict(X_test)

print("\nPredictions Completed Successfully!")

# ==================================================
# MODEL EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")

# ==================================================
# SAVE MODEL
# ==================================================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as model.pkl")
print("=" * 50)

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Target  :", y_train.shape)
print("Testing Target   :", y_test.shape)