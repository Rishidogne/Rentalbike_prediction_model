# bike_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('data/hour.csv')  # Make sure 'hour.csv' is inside the 'data/' folder

# Step 2: Drop unnecessary columns
df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)

# Step 3: Convert categorical features to 'category' type
categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
for col in categorical_features:
    df[col] = df[col].astype('category')

# Step 4: Prepare features and target
X = df.drop('cnt', axis=1)  # Features
y = df['cnt']               # Target: Total count of bike rentals

# Step 5: One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Step 6: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
os.makedirs('models', exist_ok=True)
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
print("✅ Model saved to models/feature_columns.pkl")

# Step 7: Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
joblib.dump(model, 'models/bike_model.pkl')
print("✅ Model saved to models/bike_model.pkl")
# Step 10: Plot predictions vs actual values
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.title('Actual vs Predicted Bike Rentals (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Rental Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
