import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
file_path = "city_day.csv"  #https://www.kaggle.com/code/anjusunilkumar/air-quality-index-prediction?select=city_day.csv
df = pd.read_csv(file_path)

#Display Dataset
df

# Display basic info about the dataset
df.info()

# Che+ck the total null values
df.isnull().sum()

# Handling missing values
# Drop rows where AQI is missing, as it's our target variable
df = df.dropna(subset=['AQI'])

# Fill missing values only in numeric columns with their respective column means
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Drop non-numeric and irrelevant columns
drop_columns = ['City', 'Date', 'AQI_Bucket']  # 'City' and 'Date' are categorical, 'AQI_Bucket' is redundant
df = df.drop(columns=drop_columns)

# Define target (dependent variable) and features (independent variables)
target = 'AQI'
features = [col for col in df.columns if col != target]  # All columns except AQI

X = df[features]  # Features dataset
y = df[target]  # Target variable

# Select only numerical columns for correlation
num_df = df.select_dtypes(include=['number'])  # Keep only numeric columns

# Compute correlation matrix
corr_matrix = num_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
# Standardize the dataset to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and define models to be tested
models = {
    "Linear Regression": LinearRegression(),  # Simple baseline model
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),  # Handles non-linearity well
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)  # Gradient boosting for performance
}

# Train and Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Train the model on scaled data
    y_pred = model.predict(X_test_scaled)  # Predict on test set
    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2 Score': r2}  # Store results
    print(f"{name} Performance:\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.2f}\n")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T
print(results_df)

import matplotlib.pyplot as plt

# Plot model performance
results_df.plot(kind='bar', figsize=(10, 5))
plt.title("Model Comparison")
plt.ylabel("Performance Metrics")
plt.xticks(rotation=0)
plt.show()

# Feature Importance Plot (For XGBoost)
feature_importance = models["XGBoost"].feature_importances_
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance - XGBoost")
plt.show()

# Hyperparameter Tuning for XGBoost
# Define parameter grid for tuning
tuning_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror'), tuning_params, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
print(f"Best XGBoost Parameters: {grid_search.best_params_}")  # Output the best parameters

# Save the best model
import joblib
joblib.dump(models["XGBoost"], "best_aqi_model.pkl")

from google.colab import files
files.download("best_aqi_model.pkl")

# Train the best XGBoost model with optimized parameters
best_xgb = XGBRegressor(objective='reg:squarederror',
                        learning_rate=0.2,
                        max_depth=5,
                        n_estimators=100,
                        random_state=42)

best_xgb.fit(X_train_scaled, y_train)
y_pred_xgb = best_xgb.predict(X_test_scaled)

# Evaluate the optimized model
mae = mean_absolute_error(y_test, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2 = r2_score(y_test, y_pred_xgb)

print(f"Optimized XGBoost Performance:\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

# Save the optimized model
import joblib
joblib.dump(best_xgb, "optimized_xgb_model.pkl")

new_data = X_test_scaled[:5]  # Use the first 5 test samples as an example
predictions = best_xgb.predict(new_data)
print("Predicted AQI values:", predictions)

pred_df = pd.DataFrame({'Predicted_AQI': predictions})
pred_df.to_csv("aqi_predictions.csv", index=False)
print("Predictions saved to aqi_predictions.csv")

plt.figure(figsize=(8, 5))
plt.plot(y_test[:5].values, label="Actual AQI", marker='o')
plt.plot(predictions, label="Predicted AQI", marker='s')
plt.xlabel("Sample Index")
plt.ylabel("AQI Value")
plt.title("Actual vs. Predicted AQI")
plt.legend()
plt.show()
