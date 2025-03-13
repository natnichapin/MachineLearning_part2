import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv("./data/usd2brl.csv")

# Convert 'datetime' column to datetime objects and transform into numerical values (Unix timestamps)
df['datetime'] = pd.to_datetime(df['datetime'])
df['timestamp'] = df['datetime'].astype('int64') // 10 ** 9  # Convert to seconds since epoch

# Extract features (timestamp) and target (high)
X = df[['timestamp']]
Y = df['usd_brl']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=10, shuffle=False)

# Standardize the 'timestamp' feature using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the test data using the fitted scaler

# Initialize Ridge Regression model
ridge_model = Ridge(alpha=10, solver="sag", max_iter=1000)
ridge_model.fit(X_train_scaled, Y_train)  # Fit the Ridge model on training data

# Print model coefficients
print(f'beta_0 (Intercept): {ridge_model.intercept_}')
print(f'beta_i (Coefficients): {ridge_model.coef_}')

# Predict on training data
Y_train_predicted = ridge_model.predict(X_train_scaled)
print('Predicted values (training):', Y_train_predicted)

# Predict on 10 test points
Y_test_predicted = ridge_model.predict(X_test_scaled)
print('Predicted values (test, last 10 points):', Y_test_predicted)

# Optional: Combine test predictions with actual values for comparison
test_results = pd.DataFrame({
    'Actual': Y_test.values,
    'Predicted': Y_test_predicted
}, index=Y_test.index)
print("\nTest Results (Actual vs Predicted):")
print(test_results)

# Predict on future dates (e.g., 2 example dates)
future_dates = pd.DataFrame({'datetime': pd.to_datetime(['2025-01-01', '2025-06-01'])})
future_dates['timestamp'] = future_dates['datetime'].astype('int64') // 10 ** 9

# Scale future timestamps using the previously fitted scaler
future_dates_scaled = scaler.transform(future_dates[['timestamp']])

# Predict future values
Y_future_pred = ridge_model.predict(future_dates_scaled)
print('\nFuture Predictions:', Y_future_pred)

# Evaluate Model Performance
# 1. Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, Y_test_predicted)

# 2. Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_test_predicted)

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# 4. Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((Y_test - Y_test_predicted) / Y_test)) * 100

# Print the performance metrics
print("\nPerformance Metrics:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_test_predicted, color='blue', alpha=0.7, label='Predicted Values')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--',
         label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()