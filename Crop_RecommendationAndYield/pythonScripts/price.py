import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np

# Load dataset
df = pd.read_csv(r"D:\college\MInor project\new\agridata_csv_202110311352.csv")

# Filter data for a specific commodity and location
commodity_name = "Wheat"  # Example commodity name

# Filter data based on the commodity and location
filtered_df = df[(df['commodity_name'] == commodity_name)]

# Convert the date column to datetime format
filtered_df['date'] = pd.to_datetime(filtered_df['date'])

# Sort by date to create a time series
filtered_df = filtered_df.sort_values('date')

# Set date as index
filtered_df.set_index('date', inplace=True)

# Set frequency of the date index (daily in this case)
filtered_df = filtered_df.asfreq('D')

# Select the 'modal_price' column for time series forecasting
price_series = filtered_df['modal_price']

# Split data into train and test sets
train_size = int(len(price_series) * 0.8)
train, test = price_series[:train_size], price_series[train_size:]

# Check if train data is large enough
print(f"Training data size: {len(train)}")

# Ensure no missing values in training data
train = train.fillna(method='ffill')

# ARIMA model - order (p, d, q)
# Try a simpler model to avoid potential overfitting
model = ARIMA(train, order=(1, 1, 1))  # Using ARIMA(1,1,1) instead of (5,1,2)
arima_model = model.fit()

# Save the trained model
model_filename = "arima_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(arima_model, file)
print(f"Model saved to {model_filename}")

# Forecasting with the saved model
# Load the model back
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Forecast using the loaded model
forecast = loaded_model.forecast(steps=len(test))
forecast_index = test.index

# Evaluate the model using Mean Absolute Error
mae = mean_absolute_error(test, forecast)
print(f"Mean Absolute Error (MAE): {mae}")

# Plotting the forecast against actual test values
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(forecast_index, forecast, label='Forecast')
plt.legend(loc='best')
plt.title(f'ARIMA Forecast of {commodity_name} Modal Price')
plt.xlabel('Date')
plt.ylabel('Modal Price')
plt.show()
