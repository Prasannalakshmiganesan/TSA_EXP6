## Reg no: 212222240075
## Developed By: Prasannalakshmi G
## Date: 

# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To create and implement Holt Winter's Method Model using python for goodreadsbooks dataset.

### ALGORITHM:
1. Loading and exploring the CSV data
2. Resampling the data to a monthly frequency
3. Plotting the time series and decomposing into additive components
4. Calculating RMSE for model evaluation
5. Fitting the Holt-Winters model and forecasting future predictions
6. Plotting the original and predicted values
    
### PROGRAM:
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset and parse publication_date as datetime
data = pd.read_csv("Goodreads_books.csv", parse_dates=['publication_date'])

# Set publication_date as the index
data.set_index('publication_date', inplace=True)

# Resample ratings_count to monthly frequency (sum of ratings per month)
monthly_data = data['ratings_count'].resample('MS').sum()

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(monthly_data.values.reshape(-1, 1)).flatten(), 
                        index=monthly_data.index)

# Split into training and testing sets (80% train, 20% test)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Fit the Holt-Winters additive model on training data
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast for the test data length
test_predictions_add = model_add.forecast(steps=len(test_data))

# Evaluate model performance on test data
mae = mean_absolute_error(test_data, test_predictions_add)
rmse = mean_squared_error(test_data, test_predictions_add, squared=False)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Plot 1: Train, Test, and Test Predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='Train', color='black')
plt.plot(test_data, label='Test', color='green')
plt.plot(test_predictions_add, label='Prediction', color='red')
plt.title('Holt-Winters Additive Forecast - Train vs. Test Predictions')
plt.legend(loc='best')
plt.grid('True')
plt.show()

# Fit the final model on the entire dataset (additive trend & seasonality)
final_model = ExponentialSmoothing(monthly_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast next 12 months
forecast = final_model.forecast(steps=12)

# Plot Historical Data with 12-Month Forecast
plt.figure(figsize=(12, 8))
monthly_data.plot(label='Observed', legend=True)
forecast.plot(label='Forecast', legend=True)
plt.title('Holt-Winters Additive Forecast - Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Ratings Count')
plt.grid('True')
plt.show()

# Output final predictions
print("Final Predictions for the next 12 months:")
print(final_prediction)


```

### OUTPUT:


## TEST_PREDICTION:

![{E65D90F3-F097-4382-ACDF-7F01632CE82E}](https://github.com/user-attachments/assets/dc38d7a6-643e-469e-8334-245e50214f42)


## FINAL_PREDICTION:

![{E59D0E15-3C75-4B5C-B0FD-A4216D717228}](https://github.com/user-attachments/assets/001e09b4-2bb0-4c70-bb68-9fce0c150499)

![{DF0C08FB-7241-4509-83C4-FFCF77672215}](https://github.com/user-attachments/assets/3d0a04b4-f310-4367-a323-21c286085d57)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
