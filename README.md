# POCML



# Load your data
data = pd.read_csv(r'C:\\Users\\Ayush Raj\\OneDrive\\Desktop\\performanace.csv')

# Convert 'Interval' to datetime format
try:
    data['Interval'] = pd.to_datetime(data['Interval'], format='%d-%b-%y', errors='coerce')
except Exception as e:
    print("Error parsing dates:", e)
    data['Interval'] = pd.to_datetime(data['Interval'], errors='coerce')

# Drop rows with NaT values after conversion (if any)
data.dropna(subset=['Interval'], inplace=True)

if data.empty:
    raise ValueError("The dataset became empty after preprocessing. Please check the input data.")

# Sort data by 'Interval' to maintain the time series order
data = data.sort_values(by='Interval')

# Select the relevant columns for modeling
# Assuming you want to forecast 'Average' using the historical data of 'Average', 'Min', 'Max', and 'Standard deviation'
features = ['Min', 'Max', 'Average', 'Standard deviation']
target = 'Average'

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Prepare the data for LSTM
sequence_length = 10  # Example sequence length
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, features.index(target)])

X, y = np.array(X), np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Model summary
model.summary()

# Predict on the validation set
y_pred = model.predict(X_val)

# Inverse transform to get the actual values
y_pred = scaler.inverse_transform([[0]*features.index(target) + [pred] + [0]*(len(features)-features.index(target)-1) for pred in y_pred])
y_val_actual = scaler.inverse_transform([[0]*features.index(target) + [val] + [0]*(len(features)-features.index(target)-1) for val in y_val])


mae = mean_absolute_error(y_val_actual[:, features.index(target)], y_pred[:, features.index(target)])
mse = mean_squared_error(y_val_actual[:, features.index(target)], y_pred[:, features.index(target)])
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_val_actual[:, features.index(target)], y_pred[:, features.index(target)])

# Convert MAPE to percentage
mape_percentage = mape * 100

print(f"Mean Absolute Percentage Error (MAPE): {mape_percentage}%")



# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(data['Interval'].iloc[-len(y_val):], y_val_actual[:, features.index(target)], color='blue', label='Actual Average')
plt.plot(data['Interval'].iloc[-len(y_val):], y_pred[:, features.index(target)], color='red', label='Predicted Average')
plt.title('Average Forecasting')
plt.xlabel('Date')
plt.ylabel('Average')
plt.legend()
plt.show()
