import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib

# Loading trained model
model = keras.models.load_model('trained_lstm_model.h5')

scaler = joblib.load('scaler.joblib')

# Load your DataFrame (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('processed.csv')

#Prepare test data
test_data = df[['sin_Month', 'cos_Month', 'Temp_Normalized']].tail(20).values
X_test = test_data.reshape((20, 1, 3))
y_test = df['Temp_Normalized'].tail(20).values

# Evaluate first to see MSE on test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}\n")

# Next day's temperature Prediction

new_features = df[['sin_Month', 'cos_Month', 'Temp_Normalized']].tail(1).values
new_features = new_features.reshape((1, 1, 3))

next_day_pred = model.predict(new_features)[0][0]
print(f"Model's Prediction for Next Day's Temp (Normalized): {next_day_pred}")

# "Before training" baseline comparison
baseline = df['Temp_Normalized'].tail(1).values[0]
print(f"Before training (baseline) Prediction: {baseline}")


#  Inverse transform to actual temperature

future_features = df[['sin_Month', 'cos_Month', 'Temp_Normalized']].tail(1).copy()
future_features.loc[future_features.index[0], 'Temp_Normalized'] = next_day_pred
inverse_pred = scaler.inverse_transform(future_features)[0][2]

baseline_features = df[['sin_Month', 'cos_Month', 'Temp_Normalized']].tail(1).copy()
baseline_features.loc[baseline_features.index[0], 'Temp_Normalized'] = baseline
inverse_base = scaler.inverse_transform(baseline_features)[0][2]

print(f"Model's Prediction for Next Day's Temp (Actual): {inverse_pred}")
print(f"Before training (baseline) Prediction (Actual): {inverse_base}")

# Summary
print("\nSummary:")
print(f"Model Prediction for Next Day's Temp (Normalized) : {next_day_pred}")
print(f"Before Training (baseline) Prediction (Normalized) : {baseline}")
print(f"Model Prediction for Next Day's Temp (Actual) : {inverse_pred}")
print(f"Before Training (baseline) Prediction (Actual) : {inverse_base}")
print(f"Test Loss (MSE) : {loss}")


# import matplotlib.pyplot as plt

# Predict on the test set first
y_pred = model.predict(X_test)

# Reshape back to 1D
y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Plotting true vs predicted
# plt.figure()
# plt.plot(y_test, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.title('Model Prediction vs Actual')
# plt.xlabel('Index')
# plt.ylabel('Normalized temperature')
# plt.legend()
# plt.show()


import joblib

# Loading scaler
scaler = joblib.load('temp_scaler.joblib')

# Prepare dummy array to match scaler's shape
dummy = np.ones((len(y_pred), 1))
dummy[:] = y_pred.reshape(-1, 1)

# Perform inverse transform
y_pred_real = scaler.inverse_transform(dummy).flatten()

dummy = np.ones((len(y_test), 1))
dummy[:] = y_test.reshape(-1, 1)

y_test_real = scaler.inverse_transform(dummy).flatten()

print("Model Prediction (in degree) :", y_pred_real[0]) 
print("Actual (in degree) :", y_test_real[0])