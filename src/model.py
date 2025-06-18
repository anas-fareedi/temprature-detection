import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Loading processed data
df = pd.read_csv('processed.csv')

X = df[['sin_Month', 'cos_Month', 'Temp_Normalized']]
Y = df['Target']


scaler = StandardScaler()
scaler.fit(X)

# Save scaler for future use
joblib.dump(scaler, 'scaler.joblib')

X = scaler.transform(X)
Y = Y.values

# Reshape to match LSTM's expected 3D input
X = X.reshape((X.shape[0], 1, 3))

X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                     test_size=0.2, 
                                                     random_state=42)

# Define a simple feed-forward network
model = keras.Sequential([
    keras.layers.LSTM(16, input_shape=(1, 3)),
    keras.layers.Dense(1)  # regression output
])

# model = keras.Sequential()
# model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.LSTM(32))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(1))

# Compile with an optimizer (like Adam) and a loss (like MSE)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Save the trained model
model.save('trained_lstm_model.h5')
print("Model has been trained and saved.")


