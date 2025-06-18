# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from math import sqrt
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("running on GPU")
# else:
#     print("running on CPU")

# data = np.load("../data/dataset.npz")
# train_X, train_y = data['train_X'], data['train_y']
# test_X, test_y = data['test_X'], data['test_y']
# scaler = joblib.load("../data/temp_scaler.pkl")

# print("Train_X shape:", train_X.shape)
# print("First 5 scaled temperatures (train_y):", train_y[:5])

# n_input = train_X.shape[1]
# n_features = train_X.shape[2]

# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# model.fit(train_X, train_y, epochs=50)

# scaled_test = test_y.reshape(-1, 1)
# scaled_train = train_y.reshape(-1, 1)

# test_predictions = []
# current_batch = scaled_train[-n_input:].reshape((1, n_input, n_features))

# for i in range(len(scaled_test)):
#     current_pred = model.predict(current_batch, verbose=0)[0]
#     test_predictions.append(current_pred)
#     current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# true_predictions = scaler.inverse_transform(test_predictions)
# true_actual = scaler.inverse_transform(scaled_test)

# test_df = pd.DataFrame({
#     "Actual": true_actual.flatten(),
#     "Predicted": true_predictions.flatten()
# })

# rmse = sqrt(mean_squared_error(test_df["Actual"], test_df["Predicted"]))
# mae = mean_absolute_error(test_df["Actual"], test_df["Predicted"])
# r2 = r2_score(test_df["Actual"], test_df["Predicted"])
# print(f"RMSE: {rmse:.3f}")
# print(f"MAE: {mae:.3f}")
# print(f"R² Score: {r2:.3f}")

# # loss_per_epoch = model.history.history['loss']
# # plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
# # plt.title("Loss per Epoch")
# # plt.show()

# # plt.figure(figsize=(14, 5))
# # plt.plot(test_df["Actual"], label="Actual")
# # plt.plot(test_df["Predicted"], label="Predicted")
# # plt.title("Actual vs Predicted Temperature")
# # plt.legend()
# # plt.show()

# model.save("lstm_temp_model.keras")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# --- GPU Check ---
gpus = tf.config.list_physical_devices('GPU')
print("✅ Running on GPU" if gpus else "❌ Running on CPU")

# --- Load Data ---
data = np.load("../data/dataset.npz")
train_X = data['train_X'].astype(np.float32)
train_y = data['train_y'].astype(np.float32)
test_X  = data['test_X'].astype(np.float32)
test_y  = data['test_y'].astype(np.float32)
scaler = joblib.load("../data/temp_scaler.pkl")

print("Train_X shape:", train_X.shape)
print("First 5 train_y values:", train_y[:5])

n_input    = train_X.shape[1]
n_features = train_X.shape[2]

# --- Build Optimized LSTM Model ---
model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(n_input, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=opt, loss='mse')
model.summary()

# --- ModelCheckpoint Only ---
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)

# --- Train Model (No EarlyStopping) ---
history = model.fit(train_X, train_y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[checkpoint],
                    verbose=1)

# --- Loss Plot ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Prediction Loop ---
predictions_scaled = []
current_batch = train_y[-n_input:].reshape((1, n_input, n_features))

for _ in range(len(test_y)):
    pred = model.predict(current_batch, verbose=0)[0]
    predictions_scaled.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
predictions_scaled = np.nan_to_num(predictions_scaled, nan=0.0)

# --- Inverse Transform ---
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(test_y.reshape(-1, 1))

# --- Evaluation ---
rmse = sqrt(mean_squared_error(actuals, predictions))
mae  = mean_absolute_error(actuals, predictions)
r2   = r2_score(actuals, predictions)

print("\n📊 Final Evaluation on Test Data:")
print(f"RMSE     : {rmse:.3f}")
print(f"MAE      : {mae:.3f}")
print(f"R² Score : {r2:.3f}")

# --- Save Final Model ---
model.save("optimized_lstm_temp_model.keras")
