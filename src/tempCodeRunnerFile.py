model = keras.Sequential([
    keras.layers.LSTM(16, input_shape=(1, 3)),
    keras.layers.Dense(1)  # regression output
])
