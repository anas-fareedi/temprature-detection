import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Data/temp.csv")
print(df.head())


df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)

df = df.drop(['Year', 'Month', 'Day', 'Date'], axis=1)

temps = df['Temp'].values.reshape(-1, 1)  

def scale_data(temps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(temps)
    return scaler, scaled


scaler, scaled_vals = scale_data(temps)

df['Temp_Normalized'] = scaled_vals

print(df)

df['Target'] = df['Temp_Normalized'].shift(-1)  # Next day's temperature as Target
df = df.dropna()

df.to_csv('processed.csv', index=False)

# if df.isnull().values.any():
#     print("Null Value are present")
# else:
#     print("no null")

import joblib

scaler, scaled_vals = scale_data(temps)

joblib.dump(scaler, 'temp_scaler.joblib')
print("Scaler successfully saved.")
