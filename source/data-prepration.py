import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("temprature-detection/data/daily-min-temperatures.csv")

print(df.head())

# print(df.isnull().sum()) 

df['Date'] = pd.to_datetime(df['Date']) 
 
temps = df['Temp'].values  

scaler = MinMaxScaler()

# temps_scaled = scaler.fit_transform(temps.reshape(-1, 1))  
#           OR

split_index_raw = int(len(temps) * 0.8)
train_raw = temps[:split_index_raw].reshape(-1, 1)
test_raw = temps[split_index_raw:].reshape(-1, 1)

train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

temps_scaled = np.concatenate([train_scaled, test_scaled])



df['Temp_Normalized'] = temps_scaled 

#  FEATURE AND LABELS (X,Y)

X=[]
y=[]

back = 3

for i in range(back, len(temps_scaled)):
    X.append(temps_scaled[i - back:i])  # input
    y.append(temps_scaled[i]) 

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1)) 

split_index = int(len(X) * 0.8)

train_X, test_X = X[:split_index], X[split_index:]
train_y, test_y = y[:split_index], y[split_index:]

np.savez("temprature-detection/data/dataset.npz", train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)

joblib.dump(scaler, "temprature-detection/data/temp_scaler.pkl")     