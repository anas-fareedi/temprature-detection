# 🌡️ Temperature Forecasting using LSTM

This project predicts the next day's minimum temperature using historical daily temperature data with LSTM (Long Short-Term Memory) neural networks. The dataset is enhanced with time-based cyclic features and normalized for better learning.

---

## 📁 Project Structure

📦temperature-prediction
├── processed.csv # Final preprocessed dataset

├── daily-min-temperatures.csv # Raw temperature data

├── temp_scaler.joblib # Scaler for temperature normalization

├── scaler.joblib # Scaler for input features

├── model_train.py # LSTM training script

├── data_preprocessing.py # Data cleaning and feature engineering script

├── trained_lstm_model.h5 # Saved trained model

└── README.md # You are here

---

## 📊 Dataset

- Source: [`daily-min-temperatures.csv`](https://archive.ics.uci.edu/ml/datasets/daily+minimum+temperatures+in+melbourne)
- Contains minimum daily temperatures in Melbourne, Australia (1981–1990).
- Columns:
  - `Date`
  - `Temp` (daily min temperature in °C)

---

## ⚙️ Preprocessing Steps

1. **Date Conversion:** Converts string dates to `datetime` format.
2. **Feature Engineering:**
   - Extracts month from date.
   - Encodes it cyclically using sine and cosine transforms (`sin_Month`, `cos_Month`).
3. **Normalization:** Temperature values are scaled using `MinMaxScaler`.
4. **Target Definition:** Adds a new `Target` column for predicting the next day's temperature.

All of this is saved to `processed.csv`.

---

## 🧠 Model Architecture

- Input shape: `(1, 3)` for each sample:
  - `sin_Month`, `cos_Month`, and `Temp_Normalized`
- LSTM layer with 16 units.
- Output: 1 value (next day's normalized temperature)
  
🚀 Training

Optimizer: Adam
Loss: Mean Squared Error (MSE)
Epochs: 50
Batch Size: 32
Validation Split: 10%

Final model is saved as trained_lstm_model.h5.

📈 Sample Results
Scaled Test MSE: ~0.0077

Fast training: <1 sec per epoch on GPU
Very good for simple 1-step temperature prediction

🛠️ Requirements

pip install numpy pandas matplotlib scikit-learn tensorflow joblib

📬 Usage
1. Preprocess the Data:
bash
Copy
Edit
python data_preprocessing.py
2. Train the Model:
bash
Copy
Edit
python model_train.py


👤 Author

Anas Fareedi 
Anshul verma

B.Tech (AI/ML) Student
College of Engineering Roorkee (COER)

📜 License
This project is open source and free to use under the MIT License.

yaml

Copy

Edit

---
