import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("datasets.csv")
print(df.head());

print("Dataset Info:")
df.info()
print("\nFirst 5 rows:")
print(df.head())


# Data Preprocessing
from sklearn.preprocessing import StandardScaler

cols_to_scale = ['TV', 'Radio', 'Newspaper'] # tulis di sini

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

from sklearn.model_selection import train_test_split

# tulis labelnya
X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Preprocessing Results:")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Features after scaling:")
print(X_train.head())

# Training using Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Model Inference
tv = int(input("Masukkan budget TV: "))
radio = int(input("Masukkan budget Radio: "))
newspaper = int(input("Masukkan budget Newspaper: "))

# Buat data baru dalam bentuk dictionary (lebih mudah dibaca)
data_baru = {
    'TV' : tv,
    'Radio' : radio,
    'Newspaper' : newspaper
}

new_df = pd.DataFrame([data_baru])
print(f"Data baru yang akan diprediksi:\n{new_df}\n")

new_df[cols_to_scale] = scaler.transform(new_df[cols_to_scale])

new_df = new_df[X.columns]

print(f"Data baru setelah preprocessing:\n{new_df}\n")
prediksi_charges = rf.predict(new_df)

# Tampilkan hasil
print("-" * 30)
print(f"Hasil Prediksi Sales Revenue: ${prediksi_charges[0]:.2f}")
print("-" * 30)

# Export Model
import joblib

# Export the trained model
filename = 'rf_model.joblib'
joblib.dump(rf, filename)

print(f"Model exported successfully as {filename}")

import joblib

# Export the scaler object
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Scaler exported successfully as {scaler_filename}")

# Note: No encoders needed for this regression model
# All features are numerical, no categorical encoding required
print("Model and scaler exported successfully!")