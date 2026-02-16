# -----------------------------
# 1. Create Balanced Risk Labels (Quantile-Based)
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("aqi_data.csv")

# Keep only Delhi (change city if needed)
city_name = "Delhi"
df = df[df["City"] == city_name]

# Drop unused columns
df = df.drop(columns=["City", "Datetime", "AQI_Bucket"])

# Remove rows with missing AQI
df = df.dropna(subset=["AQI"])

# Fill missing pollutant values
pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
df[pollutant_cols] = df[pollutant_cols].fillna(df[pollutant_cols].mean())

# -----------------------------
# Create Balanced Risk Classes
# -----------------------------

q1 = df["AQI"].quantile(0.33)
q2 = df["AQI"].quantile(0.66)

def quantile_risk(aqi):
    if aqi <= q1:
        return "SAFE"
    elif aqi <= q2:
        return "WARNING"
    else:
        return "DANGEROUS"

df["Risk_Level"] = df["AQI"].apply(quantile_risk)

print("Quantile thresholds:")
print("SAFE <=", round(q1,2))
print("WARNING <=", round(q2,2))
print()

# -----------------------------
# Prepare Model Data
# -----------------------------

X = df[pollutant_cols]
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("===== BALANCED AQI RISK MODEL =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Demo Prediction System
# -----------------------------

print("\n===== AQI SMART ADVISORY SYSTEM =====")

sample = [[250, 300, 80, 20, 1.2, 60]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("Input Pollutant Values:", sample[0])
print("Predicted Risk Level:", prediction[0])

if prediction[0] == "SAFE":
    print("Health Advisory: Air quality is acceptable.")
elif prediction[0] == "WARNING":
    print("Health Advisory: Sensitive groups should limit outdoor activity.")
else:
    print("Health Advisory: Avoid outdoor exposure. High health risk.")
