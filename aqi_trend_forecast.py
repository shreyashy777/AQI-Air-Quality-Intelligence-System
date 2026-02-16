import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load Dataset
# ---------------------------

df = pd.read_csv("aqi_data.csv")

city_name = "Delhi"
print("City:", city_name)


# Keep only necessary columns
df = df[["Datetime", "AQI"]]

# Convert date
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

# Remove missing AQI
df = df.dropna(subset=["AQI"])

print("===== AIR QUALITY INTELLIGENCE SYSTEM =====")
print("City:", city_name)
print("Total Records Used:", len(df))


# ---------------------------
# 2. Create High-Risk Label
# ---------------------------

# Define high-risk as AQI > 200
df["High_Risk"] = (df["AQI"] > 200).astype(int)

# ---------------------------
# 3. Create Lag Features
# ---------------------------

df["AQI_lag1"] = df["AQI"].shift(1)
df["AQI_lag2"] = df["AQI"].shift(2)
df["AQI_lag3"] = df["AQI"].shift(3)

df = df.dropna()

X = df[["AQI_lag1", "AQI_lag2", "AQI_lag3"]]
y = df["High_Risk"]

# ---------------------------
# 4. Train Model
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== HIGH RISK FORECAST MODEL =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------
# 5. Trend Visualization
# ---------------------------

plt.figure(figsize=(10,5))
plt.plot(df["Datetime"], df["AQI"])
plt.title("Delhi AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

# ---------------------------
# 6. Demo Forecast
# ---------------------------

print("\n===== NEXT DAY RISK FORECAST =====")

sample = [[250, 230, 210]]  # Last 3 days AQI
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Forecast: HIGH RISK Tomorrow ⚠")
    print("Advice: Avoid outdoor exposure.")
else:
    print("Forecast: Moderate/Safe Tomorrow")
    print("Advice: Normal outdoor activity acceptable.")
