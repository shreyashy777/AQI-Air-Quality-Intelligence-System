import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("aqi_data.csv")

# Use Delhi only
df = df[df["City"] == "Delhi"]

# Drop unnecessary columns
df = df.drop(columns=["City", "Datetime"])

# Drop rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Fill missing pollutant values
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
df[pollutants] = df[pollutants].fillna(df[pollutants].mean())

# Create CLEAN 3-level Risk Category based on AQI value
def create_risk(aqi):
    if aqi <= 100:
        return "SAFE"
    elif aqi <= 200:
        return "WARNING"
    else:
        return "DANGEROUS"

df["Risk_Level"] = df["AQI"].apply(create_risk)

# Features & Target
X = df[pollutants]
y = df["Risk_Level"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# Stratified split (keeps class balance)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Stronger classifier
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== FINAL BALANCED AQI RISK MODEL =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))
