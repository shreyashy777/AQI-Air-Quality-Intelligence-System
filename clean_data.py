import pandas as pd

# Load dataset
df = pd.read_csv("aqi_data.csv")

print("Initial shape:", df.shape)

# Drop columns we will NOT use
columns_to_drop = ["City", "Datetime", "AQI_Bucket"]
df = df.drop(columns=columns_to_drop)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing values with column mean
df = df.fillna(df.mean(numeric_only=True))

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal shape:", df.shape)

# Separate features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

print("\nFeatures used for prediction:")
print(X.columns)

print("\nTarget variable:")
print(y.name)
