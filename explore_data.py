import pandas as pd

# Load the dataset
df = pd.read_csv("aqi_data.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)
