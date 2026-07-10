import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("="*50)
print("SMART CITY TRAFFIC PREDICTION")
print("="*50)

# Load Dataset
df = pd.read_csv("dataset/traffic.csv")

print(df.head())

# Data Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Features
features = ['Junction']
target = 'Vehicles'

X = df[features]
y = df[target]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

print("Prediction Completed Successfully")

# Graph
plt.figure(figsize=(8,5))
plt.plot(prediction)
plt.title("Traffic Prediction")
plt.xlabel("Samples")
plt.ylabel("Vehicle Count")
plt.show()

print("Smart City Traffic Prediction Completed")
