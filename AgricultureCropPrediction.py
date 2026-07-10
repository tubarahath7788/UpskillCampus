import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("="*50)
print("AGRICULTURE CROP PRODUCTION PREDICTION")
print("="*50)

# Load Dataset
df = pd.read_csv("dataset/crop_production.csv")

# Display Dataset
print(df.head())

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Feature Selection
features = ['Area']
target = 'Production'

X = df[features]
y = df[target]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
score = r2_score(y_test, prediction)

print("Model Accuracy:", score)

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(df['Area'], df['Production'])
plt.xlabel("Area")
plt.ylabel("Production")
plt.title("Crop Production Prediction")
plt.show()

print("Project Completed Successfully")
