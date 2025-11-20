import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Loaded dataset 
try:
    df = pd.read_csv("data/flight_fares.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/flight_fares.csv' not found. Please ensure the file is in the 'data' folder.")
    exit()

#  Droped columns
df = df.drop(columns=["Unnamed: 0", "flight"], errors="ignore")

#  Encoded categorical by LabelEncoder
label_encoders = {}
categorical_cols = [
    "airline", "source_city", "departure_time",
    "stops", "arrival_time", "destination_city", "class"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # store encoder for later use

#   features of X and target y
X = df.drop(columns=["price"])
y = df["price"]

# Train ,test ,split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Trained RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Evaluated the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model Evaluation - MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}")

#  Save trained model and label encoders together
with open("fare_model.pkl", "wb") as f:
    pickle.dump((model, label_encoders), f)

print("✅ Model and encoders saved as 'fare_model.pkl'")
