import pickle

# Load model + encoders
with open("fare_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

print("✅ Keys available in label_encoders:\n", label_encoders.keys())

# Loop through all encoders and print their classes
print("\n✅ Categories learned by model:\n")
for key, encoder in label_encoders.items():
    print(f"{key}: {encoder.classes_}")
