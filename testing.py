import json
import pandas as pd

# Load model feature list
with open("models/features.json") as f:
    model_features = json.load(f)

print("Model expects:", len(model_features), "features")

# Try your suspected dataset
df = pd.read_csv("data/external/enhanced_realistic_smartgrid.csv")

print("Dataset features:", len(df.columns))
print("Matching columns:",
      set(model_features).issubset(set(df.columns)))