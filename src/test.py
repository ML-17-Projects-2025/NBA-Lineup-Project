import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load test data
nba_test_path = "../datasets/NBA_test.csv"
test_data = pd.read_csv(nba_test_path)

# Load trained model and encoders
model = joblib.load("nba_fifth_player_model.pkl")
label_encoders = joblib.load("nba_label_encoders.pkl")

# Define feature columns used during training
feature_columns = ["season", "home_team", "away_team", "starting_min",
                   "home_0", "home_1", "home_2", "home_3", "home_4",
                   "away_0", "away_1", "away_2", "away_3", "away_4"]

# Ensure test data has the necessary features
test_data = test_data[feature_columns]

# Convert to numeric type for consistency
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Extract missing player labels (assumed to be in a column named "missing_player")
y_test = test_data["home_4"].copy()
test_data["home_4"] = np.nan  # Simulate missing player

# Make predictions
predictions = model.predict(test_data)

# Decode predictions
decoded_predictions = label_encoders["home_4"].inverse_transform(predictions)
decoded_actuals = label_encoders["home_4"].inverse_transform(y_test.dropna().astype(int))

# Compute accuracy
accuracy = accuracy_score(decoded_actuals, decoded_predictions)

# Display results
print(f"Model Test Accuracy: {accuracy:.4f}")
print("Example Predictions:")
for actual, predicted in zip(decoded_actuals[:10], decoded_predictions[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
