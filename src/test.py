import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load test data
nba_test_path = "../datasets/NBA_test.csv"
nba_test_labels_path = "../datasets/NBA_test_labels.csv"
test_data = pd.read_csv(nba_test_path)
test_labels = pd.read_csv(nba_test_labels_path)

# Load trained model and encoders
model = joblib.load("nba_fifth_player_model.pkl")
label_encoders = joblib.load("nba_label_encoders.pkl")

# Define feature columns used during training
feature_columns = ["home_0", "home_1", "home_2", "home_3", "home_4",
                   "away_0", "away_1", "away_2", "away_3", "away_4",
                   "home_team", "away_team", "starting_min", "season"]

# Ensure test data has the necessary features
test_data = test_data[feature_columns]

# Convert to numeric type for consistency
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Extract true missing player names from test_labels
y_test = test_labels["removed_value"].astype(str)

# Remove the missing player (simulate as model expects)
test_data["home_4"] = np.nan

# Make predictions
predictions = model.predict(test_data)

# Decode predictions
decoded_predictions = label_encoders["home_4"].inverse_transform(predictions)

# Compute accuracy
accuracy = accuracy_score(y_test, decoded_predictions)

# Display results
print(f"Model Test Accuracy: {accuracy:.4f}")
print("Example Predictions:")
for actual, predicted in zip(y_test[:10], decoded_predictions[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
