import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load test data
nba_test_path = "../datasets/NBA_test.csv"
nba_test_labels_path = "../datasets/NBA_test_labels.csv"

# Define the feature columns used during training
feature_columns = ["home_0", "home_1", "home_2", "home_3", "home_4",
                   "away_0", "away_1", "away_2", "away_3", "away_4",
                   "home_team", "away_team", "starting_min", "season"]

# Load test data and labels with necessary columns only
test_data = pd.read_csv(nba_test_path, usecols=feature_columns)
test_labels = pd.read_csv(nba_test_labels_path)

# Load trained model and encoders
model = joblib.load("nba_fifth_player_model.pkl")
label_encoders = joblib.load("nba_label_encoders.pkl")

# Detect columns that have '?' values
missing_player_column = None
for column in test_data.columns:
    if test_data[column].isin(["?"]).any():  # Check for '?' values in the column
        missing_player_column = column
        break

# If no missing player is found, raise an error or handle appropriately
if missing_player_column is None:
    raise ValueError("No missing player found in the test data.")

# Simulate the missing player by setting the corresponding column to NaN
test_data[missing_player_column] = np.nan

# Ensure the test data columns match the order used during training
test_data = test_data[feature_columns]

# Convert the data to numeric for model consistency
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Extract true missing player names from test_labels
y_test = test_labels["removed_value"].astype(str)

# Make predictions
predictions = model.predict(test_data)

# Decode predictions using the corresponding label encoder
decoded_predictions = label_encoders[missing_player_column].inverse_transform(predictions)

# Compute accuracy
accuracy = accuracy_score(y_test, decoded_predictions)

# Display results
print(f"Model Test Accuracy: {accuracy:.4f}")
print("Example Predictions:")
for actual, predicted in zip(y_test[:10], decoded_predictions[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")
