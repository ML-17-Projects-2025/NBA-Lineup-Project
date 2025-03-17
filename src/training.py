import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load dataset
file_paths = [f"../datasets/encoded/matchups-{year}-encoded.csv" for year in range(2007, 2016)]
dataframes = [pd.read_csv(file) for file in file_paths]
matchup_data = pd.concat(dataframes, ignore_index=True)

# Load NBA encoded labels
label_encoders = joblib.load("nba_label_encoders.pkl")

# Define features and target
features = ["game", "season", "starting_min",
            "home_0", "home_1", "home_2", "home_3" "home_4",
            "away_0", "away_1", "away_2", "away_3", "away_4",
            "home_team", "away_team",]
target = "home_0", "home_1", "home_2", "home_3", "home_4"

# Convert features to numeric (downcast to float32)
matchup_data[features] = matchup_data[features].apply(pd.to_numeric, downcast='float')

# Convert target to category type (which uses less memory for categorical data)
matchup_data[target] = matchup_data[target].astype('category')

# Prepare the feature matrix (X) and target vector (y)
X = matchup_data[features]
y = matchup_data[target]

# Set number of iterations and initialize lists to store accuracy results
n_iterations = 100
train_accuracies = []
test_accuracies = []
predicted_players = []  # To store predicted player names

# Run model 100 times with different random states
for i in range(n_iterations):
    random_state = np.random.randint(0, 10000)  # Random state for each iteration

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialize the Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=random_state)

    # Train the model
    dt_model.fit(X_train, y_train)

    # Evaluate the model on training and test data
    train_accuracy = dt_model.score(X_train, y_train)
    test_accuracy = dt_model.score(X_test, y_test)

    # Example new lineup (replace with actual data)
    new_lineup = np.array([[123, 456, 789, 101, 202, 303, 10, 5, 8, 4, 2, 6, 3, 7, 2, 1, 12, 9, 5, 3, 20]])

    # Convert the new lineup to a DataFrame with feature names
    new_lineup_df = pd.DataFrame(new_lineup, columns=features)

    # Predict the fifth player
    predicted_home_4 = dt_model.predict(new_lineup_df)[0]

    # Decode predicted player ID back to player name
    predicted_player_name = label_encoders["home_4"].inverse_transform([predicted_home_4])[0]

    # Store the accuracy results and predicted player name
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    predicted_players.append(predicted_player_name)

    print(
        f"Iteration {i + 1}: Train Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}, Predicted Fifth Player = {predicted_player_name}")

# 9. Save the Model and Encoders for Future Use
model_path = "nba_fifth_player_model.pkl"
encoders_path = "nba_label_encoders.pkl"

joblib.dump(dt_model, model_path)
joblib.dump(label_encoders, encoders_path)

# Compute and display average accuracy over all runs
avg_train_accuracy = np.mean(train_accuracies)
avg_test_accuracy = np.mean(test_accuracies)

print("\nAverage Training Accuracy over 100 runs:", avg_train_accuracy)
print("Average Testing Accuracy over 100 runs:", avg_test_accuracy)

input('Press ENTER to exit')
