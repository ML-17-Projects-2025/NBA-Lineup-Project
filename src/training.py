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

# Debug: Check if data is loaded properly
print("Dataset loaded. Number of rows:", matchup_data.shape[0])

# Load NBA encoded labels
if os.path.exists("nba_label_encoders.pkl"):
    label_encoders = joblib.load("nba_label_encoders.pkl")
else:
    raise FileNotFoundError("nba_label_encoders.pkl not found!")

# Define features (all player positions and other relevant features)
player_positions = ["home_0", "home_1", "home_2", "home_3", "home_4"]
features = player_positions + ["away_0", "away_1", "away_2", "away_3", "away_4", "home_team", "away_team", "starting_min", "season"]

# Augment dataset: create multiple versions where each home_* player is missing
augmented_data = []

print("Augmenting dataset...")
for idx, row in matchup_data.iterrows():
    for missing_index in range(5):  # Iterate over home_0 to home_4
        temp_row = row.copy()
        missing_player = temp_row[player_positions[missing_index]]  # Store the missing player as target
        temp_row[player_positions[missing_index]] = np.nan  # Remove player
        augmented_data.append((temp_row.drop(columns=player_positions), missing_player))
        
    
    # Debug: Print progress every 10,000 rows
    if len(augmented_data) % 10000 == 0:
        print(f"Processed {len(augmented_data)} augmented rows...")

# Convert augmented data into DataFrame
X_augmented = pd.DataFrame([x[0] for x in augmented_data])
y_augmented = np.array([x[1] for x in augmented_data])

print(X_augmented.dtypes)
print(X_augmented.head())

# Ensure numerical consistency
X_augmented = X_augmented.apply(pd.to_numeric, downcast='float')

# Convert target to categorical
y_augmented = pd.Series(y_augmented).astype('category')

# Prepare feature matrix (X) and target vector (y)
X = X_augmented
y = y_augmented

# Initialize lists to store accuracy results
n_iterations = 100
train_accuracies = []
test_accuracies = []
predicted_players = []

# Run model 100 times with different random states
for i in range(n_iterations):
    rng = np.random.default_rng()
    random_state = rng.integers(0, 10000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=random_state, ccp_alpha=0.0)
    dt_model.fit(X_train, y_train)

    # Evaluate
    train_accuracy = dt_model.score(X_train, y_train)
    test_accuracy = dt_model.score(X_test, y_test)

    # Example new lineup with a missing player (ensure it matches features)
    new_lineup_dict = {col: np.nan for col in X.columns}  # Create a dictionary with all required features
    new_lineup_dict.update({
    "home_0": 123, "home_1": 456, "home_2": 789, "home_3": 101,
    "away_0": 303, "away_1": 10, "away_2": 5, "away_3": 8, "away_4": 4,
    "home_team": 2, "away_team": 6, "starting_min": 3, "season": 2,
    })

    
    # Convert to DataFrame
    new_lineup_df = pd.DataFrame([new_lineup_dict])

    # Predict and decode
    predicted_home = int(round(dt_model.predict(new_lineup_df)[0]))
    predicted_player_name = label_encoders["home_4"].inverse_transform([predicted_home])[0]

    # Store results
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    predicted_players.append(predicted_player_name)

    print(f"Iteration {i + 1}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}, Predicted: {predicted_player_name}")

# Save model and encoders
joblib.dump(dt_model, "nba_fifth_player_model.pkl")
joblib.dump(label_encoders, "nba_label_encoders.pkl")

# Compute and display average accuracy
print("\nAverage Training Accuracy:", np.mean(train_accuracies))
print("Average Testing Accuracy:", np.mean(test_accuracies))

input('Press ENTER to exit')