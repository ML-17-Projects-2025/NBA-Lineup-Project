import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Define the file paths for the datasets
file_paths = [f"../datasets/original/matchups-{year}.csv" for year in range(2007, 2016)]
encoded_folder = "../datasets/encoded"

# Iterate through each file, read it, encode labels, and save the updated DataFrame
for file in file_paths:
    # Load the dataset
    matchup_data = pd.read_csv(file)

    # Encode categorical columns (players and teams) using LabelEncoder
    label_encoders = {}
    for col in ["home_0", "home_1", "home_2", "home_3", "home_4", "home_team", "away_team"]:
        le = LabelEncoder()
        matchup_data[col] = le.fit_transform(matchup_data[col])
        label_encoders[col] = le  # Save encoders for later decoding

    # Extract the file name from the path
    file_name = os.path.basename(file)  
    new_file_path = os.path.join(encoded_folder, file_name.replace(".csv", "-encoded.csv"))

    # Save the updated DataFrame with encoded labels to a new file
    matchup_data.to_csv(new_file_path, index=False)
