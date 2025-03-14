import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file, output_file):
    """Preprocess the NBA matchup dataset."""
    df = pd.read_csv(input_file)
    
    # Encode categorical variables
    player_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    
    all_players = df[['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 
                      'away_0', 'away_1', 'away_2', 'away_3', 'away_4']].values.flatten()
    player_encoder.fit(all_players)
    
    all_teams = df[['home_team', 'away_team']].values.flatten()
    team_encoder.fit(all_teams)
    
    for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
        df[col] = player_encoder.transform(df[col])
    
    df['home_team'] = team_encoder.transform(df['home_team'])
    df['away_team'] = team_encoder.transform(df['away_team'])
    
    # Save processed file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    preprocess_data("../data/matchups_2007.csv", "../data/matchups_2007_processed.csv")
