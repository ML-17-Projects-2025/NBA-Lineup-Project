import pandas as pd
import joblib
import argparse

def predict_fifth_player(model_file, input_file, output_file):
    """Generate predictions for the optimal fifth player."""
    model = joblib.load(model_file)
    df = pd.read_csv(input_file)
    
    # Remove the target column if present
    feature_cols = [col for col in df.columns if col not in ['home_4', 'game', 'season']]
    
    # Generate predictions
    predictions = model.predict(df[feature_cols])
    
    # Save predictions
    results = df[['game', 'home_team']].copy()
    results['fifth_player'] = predictions
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--output", required=True, help="Path to save predictions")
    
    args = parser.parse_args()
    predict_fifth_player(args.model, args.input, args.output)
