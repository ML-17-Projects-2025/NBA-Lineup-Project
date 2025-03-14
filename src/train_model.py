import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(input_file, model_output):
    """Train a Random Forest model on NBA lineup data."""
    df = pd.read_csv(input_file)
    
    # Define features and target
    feature_cols = [col for col in df.columns if col not in ['home_4', 'game', 'season']]
    target_col = 'home_4'
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save model
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")

# Example usage
if __name__ == "__main__":
    train_model("../data/matchups_2007_processed.csv", "../models/rf_model.pkl")
