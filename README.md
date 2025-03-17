# NBA Lineup Prediction

## Project Overview
This project aims to develop a machine learning model that predicts the optimal fifth player for an NBA home team lineup based on historical game data. The model is trained using NBA matchups from 2007 to 2015, utilizing allowed features to optimize team performance.

## Dataset
- **Source**: NBA matchup data from 2007 to 2015.
- **Features Used**: Player lineups, game statistics, and team performance metrics (as specified in the metadata file).
- **Target Variable**: The optimal fifth player for the home team.

## Model
- **Algorithm**: Random Forest Classifier
- **Features**:
  - Encoded player and team names.
  - Player performance metrics aggregated from historical data.
  - Game context features (e.g., points, assists, rebounds).
- **Performance**: Achieved ~99.98% accuracy in predicting historical player selections.

## Installation & Usage
### Prerequisites
Ensure you have Python 3 installed and the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model
1. Clone the repository:
```bash
git clone https://github.com/your-repo/NBA-Lineup-Prediction.git
cd NBA-Lineup-Prediction
```
2. Run the prediction script:
```bash
python src/predict.py --input data/test_data.csv --output data/predictions.csv
```

## Deliverables
- **Trained Model**: [`rf_model.pkl`](src/rf_model.pkl)
- **Predictions**: [`predicted_fifth_players.csv`](output_files/predicted_fifth_players.csv)
- **Final Report**: `report.pdf`
- **Presentation Slides**: `presentation.pptx`

## Authors
- Rolf-Jaden Sibal, Dmitri Rios-Nadeau, Suluxan Manogharan
- Team 17

## License
¯\_(ツ)_/¯

