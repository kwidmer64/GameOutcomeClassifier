import joblib
import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "./model/game_outcome_classifier.pkl"
STATS_PATH = "./data/nfl_2025_final_stats.csv"

app = FastAPI()
model = joblib.load(MODEL_PATH)
stats_2025 = pl.read_csv(STATS_PATH)

class MatchupRequest(BaseModel):
    home: str
    away: str

@app.post("/predict")
def predict(request: MatchupRequest):
    # Get the stats for the home and away teams from the 2025 season
    home = stats_2025.filter(pl.col('team') == request.home)
    away = stats_2025.filter(pl.col('team') == request.away)

    # Select the features for both teams
    features = [[
        home['win_pct'][0],
        home['ppg'][0],
        home['opp_ppg'][0],
        away['win_pct'][0],
        away['ppg'][0],
        away['opp_ppg'][0]
    ]]

    # Predict the outcome of the matchup with the selected features
    prediction = model.predict(features)[0]

    # Calculate the probability of the prediction
    probability = model.predict_proba(features)[0]

    # Get the winner of the matchup
    winner = request.home if prediction == 1 else request.away

    # Get the confidence of the prediction (in %)
    # selecting index 0 if the home team wins otherwise index 1 for away team
    confidence = probability[1] if prediction == 1 else probability[0]

    return {
        "winner": winner,
        "confidence": round(float(confidence), 2),
        "home_team": request.home,
        "away_team": request.away
    }