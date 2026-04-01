import joblib
import polars as pl
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Paths to the models and data
OUTCOME_MODEL_PATH = "./model/game_outcome_classifier.pkl"
SCORE_MODEL_PATH = "./model/nfl_score_model.joblib"
STATS_PATH = "./data/nfl_2025_final_stats.csv"

# Define team names and abreviations
team_names = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders"
}

# Init FastAPI
app = FastAPI()

# Load the models
outcome_model = joblib.load(OUTCOME_MODEL_PATH)
score_model = joblib.load(SCORE_MODEL_PATH)

# Get the score bucket labels from the model
BUCKET_LABELS = score_model.classes_

# Load the stats into a Polars dataframe
stats_2025 = pl.read_csv(STATS_PATH)

# Define the request body for the prediction endpoint
class MatchupRequest(BaseModel):
    home_team: str
    away_team: str

# POST /predict endpoint
@app.post("/predict")
def predict(request: MatchupRequest):
    # Get the stats for the home and away teams from the 2025 season
    home = stats_2025.filter(pl.col('team') == request.home_team)
    away = stats_2025.filter(pl.col('team') == request.away_team)

    # === Predict the outcome of the game ===

    # Select the features for both teams
    outcome_features = [[
        home['win_pct'][0],
        home['ppg'][0],
        home['opp_ppg'][0],
        away['win_pct'][0],
        away['ppg'][0],
        away['opp_ppg'][0]
    ]]

    # Predict the outcome of the matchup with the selected features
    outcome_prediction = outcome_model.predict(outcome_features)[0]

    # Calculate the probability of the prediction
    outcome_probability = outcome_model.predict_proba(outcome_features)[0]

    # Get the winner of the matchup
    winner = request.home_team if outcome_prediction == 1 else request.away_team

    # Get the confidence of the prediction (in %)
    # selecting index 0 if the home team wins otherwise index 1 for away team
    confidence = outcome_probability[1] if outcome_prediction == 1 else outcome_probability[0]

    # === Predict the score range of the game ===

    # Create features for home and away
    # The model is run once per team so we need separate features
    home_score_features = [[
        home['ppg'][0],     # Home team points per game
        home['opp_ppg'][0], # Home team points allowed per game
        away['opp_ppg'][0]  # Opponents points allowed per game
    ]]

    away_score_features = [[
        away['ppg'][0],
        away['opp_ppg'][0],
        home['opp_ppg'][0]
    ]]

    # Use the model to predict the probabilities for each score bucket for the teams.
    home_score_probability = score_model.predict_proba(home_score_features)[0]
    away_score_probability = score_model.predict_proba(away_score_features)[0]

    # For home and away score probabilities, create a dictionary that maps labels to probabilities.
    # This is a shorthand way to build a dictionary in Python called dctionary comprehension.
    # Zip function: built in to Python and can combine two lists into a set of tuples, pairing the first items together, then the second items, etc.
    home_score_dict = {label: round(probability, 2) for label, probability in zip(BUCKET_LABELS, home_score_probability.tolist())}
    away_score_dict = {label: round(probability, 2) for label, probability in zip(BUCKET_LABELS, away_score_probability.tolist())}

    # Return the predicted winner, confidence, and score probabilities for both teams.
    return {
        "winner": winner,
        "confidence": round(float(confidence), 2),
        "home_team": request.home_team,
        "away_team": request.away_team,

        "home_score_probabilities": home_score_dict,
        "away_score_probabilities": away_score_dict
    }

# GET /teams endpoints
@app.get('/teams')
def get_teams():
    teams = {team: team_names[team] for team in sorted(stats_2025['team'].to_list())}
    return teams

# === Error handling for validation errors ===
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)