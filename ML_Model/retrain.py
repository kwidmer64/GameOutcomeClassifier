"""
Retrains the game outcome classifier with 4 new PBP-derived features.

New features:
  turnover_diff       — rolling avg of (turnovers forced − turnovers committed)
  third_down_pct      — rolling avg of offensive 3rd-down conversion %
  third_down_stop_pct — rolling avg of defensive 3rd-down stop %
  sacks_per_game      — rolling avg of defensive sacks per game

Run from ML_Model/:
  python retrain.py
"""

import nflreadpy as nfl
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import joblib

SEASONS = list(range(2019, 2026))

# ── 1. Schedules ──────────────────────────────────────────────────────────────

print("Loading schedules...")
games = nfl.load_schedules(SEASONS)
games = games.drop([
    'game_id', 'gameday', 'weekday', 'gametime', 'location', 'old_game_id',
    'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn', 'away_rest',
    'home_rest', 'div_game', 'roof', 'surface', 'temp', 'wind',
    'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name',
    'away_coach', 'home_coach', 'referee', 'stadium_id', 'stadium'
])
games = games.filter(pl.col('game_type') == 'REG')
games = games.with_columns(
    (pl.col('result') > 0).cast(pl.Int8).alias('home_team_win')
)

# ── 2. Per-team game records ──────────────────────────────────────────────────

home = games.select(['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_team_win'])
home.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed', 'won']

away = games.select(['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score', 'home_team_win'])
away = away.with_columns((1 - pl.col('home_team_win')).alias('home_team_win'))
away.columns = ['season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed', 'won']

games_full = pl.concat([home, away]).sort(['season', 'week'])

# ── 3. Existing rolling features ──────────────────────────────────────────────

games_full = games_full.with_columns(
    (pl.col('won').cum_sum().over(['team', 'season']) /
     pl.col('won').cum_count().over(['team', 'season'])).alias('win_percentage'),
    (pl.col('points_scored').cum_sum().over(['team', 'season']) /
     pl.col('points_scored').cum_count().over(['team', 'season'])).alias('points_per_game'),
    (pl.col('points_allowed').cum_sum().over(['team', 'season']) /
     pl.col('points_allowed').cum_count().over(['team', 'season'])).alias('allowed_points_per_game'),
).with_columns(
    pl.col('win_percentage').shift(1).over(['team', 'season']),
    pl.col('points_per_game').shift(1).over(['team', 'season']),
    pl.col('allowed_points_per_game').shift(1).over(['team', 'season']),
)

# ── 4. PBP data ───────────────────────────────────────────────────────────────

print("Loading play-by-play data (this may take a few minutes)...")
pbp = nfl.load_pbp(SEASONS)

pbp = pbp.filter(
    (pl.col('season_type') == 'REG') &
    pl.col('posteam').is_not_null() &
    pl.col('defteam').is_not_null()
).select([
    'season', 'week', 'posteam', 'defteam', 'down',
    'third_down_converted', 'interception', 'fumble_lost', 'sack',
]).with_columns([
    pl.col('third_down_converted').cast(pl.Float64).fill_null(0),
    pl.col('interception').cast(pl.Float64).fill_null(0),
    pl.col('fumble_lost').cast(pl.Float64).fill_null(0),
    pl.col('sack').cast(pl.Float64).fill_null(0),
    pl.col('down').cast(pl.Float64).fill_null(0),
])

# Offensive: 3rd down conversion %
pbp_3rd = pbp.filter(pl.col('down') == 3)

off_3rd = (
    pbp_3rd.group_by(['season', 'week', 'posteam'])
    .agg([
        pl.col('third_down_converted').sum().alias('third_conv'),
        pl.col('third_down_converted').count().alias('third_att'),
    ])
    .with_columns(
        (pl.col('third_conv') / pl.col('third_att').clip(lower_bound=1)).alias('third_down_pct_game')
    )
    .select(['season', 'week', 'posteam', 'third_down_pct_game'])
    .rename({'posteam': 'team'})
)

# Offensive: turnovers committed
off_to = (
    pbp.with_columns(
        (pl.col('interception') + pl.col('fumble_lost')).alias('to_committed')
    )
    .group_by(['season', 'week', 'posteam'])
    .agg(pl.col('to_committed').sum().alias('to_committed_game'))
    .select(['season', 'week', 'posteam', 'to_committed_game'])
    .rename({'posteam': 'team'})
)

# Defensive: 3rd down stop %
def_3rd = (
    pbp_3rd.group_by(['season', 'week', 'defteam'])
    .agg([
        pl.col('third_down_converted').sum().alias('third_conv_allowed'),
        pl.col('third_down_converted').count().alias('third_att_def'),
    ])
    .with_columns(
        (1 - pl.col('third_conv_allowed') / pl.col('third_att_def').clip(lower_bound=1)).alias('third_down_stop_pct_game')
    )
    .select(['season', 'week', 'defteam', 'third_down_stop_pct_game'])
    .rename({'defteam': 'team'})
)

# Defensive: sacks + turnovers forced
def_stats = (
    pbp.with_columns(
        (pl.col('interception') + pl.col('fumble_lost')).alias('to_forced')
    )
    .group_by(['season', 'week', 'defteam'])
    .agg([
        pl.col('sack').sum().alias('sacks_game'),
        pl.col('to_forced').sum().alias('to_forced_game'),
    ])
    .select(['season', 'week', 'defteam', 'sacks_game', 'to_forced_game'])
    .rename({'defteam': 'team'})
)

# ── 5. Merge PBP stats into games_full ───────────────────────────────────────

games_full = (
    games_full
    .join(off_3rd,   on=['season', 'week', 'team'], how='left')
    .join(def_3rd,   on=['season', 'week', 'team'], how='left')
    .join(def_stats, on=['season', 'week', 'team'], how='left')
    .join(off_to,    on=['season', 'week', 'team'], how='left')
)

games_full = games_full.with_columns(
    (pl.col('to_forced_game').fill_null(0) - pl.col('to_committed_game').fill_null(0)).alias('turnover_diff_game')
)

# ── 6. Rolling averages for new features ─────────────────────────────────────

games_full = games_full.with_columns(
    (pl.col('third_down_pct_game').cum_sum().over(['team', 'season']) /
     pl.col('third_down_pct_game').cum_count().over(['team', 'season'])).alias('third_down_pct'),
    (pl.col('third_down_stop_pct_game').cum_sum().over(['team', 'season']) /
     pl.col('third_down_stop_pct_game').cum_count().over(['team', 'season'])).alias('third_down_stop_pct'),
    (pl.col('sacks_game').cum_sum().over(['team', 'season']) /
     pl.col('sacks_game').cum_count().over(['team', 'season'])).alias('sacks_per_game'),
    (pl.col('turnover_diff_game').cum_sum().over(['team', 'season']) /
     pl.col('turnover_diff_game').cum_count().over(['team', 'season'])).alias('turnover_diff'),
).with_columns(
    pl.col('third_down_pct').shift(1).over(['team', 'season']),
    pl.col('third_down_stop_pct').shift(1).over(['team', 'season']),
    pl.col('sacks_per_game').shift(1).over(['team', 'season']),
    pl.col('turnover_diff').shift(1).over(['team', 'season']),
)

# ── 7. Join home/away stats back to game level ────────────────────────────────

def select_team_stats(side: str) -> pl.DataFrame:
    col = f'{side}_team'
    return (
        games_full.join(
            games.select(['season', 'week', 'home_team', 'away_team']),
            left_on=['season', 'week', 'team'],
            right_on=['season', 'week', col],
        )
        .rename({
            'team': col,
            'win_percentage': f'{side}_win_pct',
            'points_per_game': f'{side}_ppg',
            'allowed_points_per_game': f'{side}_opp_ppg',
            'third_down_pct': f'{side}_third_down_pct',
            'third_down_stop_pct': f'{side}_third_down_stop_pct',
            'sacks_per_game': f'{side}_sacks_per_game',
            'turnover_diff': f'{side}_turnover_diff',
        })
        .select([
            'season', 'week', col,
            f'{side}_win_pct', f'{side}_ppg', f'{side}_opp_ppg',
            f'{side}_third_down_pct', f'{side}_third_down_stop_pct',
            f'{side}_sacks_per_game', f'{side}_turnover_diff',
        ])
    )

home_stats = select_team_stats('home')
away_stats = select_team_stats('away')

gws = (
    games
    .join(home_stats, on=['season', 'week', 'home_team'])
    .join(away_stats, on=['season', 'week', 'away_team'])
)

# ── 8. Save CSVs ─────────────────────────────────────────────────────────────

gws_train = gws.filter(pl.col('season') < 2025)
gws_2025  = gws.filter(pl.col('season') == 2025)

gws_train.write_csv('../Backend/data/nfl_features.csv')
gws_2025.write_csv('../Backend/data/nfl_2025_stats.csv')
print(f"Training rows: {len(gws_train)} | 2025 rows: {len(gws_2025)}")

# Latest per-team stats for runtime predictions
def build_team_stats(side: str, df: pl.DataFrame) -> pl.DataFrame:
    col = f'{side}_team'
    return df.select([
        'season', 'week', col,
        f'{side}_win_pct', f'{side}_ppg', f'{side}_opp_ppg',
        f'{side}_third_down_pct', f'{side}_third_down_stop_pct',
        f'{side}_sacks_per_game', f'{side}_turnover_diff',
    ]).rename({
        col: 'team',
        f'{side}_win_pct': 'win_pct',
        f'{side}_ppg': 'ppg',
        f'{side}_opp_ppg': 'opp_ppg',
        f'{side}_third_down_pct': 'third_down_pct',
        f'{side}_third_down_stop_pct': 'third_down_stop_pct',
        f'{side}_sacks_per_game': 'sacks_per_game',
        f'{side}_turnover_diff': 'turnover_diff',
    })

team_stats = (
    pl.concat([build_team_stats('home', gws_2025), build_team_stats('away', gws_2025)])
    .sort(['team', 'week'])
    .group_by('team')
    .last()
)
team_stats.write_csv('../Backend/data/nfl_2025_final_stats.csv')
print(f"2025 team stats saved: {len(team_stats)} teams")

# ── 9. Train model ────────────────────────────────────────────────────────────

print("\nTraining model...")
df = pl.read_csv('../Backend/data/nfl_features.csv')

features_full = [
    'home_win_pct', 'home_ppg', 'home_opp_ppg',
    'home_third_down_pct', 'home_third_down_stop_pct', 'home_sacks_per_game', 'home_turnover_diff',
    'away_win_pct', 'away_ppg', 'away_opp_ppg',
    'away_third_down_pct', 'away_third_down_stop_pct', 'away_sacks_per_game', 'away_turnover_diff',
]

features_reduced = [
    'home_win_pct', 'home_ppg', 'home_opp_ppg', 'home_third_down_stop_pct',
    'away_win_pct', 'away_ppg', 'away_opp_ppg', 'away_third_down_stop_pct',
]

def evaluate(label: str, feature_set: list[str]) -> tuple:
    clean = df.drop_nulls(subset=feature_set)
    X_all = clean.select(feature_set).to_numpy()
    y_all = clean.select('home_team_win').to_numpy().ravel()
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=False)
    m = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    m.fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, m.predict(X_te))
    cv_acc   = cross_val_score(m, X_all, y_all, cv=5, scoring='accuracy').mean()
    print(f"  {label:<40} test={test_acc:.4f}  CV={cv_acc:.4f}  (n={len(clean)})")
    return m, test_acc, cv_acc

print(f"\nBaseline logistic regression (3 features, no scaler): test=0.6877")
print(f"{'Feature set':<40} {'test acc':>10} {'5-fold CV':>10}  rows")
print(f"{'-'*65}")
_, full_test, full_cv = evaluate("7 features (all PBP)", features_full)
model, red_test, red_cv = evaluate("5 features (+ stop% only)", features_reduced)

# -- Feature importance for the reduced model ---------------------------------

clean_red = df.drop_nulls(subset=features_reduced)
X_red = clean_red.select(features_reduced).to_numpy()
y_red = clean_red.select('home_team_win').to_numpy().ravel()
_, X_te_red, _, y_te_red = train_test_split(X_red, y_red, test_size=0.2, random_state=42, shuffle=False)

print("\nFeature importance (permutation, reduced model):")
pi = permutation_importance(model, X_te_red, y_te_red, n_repeats=30, random_state=42)
for idx in pi.importances_mean.argsort()[::-1]:
    print(f"  {features_reduced[idx]:<35} {pi.importances_mean[idx]:+.4f} +/- {pi.importances_std[idx]:.4f}")

# -- Save best model (by CV) --------------------------------------------------

features = features_reduced if red_cv >= full_cv else features_full
best_model = model
best_label = "reduced (5 features)" if red_cv >= full_cv else "full (7 features)"
print(f"\nSaving: {best_label}")
print(classification_report(y_te_red, best_model.predict(X_te_red)))

joblib.dump(best_model, '../Backend/model/game_outcome_classifier.pkl')
print(f"Model saved to ../Backend/model/game_outcome_classifier.pkl")
