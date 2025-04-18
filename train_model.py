import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import os

# Load data
matches = pd.read_csv('matches.csv', parse_dates=['date'])
deliveries = pd.read_csv('deliveries.csv')

# --- add these two lines ---
matches.columns   = matches.columns.str.lower()
deliveries.columns = deliveries.columns.str.lower()

# Batting stats
bat = deliveries.groupby(['match_id','batsman']).agg(
    runs=('batsman_runs','sum'),
    fours=('batsman_runs', lambda x: (x==4).sum()),
    sixes=('batsman_runs', lambda x: (x==6).sum()),
    balls=('batsman_runs','count'),
    dismissals=('player_dismissed', lambda x: x.notnull().sum())
).reset_index().rename(columns={'batsman':'player'})

# Bowling stats
bowl = deliveries.groupby(['match_id','bowler']).agg(
    wickets=('player_dismissed', lambda x: x.notnull().sum()),
    runs_conceded=('total_runs','sum'),
    deliveries_bowled=('ball','count')
).reset_index().rename(columns={'bowler':'player'})

# Fielding stats
field = deliveries[deliveries['dismissal_kind'].isin(['caught','stumped','run out'])] \
    .groupby(['match_id','fielder']).size().reset_index(name='field_points') \
    .rename(columns={'fielder':'player'})

df = pd.merge(bat, bowl, on=['match_id','player'], how='outer').fillna(0)
df = pd.merge(df, field, on=['match_id','player'], how='left').fillna(0)

def compute_points(r):
    pts = r['runs'] + r['fours']*1 + r['sixes']*2
    if r['runs']>=30: pts+=4
    if r['runs']>=50: pts+=8
    if r['runs']>=100: pts+=16
    if r['runs']==0 and r['balls']>0: pts-=2
    pts += r['wickets']*25
    pts += r['field_points']*8
    return pts

df['points'] = df.apply(compute_points, axis=1)

features = df[['runs','fours','sixes','balls','wickets','runs_conceded','deliveries_bowled','field_points']]
target = df['points']

model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
model.fit(features, target)

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/xgb_model.pkl')
print("Model trained and saved to model/xgb_model.pkl")
