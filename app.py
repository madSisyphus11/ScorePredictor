import os
import re
import logging
import traceback

import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for
from pulp import LpProblem, LpVariable, lpSum, LpMaximize

# ——— Setup logging ———
logging.basicConfig(level=logging.INFO)

# ——— Flask app ———
app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong", 200

# ——— Load squads dynamically ———
IPL_XLSX = os.path.join("data", "IPL_Data.xlsx")
xls = pd.ExcelFile(IPL_XLSX)
logging.info(f"Available sheets in {IPL_XLSX}: {xls.sheet_names}")

# Pick the first sheet containing 'squad' or 'team', else first sheet
sheet_name = next(
    (s for s in xls.sheet_names if "squad" in s.lower() or "team" in s.lower()),
    xls.sheet_names[0]
)
logging.info(f"Loading squads from sheet: {sheet_name}")
squads_df = pd.read_excel(IPL_XLSX, sheet_name=sheet_name)
SQUADS = {
    team: squads_df.at[0, team].split("\n")
    for team in squads_df.columns
}

# ——— Load the trained model ———
MODEL_PATH = os.path.join("model", "xgb_model.pkl")
model = joblib.load(MODEL_PATH)
# Monkey‑patch missing gpu_id
if not hasattr(model, "gpu_id"):
    model.gpu_id = None

# ——— Feature extraction stub ———
def extract_features(players, match_context):
    """
    Stub: returns DataFrame with columns needed downstream.
    """
    df = pd.DataFrame({
        "player": players,
        "team": [p for p in players],
        "role": ["Batter"] * len(players),
        "hist_std": [10.0] * len(players),
        "cricket_credit": [8] * len(players),
        "is_foreign": [False] * len(players)
    })
    return df

# ——— Predict stats with sanity checks & rounding ———
def predict_player_stats(features_df):
    required = {"player","team","role","hist_std","cricket_credit","is_foreign"}
    missing = required - set(features_df.columns)
    if missing:
        raise KeyError(f"Features missing columns: {missing}")

    # Drop identifier columns, leave only numeric features for the model
    X = features_df.drop(columns=list(required))
    preds = model.predict(X)

    std = features_df["hist_std"].values
    ceiling = preds + 1.5 * std
    floor   = preds - np.minimum(std, 0.8 * preds)
    floor   = np.clip(floor, 0, None)

    # Avoid division by near‑zero predictions
    eps = 1.0
    dh = (ceiling - floor) / np.where(preds > eps, preds, eps)

    out = features_df[["player","team","role","cricket_credit","is_foreign"]].copy()
    out["pred"]       = np.round(preds,   2)
    out["std"]        = np.round(std,     2)
    out["floor"]      = np.round(floor,   2)
    out["ceiling"]    = np.round(ceiling, 2)
    out["dark_horse"] = np.round(dh,      2)
    return out

# ——— Dream11 team optimizer ———
def select_best_team(preds_df, budget=100, max_from_team=7, max_foreign=4):
    costs   = dict(zip(preds_df.player, preds_df.cricket_credit))
    foreign = dict(zip(preds_df.player, preds_df.is_foreign))
    roles   = dict(zip(preds_df.player, preds_df.role))

    prob = LpProblem("Dream11", LpMaximize)
    x = {p: LpVariable(f"x_{i}", cat="Binary") for i,p in enumerate(preds_df.player)}

    # Objective: maximize predicted points
    prob += lpSum(preds_df.loc[preds_df.player==p,"pred"].iloc[0] * x[p] for p in x)

    # Exactly 11 players
    prob += lpSum(x.values()) == 11
    # Budget cap
    prob += lpSum(costs[p] * x[p] for p in x) <= budget
    # Team caps
    for team in preds_df.team.unique():
        prob += lpSum(x[p] for p in x if preds_df.loc[preds_df.player==p,"team"].iloc[0]==team) <= max_from_team
    # Foreign cap
    prob += lpSum(x[p] for p in x if foreign[p]) <= max_foreign
    # Role minima
    for must in ["WK-BAT","Batter","Allrounder","Bowler"]:
        prob += lpSum(x[p] for p in x if roles[p]==must) >= 1

    prob.solve()
    sel = [p for p in x if x[p].value()==1]
    best = preds_df[preds_df.player.isin(sel)]
    backups = preds_df[~preds_df.player.isin(sel)].nlargest(5, "pred")
    return best.to_dict("records"), backups.to_dict("records")

# ——— Routes ———
@app.route("/", methods=["GET","POST"])
def index():
    logging.info("INDEX ROUTE HIT")
    if request.method == "POST":
        return redirect(url_for("predict", match_url=request.form["match_url"]))
    return render_template("index.html")

@app.route("/predict")
def predict():
    try:
        match_url = request.args.get("match_url","").strip()
        if not match_url:
            return redirect(url_for("index"))

        # Parse match ID & team codes
        m = re.search(r"/(\d{5,6})/.*?([a-z]+)-vs-([a-z]+)-", match_url)
        if not m:
            return "<h1>Invalid match URL</h1>", 400
        _, code1, code2 = m.groups()

        # Map URL codes to full squad keys
        TEAM_MAP = {
            "rcb": "Royal Challengers Bengaluru",
            "pbks": "Punjab Kings",
            # add the other IPL teams here...
        }
        team1 = TEAM_MAP.get(code1.lower())
        team2 = TEAM_MAP.get(code2.lower())
        if not team1 or not team2:
            return "<h1>Unknown team code</h1>", 400

        # Fetch player lists
        squad1 = SQUADS.get(team1, [])
        squad2 = SQUADS.get(team2, [])
        players = squad1 + squad2
        match_context = {"venue":"", "opp":None}

        # Run the real prediction pipeline
        features_df = extract_features(players, match_context)
        stats_df    = predict_player_stats(features_df)
        best_xi, backups = select_best_team(stats_df)

        return render_template(
            "results.html",
            url=match_url,
            best_xi=best_xi,
            backups=backups,
            all_players=stats_df.to_dict("records")
        )

    except Exception:
        tb = traceback.format_exc()
        logging.error("Prediction exception:\n%s", tb)
        return f"<h1>Prediction Error</h1><pre>{tb}</pre>", 500

if __name__ == "__main__":
    app.run(debug=True)
