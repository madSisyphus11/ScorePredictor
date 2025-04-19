import os
import re
import logging
import traceback

import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, PULP_CBC_CMD

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong", 200

# Load squads
IPL_XLSX = os.path.join("data", "IPL_Data.xlsx")
xls = pd.ExcelFile(IPL_XLSX)
sheet_name = next((s for s in xls.sheet_names if "squad" in s.lower() or "team" in s.lower()), xls.sheet_names[0])
squads_df = pd.read_excel(IPL_XLSX, sheet_name=sheet_name)
SQUADS = {team: squads_df.at[0, team].split("\n") for team in squads_df.columns}

# Load model and patch missing attrs
MODEL_PATH = os.path.join("model", "xgb_model.pkl")
model = joblib.load(MODEL_PATH)
if not hasattr(model, "gpu_id"):
    model.gpu_id = None
if not hasattr(model, "predictor"):
    model.predictor = "auto"

def extract_features(players, match_context):
    """
    Stub that returns all features the XGB model expects:
      runs, fours, sixes, balls,
      wickets, runs_conceded, deliveries_bowled, field_points
    plus metadata: player, team, role, hist_std, cricket_credit, is_foreign
    """
    n = len(players)
    # Dummy zeros (or any reasonable default)
    df = pd.DataFrame({
        "player":             players,
        "team":               players,               # placeholder; real code would assign properly
        "role":               ["Batter"] * n,
        "hist_std":           [10.0] * n,
        "cricket_credit":     [8] * n,
        "is_foreign":         [False] * n,
        # XGB features:
        "runs":               [0] * n,
        "fours":              [0] * n,
        "sixes":              [0] * n,
        "balls":              [0] * n,
        "wickets":            [0] * n,
        "runs_conceded":      [0] * n,
        "deliveries_bowled":  [0] * n,
        "field_points":       [0] * n,
    })
    return df

def predict_player_stats(features_df):
    required = {"player","team","role","hist_std","cricket_credit","is_foreign"}
    missing = required - set(features_df.columns)
    if missing:
        raise KeyError(f"Missing columns for stub: {missing}")

    # Drop metadata, leaving the 8 numeric features
    X = features_df.drop(columns=list(required))
    preds = model.predict(X)

    std = features_df["hist_std"].values
    ceiling = preds + 1.5 * std
    floor   = preds - np.minimum(std, 0.8 * preds)
    floor   = np.clip(floor, 0, None)

    eps = 1.0
    dh = (ceiling - floor) / np.where(preds > eps, preds, eps)

    out = features_df[["player","team","role","cricket_credit","is_foreign"]].copy()
    out["pred"]       = np.round(preds,   2)
    out["std"]        = np.round(std,     2)
    out["floor"]      = np.round(floor,   2)
    out["ceiling"]    = np.round(ceiling, 2)
    out["dark_horse"] = np.round(dh,      2)
    return out

def select_best_team(preds_df, budget=100, max_from_team=7, max_foreign=4):
    costs   = dict(zip(preds_df.player, preds_df.cricket_credit))
    foreign = dict(zip(preds_df.player, preds_df.is_foreign))
    roles   = dict(zip(preds_df.player, preds_df.role))

    prob = LpProblem("Dream11", LpMaximize)
    x = {p: LpVariable(f"x_{i}", cat="Binary") for i,p in enumerate(preds_df.player)}

    # Objective & constraints as before
    prob += lpSum(preds_df.loc[preds_df.player==p,"pred"].iloc[0] * x[p] for p in x)
    prob += lpSum(x.values()) == 11
    prob += lpSum(costs[p] * x[p] for p in x) <= budget
    for team in preds_df.team.unique():
        prob += lpSum(x[p] for p in x 
                      if preds_df.loc[preds_df.player==p,"team"].iloc[0]==team) \
                <= max_from_team
    prob += lpSum(x[p] for p in x if foreign[p]) <= max_foreign
    for must in ["WK-BAT","Batter","Allrounder","Bowler"]:
        prob += lpSum(x[p] for p in x if roles[p]==must) >= 1

    # --- Use CBC solver with a small time limit to avoid hanging ---
    solver = PULP_CBC_CMD(msg=False, timeLimit=5)
    prob.solve(solver)

    sel = [p for p in x if x[p].value()==1]
    best = preds_df[preds_df.player.isin(sel)]
    backups = preds_df[~preds_df.player.isin(sel)].nlargest(5, "pred")
    return best.to_dict("records"), backups.to_dict("records")

@app.route("/", methods=["GET","POST"])
def index():
    logging.info("INDEX HIT")
    if request.method=="POST":
        return redirect(url_for("predict", match_url=request.form["match_url"]))
    return render_template("index.html")

@app.route("/predict")
def predict():
    try:
        match_url = request.args.get("match_url","").strip()
        if not match_url:
            return redirect(url_for("index"))

        m = re.search(r"/(\d{5,6})/.*?([a-z]+)-vs-([a-z]+)-", match_url)
        if not m:
            return "<h1>Invalid match URL</h1>", 400
        _, c1, c2 = m.groups()

        TEAM_MAP = {
            "rcb":"Royal Challengers Bengaluru","pbks":"Punjab Kings",
            "rr":"Rajasthan Royals","lsg":"Lucknow Super Giants",
            "mi":"Mumbai Indians","csk":"Chennai Super Kings",
            "kkr":"Kolkata Knight Riders","dc":"Delhi Capitals",
            "srh":"Sunrisers Hyderabad","gt":"Gujarat Titans"
        }
        t1 = TEAM_MAP.get(c1.lower()); t2 = TEAM_MAP.get(c2.lower())
        if not t1 or not t2:
            return "<h1>Unknown team code</h1>", 400

        players = SQUADS.get(t1,[]) + SQUADS.get(t2,[])
        ctx = {"venue":"", "opp":None}

        feats = extract_features(players, ctx)
        stats = predict_player_stats(feats)
        best, backs = select_best_team(stats)

        return render_template("results.html",
                               url=match_url,
                               best_xi=best,
                               backups=backs,
                               all_players=stats.to_dict("records"))
    except Exception:
        tb = traceback.format_exc()
        logging.error(tb)
        return f"<h1>Prediction Error</h1><pre>{tb}</pre>", 500

if __name__=="__main__":
    app.run(debug=True)
