from flask import Flask, render_template, request, redirect, url_for
import logging
logging.basicConfig(level=logging.INFO)
import re, os, requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import numpy as np
import joblib
from pulp import LpProblem, LpVariable, lpSum, LpMaximize

app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong", 200

import pandas as pd
import logging
import os

# — initialize logging so we can see which sheet is used —
logging.basicConfig(level=logging.INFO)

# path to your Excel file
IPL_XLSX = os.path.join("data", "IPL_Data.xlsx")

# 1. List available sheets
xls = pd.ExcelFile(IPL_XLSX)
logging.info(f"Available sheets in {IPL_XLSX}: {xls.sheet_names}")

# 2. Pick the sheet that looks like your squads/team sheet
sheet_name = None
for name in xls.sheet_names:
    if "squad" in name.lower() or "team" in name.lower():
        sheet_name = name
        break
if sheet_name is None:
    # fallback to the first sheet
    sheet_name = xls.sheet_names[0]
logging.info(f"Loading squads from sheet: {sheet_name}")

# 3. Read that sheet
squads_df = pd.read_excel(IPL_XLSX, sheet_name=sheet_name)

# 4. Build your SQUADS dict
SQUADS = {
    team: squads_df.at[0, team].split("\n")
    for team in squads_df.columns
}

# Load model
MODEL_PATH = os.path.join("model", "xgb_model.pkl")
model = joblib.load(MODEL_PATH)

def extract_features(players, match_context):
    import pandas as pd
    # one row per player with the minimal columns needed
    df = pd.DataFrame({
        "player": players,
        "team": ["RCB" if "RCB" in p else "PBKS" for p in players],
        "role": ["Batter"] * len(players),
        "hist_std": [10] * len(players),
        "cricket_credit": [8] * len(players),
        "is_foreign": [False] * len(players),
    })
    return df

def predict_player_stats(features_df):
    preds = model.predict(features_df.drop(columns=["player","team","role"]))
    std = features_df["hist_std"].values
    ceiling = preds + 1.5 * std
    floor = preds - np.minimum(std, 0.8 * preds)
    floor = np.clip(floor, a_min=0, a_max=None)
    dh = (ceiling - floor) / np.where(preds>0, preds, 1)
    out = features_df[["player","team","role"]].copy()
    out["pred"] = preds
    out["std"] = std
    out["floor"] = floor
    out["ceiling"] = ceiling
    out["dark_horse"] = dh
    return out

def select_best_team(preds_df, budget=100, max_from_team=7, max_foreign=4):
    costs = dict(zip(preds_df.player, preds_df.cricket_credit))
    foreign = dict(zip(preds_df.player, preds_df.is_foreign))
    roles = dict(zip(preds_df.player, preds_df.role))

    prob = LpProblem("Dream11", LpMaximize)
    x = {p: LpVariable(f"x_{i}", cat="Binary") for i, p in enumerate(preds_df.player)}
    prob += lpSum(preds_df.loc[preds_df.player==p, "pred"].iloc[0] * x[p] for p in x)
    prob += lpSum(x.values()) == 11
    prob += lpSum(costs[p] * x[p] for p in x) <= budget
    for team in preds_df.team.unique():
        prob += lpSum(x[p] for p in x if preds_df.loc[preds_df.player==p, "team"].iloc[0]==team) <= max_from_team
    prob += lpSum(x[p] for p in x if foreign[p]) <= max_foreign
    for must in ["WK-BAT","Batter","Allrounder","Bowler"]:
        prob += lpSum(x[p] for p in x if roles[p]==must) >= 1
    prob.solve()
    sel = [p for p in x if x[p].value()==1]
    best = preds_df[preds_df.player.isin(sel)]
    backups = preds_df[~preds_df.player.isin(sel)].nlargest(5, "pred")
    return best.to_dict("records"), backups.to_dict("records")
    
import os

@app.route("/", methods=["GET","POST"])
@app.route("/", methods=["GET","POST"])
def index():
    logging.info("INDEX ROUTE HIT")
    if request.method == "POST":
        match_url = request.form.get("match_url")
        return redirect(url_for("predict", match_url=match_url))
    return render_template("index.html")

@app.route("/predict")
def predict():
    url = request.args.get("match_url","").strip()
    if not url:
        return redirect(url_for("index"))
    m = re.search(r"/(\d{5,6})/.*?([a-z]+)-vs-([a-z]+)-", url)
    if not m:
        return render_template("error.html", msg="Invalid URL.")
    match_id, t1, t2 = m.groups()
    TEAM_MAP = {"rcb":"Royal Challengers Bengaluru","pbks":"Punjab Kings"}
    team1, team2 = TEAM_MAP.get(t1), TEAM_MAP.get(t2)
    squad1, squad2 = SQUADS[team1], SQUADS[team2]
    players = squad1 + squad2
    match_context = {"venue":"","opp":""}
    features_df = extract_features(players, match_context)
    stats_df = predict_player_stats(features_df)
    best_xi, backups = select_best_team(stats_df)
    return render_template("results.html", url=url, best_xi=best_xi, backups=backups, all_players=stats_df.to_dict("records"))

if __name__=="__main__":
    app.run(debug=True)
