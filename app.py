from flask import Flask, render_template, request, redirect, url_for
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        match_url = request.form.get("match_url")
        return redirect(url_for("predict", match_url=match_url))
    return render_template("index.html")

@app.route("/predict")
def predict():
    match_url = request.args.get("match_url")
    if not match_url:
        return redirect(url_for("index"))
    # Placeholder for predictions
    best_xi = [
        {"player": "Phil Salt", "team": "RCB", "role": "WK-BAT", "pred": 50.0},
        # … fill in 11 entries
    ]
    backups = [
        {"player": "Romario Shepherd", "team": "RCB", "role": "AR", "pred": 22.4},
        # … next 5
    ]
    all_players = []  # populate with player stats
    return render_template("results.html",
                           url=match_url,
                           best_xi=best_xi,
                           backups=backups,
                           all_players=all_players)

if __name__ == "__main__":
    app.run(debug=True)
