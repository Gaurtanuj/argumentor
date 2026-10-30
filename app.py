import csv
import os
import sqlite3
from functools import wraps

import joblib
import numpy as np
from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)

import train as trainer

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
DB_PATH = os.path.join(trainer.BASE, "argumentor.db")

LABELS = ["No Fallacy", "Ad Hominem", "Straw Man", "Slippery Slope",
          "False Dilemma", "Hasty Generalization"]
EXPLANATIONS = {
    "No Fallacy": "No common fallacy detected. The reasoning looks sound.",
    "Ad Hominem": "Attacks the person instead of addressing the argument.",
    "Straw Man": "Misrepresents the opponent's position to make it easier to attack.",
    "Slippery Slope": "Claims one small step will lead to extreme consequences without evidence.",
    "False Dilemma": "Presents only two options when more exist.",
    "Hasty Generalization": "Draws a broad conclusion from too few examples.",
}

models = {}


# ---------- Database (SQLite) ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS arguments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            topic TEXT NOT NULL,
            text TEXT NOT NULL,
            predicted_fallacy TEXT NOT NULL,
            predicted_score REAL NOT NULL,
            corrected_fallacy TEXT,
            corrected_score REAL,
            reviewed INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")


# ---------- ML ----------
def load_models():
    if not os.path.exists(os.path.join(trainer.MODEL_DIR, "classifier.joblib")):
        trainer.train()
    for key, name in [("vec", "vectorizer"), ("clf", "classifier"), ("reg", "regressor")]:
        models[key] = joblib.load(os.path.join(trainer.MODEL_DIR, name + ".joblib"))


def predict(text):
    X = models["vec"].transform([text.strip()])
    label = models["clf"].predict(X)[0]
    confidence = float(models["clf"].predict_proba(X).max())
    score = float(np.clip(models["reg"].predict(X)[0], 0, 100))
    return label, confidence, round(score, 1)


# ---------- Auth ----------
def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin"))
        return view(*args, **kwargs)
    return wrapper


# ---------- Pages ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html", rows=leaderboard_rows())


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
        else:
            flash("Wrong password.")
        return redirect(url_for("admin"))
    if not session.get("admin"):
        return render_template("admin.html", logged_in=False)
    db = get_db()
    pending = db.execute("SELECT * FROM arguments WHERE reviewed = 0 ORDER BY id DESC").fetchall()
    stats = db.execute("SELECT COUNT(*) total, SUM(reviewed) reviewed FROM arguments").fetchone()
    return render_template("admin.html", logged_in=True, pending=pending,
                           stats=stats, labels=LABELS, metrics=last_metrics())


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------- REST endpoints ----------
@app.route("/api/submit", methods=["POST"])
def api_submit():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    topic = (data.get("topic") or "").strip()
    text = (data.get("text") or "").strip()
    if not name or not topic or len(text) < 20:
        return jsonify(error="Enter name, topic and an argument of at least 20 characters."), 400

    label, confidence, score = predict(text)
    db = get_db()
    db.execute("""INSERT INTO arguments (user_name, topic, text, predicted_fallacy, predicted_score)
                  VALUES (?, ?, ?, ?, ?)""", (name, topic, text, label, score))
    db.commit()
    return jsonify(fallacy=label, confidence=round(confidence * 100, 1),
                   score=score, explanation=EXPLANATIONS[label])


def leaderboard_rows():
    return get_db().execute("""
        SELECT user_name, COUNT(*) AS arguments,
               ROUND(AVG(COALESCE(corrected_score, predicted_score)), 1) AS avg_score,
               ROUND(MAX(COALESCE(corrected_score, predicted_score)), 1) AS best_score
        FROM arguments GROUP BY user_name ORDER BY avg_score DESC, arguments DESC
        LIMIT 20""").fetchall()


@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify([dict(r) for r in leaderboard_rows()])


@app.route("/api/correct/<int:arg_id>", methods=["POST"])
@admin_required
def api_correct(arg_id):
    fallacy = request.form.get("fallacy")
    try:
        score = float(request.form.get("score"))
    except (TypeError, ValueError):
        score = None
    if fallacy not in LABELS or score is None or not 0 <= score <= 100:
        flash("Invalid label or score.")
        return redirect(url_for("admin"))

    db = get_db()
    row = db.execute("SELECT * FROM arguments WHERE id = ? AND reviewed = 0", (arg_id,)).fetchone()
    if row is None:
        flash("Argument not found or already reviewed.")
        return redirect(url_for("admin"))

    db.execute("UPDATE arguments SET corrected_fallacy=?, corrected_score=?, reviewed=1 WHERE id=?",
               (fallacy, score, arg_id))
    db.commit()
    with open(trainer.COLLECTED_CSV, "a", newline="", encoding="utf-8") as f:  # dataset grows
        csv.writer(f).writerow([row["text"], fallacy, score])
    flash("Label saved and added to collected_data.csv.")
    return redirect(url_for("admin"))


@app.route("/api/retrain", methods=["POST"])
@admin_required
def api_retrain():
    r = trainer.train()
    load_models()
    flash(f"Retrained on {r['size']} arguments - accuracy {r['accuracy']:.2f}, "
          f"F1 {r['f1']:.2f}, score MAE {r['mae']:.1f}.")
    return redirect(url_for("admin"))


def last_metrics():
    if not os.path.exists(trainer.METRICS_CSV):
        return []
    with open(trainer.METRICS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))[-5:][::-1]


init_db()
load_models()

if __name__ == "__main__":
    app.run(debug=True)
