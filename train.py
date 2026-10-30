"""ArguMentor - model training / retraining (Synopsis Section 5.4).

Run:  python train.py
"""
import csv
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
SEED_CSV = os.path.join(BASE, "seed_data.csv")
COLLECTED_CSV = os.path.join(BASE, "collected_data.csv")
METRICS_CSV = os.path.join(BASE, "metrics.csv")
MODEL_DIR = os.path.join(BASE, "models")


def load_data():
    """Step 1: load and merge seed_data.csv and collected_data.csv into D."""
    frames = [pd.read_csv(SEED_CSV)]
    if os.path.exists(COLLECTED_CSV):
        frames.append(pd.read_csv(COLLECTED_CSV))
    data = pd.concat(frames, ignore_index=True).dropna()
    data["text"] = data["text"].astype(str).str.strip()      # pre-processing
    return data.drop_duplicates(subset="text").reset_index(drop=True)


def log_metrics(n, acc, f1, mae):
    """Append evaluation results so dataset-growth impact can be studied."""
    new_file = not os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "dataset_size", "accuracy", "macro_f1", "score_mae"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), n,
                    round(acc, 3), round(f1, 3), round(mae, 2)])


def train():
    data = load_data()
    texts, y_label, y_score = data["text"], data["fallacy"], data["score"].astype(float)

    # Step 2: TF-IDF on unigrams + bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)

    # Step 3: train / holdout split
    stratify = y_label if y_label.value_counts().min() >= 2 else None
    X_tr, X_te, yl_tr, yl_te, ys_tr, ys_te = train_test_split(
        X, y_label, y_score, test_size=0.2, random_state=42, stratify=stratify)

    # Steps 4-5: Logistic Regression (fallacy) and Ridge Regression (score)
    clf = LogisticRegression(C=10, max_iter=1000).fit(X_tr, yl_tr)
    reg = Ridge(alpha=1.0).fit(X_tr, ys_tr)

    # Step 6: evaluate on holdout
    acc = accuracy_score(yl_te, clf.predict(X_te))
    f1 = f1_score(yl_te, clf.predict(X_te), average="macro")
    mae = mean_absolute_error(ys_te, np.clip(reg.predict(X_te), 0, 100))

    # Step 7: refit on the full dataset
    clf.fit(X, y_label)
    reg.fit(X, y_score)

    # Step 8: save for the web application
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "regressor.joblib"))

    log_metrics(len(data), acc, f1, mae)
    return {"size": len(data), "accuracy": acc, "f1": f1, "mae": mae}


if __name__ == "__main__":
    r = train()
    print(f"Dataset size : {r['size']}")
    print(f"Accuracy     : {r['accuracy']:.3f}")
    print(f"Macro F1     : {r['f1']:.3f}")
    print(f"Score MAE    : {r['mae']:.2f}")
    print("Models saved in ./models")
