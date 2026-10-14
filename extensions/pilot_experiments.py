"""Pilot experiments for three extensions listed in docs/EXTENSIONS.md.

1. Gradient boosting (HistGradientBoosting) instead of the random forest, on Paris.
2. Predicting log(price) with a regressor and cutting the prediction into quartiles, on Paris.
3. Leave-one-city-out with gradient boosting, and a few-shot curve: how many labelled
   listings of the target city (Vienna) are needed to close the domain gap?

Run from the repository root:  python extensions/pilot_experiments.py   (~10 min on a laptop)
Outputs: results/pilot_*.csv and figures/extensions_pilot/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

DATA = Path("data")
RESULTS = Path("results")
FIGS = Path("figures/extensions_pilot")
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CITIES = ["Paris", "Rome", "Madrid", "Vienna"]
DROP = ["target_class", "price_clean", "city_label"]
CONT_COLS = ["accommodates", "bedrooms", "beds", "minimum_nights_avg_ntm", "number_of_reviews", "dist_to_center",
             "bathrooms_qty", "standing_thermal", "score_chef_kitchen", "score_coffee", "score_wellness",
             "score_work", "score_view", "score_secure", "neigh_popularity"]
# Same random forest as the final model of notebook 2
RF_PARAMS = dict(max_depth=20, max_features="sqrt", min_samples_leaf=2, min_samples_split=5, n_estimators=300)


def standardize_per_city(df):
    out = df.copy()
    grp = out.groupby("city_label")[CONT_COLS]
    out[CONT_COLS] = (out[CONT_COLS] - grp.transform("mean")) / grp.transform("std").replace(0, 1)
    return out


def rerank(proba):
    ranks = pd.Series(proba @ np.arange(proba.shape[1])).rank(method="first")
    return pd.qcut(ranks, 4, labels=[0, 1, 2, 3]).astype(int).values


def f1(y, p):
    return f1_score(y, p, average="macro")


cities = pd.concat([pd.read_csv(DATA / f"{c.lower()}_dataset_final_ready.csv.gz") for c in CITIES], ignore_index=True)
for col in cities.columns:
    if col.startswith(("pg_", "rt_")):
        cities[col] = cities[col].astype(int)

# ------------------------------------------------------------------ 1-2. Paris: model families and target choice
paris = cities[cities["city_label"] == "Paris"]
X = paris.drop(columns=DROP + [c for c in paris.columns if c.startswith("pg_")])
y, price = paris["target_class"].values, paris["price_clean"].values
X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(X, y, price, test_size=0.2, stratify=y, random_state=42)

rows = []
rf = RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
rows.append({"Model": "Random forest (classes)", "F1-macro": f1(y_te, rf.predict(X_te))})

hgb = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, early_stopping=True, random_state=42).fit(X_tr, y_tr)
rows.append({"Model": "HistGradientBoosting (classes)", "F1-macro": f1(y_te, hgb.predict(X_te))})

# Regress log(price), then cut the predictions with the quartile thresholds of the training prices
reg = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, early_stopping=True, random_state=42)
reg.fit(X_tr, np.log(p_tr))
thresholds = np.quantile(np.log(p_tr), [0.25, 0.5, 0.75])
rows.append({"Model": "HistGradientBoosting (log-price regression -> quartiles)",
             "F1-macro": f1(y_te, np.searchsorted(thresholds, reg.predict(X_te), side="left"))})
paris_res = pd.DataFrame(rows)
paris_res.to_csv(RESULTS / "pilot_paris_models.csv", index=False)
print(paris_res.round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------ 3a. LODO with gradient boosting
data = standardize_per_city(cities)
rows = []
for test_city in CITIES:
    tr, te = data[data["city_label"] != test_city], data[data["city_label"] == test_city]
    w = compute_sample_weight("balanced", tr["city_label"])
    for name, model in [("Random forest", RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1)),
                        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05,
                                                                               early_stopping=True, random_state=42))]:
        model.fit(tr.drop(columns=DROP), tr["target_class"], sample_weight=w)
        proba = model.predict_proba(te.drop(columns=DROP))
        rows.append({"Model": name, "City": test_city,
                     "F1 raw": f1(te["target_class"], proba.argmax(axis=1)),
                     "F1 reranked": f1(te["target_class"], rerank(proba))})
        print(rows[-1], flush=True)
lodo = pd.DataFrame(rows)
lodo.to_csv(RESULTS / "pilot_lodo_gradient_boosting.csv", index=False)
print(lodo.groupby("Model")[["F1 raw", "F1 reranked"]].mean().round(4), flush=True)

# ------------------------------------------------------------------ 3b. Few-shot adaptation curve on Vienna
source = data[data["city_label"] != "Vienna"]
vienna = data[data["city_label"] == "Vienna"]
v_pool, v_test = train_test_split(vienna, test_size=0.4, stratify=vienna["target_class"], random_state=42)
rows = []
for n in [0, 50, 200, 500, 1000, 2000, 4000, len(v_pool)]:
    v_lab = v_pool.sample(n=n, random_state=42) if n else v_pool.iloc[:0]
    # (a) source cities + n labelled Vienna listings, Vienna weighted like one more city
    train = pd.concat([source, v_lab])
    w = compute_sample_weight("balanced", train["city_label"])
    m = RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1)
    m.fit(train.drop(columns=DROP), train["target_class"], sample_weight=w)
    rows.append({"n_vienna_labels": n, "Training data": "3 source cities + n Vienna listings",
                 "F1 reranked": f1(v_test["target_class"], rerank(m.predict_proba(v_test.drop(columns=DROP))))})
    # (b) Vienna only, n labelled listings
    if n >= 50:
        m = RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1)
        m.fit(v_lab.drop(columns=DROP), v_lab["target_class"])
        rows.append({"n_vienna_labels": n, "Training data": "n Vienna listings only",
                     "F1 reranked": f1(v_test["target_class"], rerank(m.predict_proba(v_test.drop(columns=DROP))))})
    print(rows[-1], flush=True)
fewshot = pd.DataFrame(rows)
fewshot.to_csv(RESULTS / "pilot_fewshot_vienna.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
for name, g in fewshot.groupby("Training data"):
    g = g[g["n_vienna_labels"] > 0]
    ax.plot(g["n_vienna_labels"], g["F1 reranked"], marker="o", label=name)
zero = fewshot.loc[fewshot["n_vienna_labels"] == 0, "F1 reranked"].iloc[0]
ax.axhline(zero, color="grey", linestyle="--", label="0 Vienna labels (pure domain generalisation)")
ax.set_xscale("log")
ax.set_xlabel("Number of labelled Vienna listings used for training (log scale)")
ax.set_ylabel("F1-macro on held-out Vienna listings (reranked)")
ax.set_title("Few-shot adaptation to Vienna")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fewshot_vienna.png", dpi=150)
print(fewshot.round(4).to_string(index=False))
