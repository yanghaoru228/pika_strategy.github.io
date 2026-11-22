"""
AdaBoost on Pokémon TCG cards
=============================
Question: from a card's printed attributes alone (rarity, era, stage, HP, attacks, ...),
can we predict whether it sells for $5 or more on TCGplayer?

Inputs  : data/modern_pkmn_cards_feb2025.csv, data/vintage_pkmn_cards_feb2025.csv
          (data/*_mar2025.csv is used once, at the end, as a "next month" check)
Outputs : out/*.csv (sample, cleaned, encoded, train, test), out/fig/*.png, out/results.json

Run:  python adaboost_pokemon.py
"""
import json
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42
THRESHOLD = 5.00          # "high value" = best TCGplayer market price >= $5
DATA = Path("data")
OUT = Path("out"); FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

# ---- chart styling (one palette across every figure) -------------------------------
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})
BLUES = LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#9cc3ef", BLUE, "#0f3f7a"])


def table_png(df, path, title, col_width=None, max_chars=22, fontsize=9):
    """Render a small DataFrame as an image (used for the Data Prep screenshots)."""
    import textwrap
    show = df.map(lambda v: "NaN" if pd.isna(v) else str(v)[:max_chars])
    headers = ["\n".join(textwrap.wrap(str(c), 14, break_long_words=False)) or str(c) for c in show.columns]
    # width of each column ~ longest line in it
    widths = [max([len(l) for l in h.split("\n")] + [len(v) for v in show[c]]) for h, c in zip(headers, show.columns)]
    units = np.array(widths, float) + 2
    fig_w = units.sum() * 0.085 + 0.4
    hdr_lines = max(h.count("\n") + 1 for h in headers)
    fig_h = 0.30 * (len(show) + hdr_lines) + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.01, 0.01, 0.98, 1 - 0.5 / fig_h]); ax.axis("off")
    t = ax.table(cellText=show.values, colLabels=headers, cellLoc="left", bbox=[0, 0, 1, 1],
                 colWidths=list(units / units.sum()))
    t.auto_set_font_size(False); t.set_fontsize(fontsize)
    n = len(show) + hdr_lines
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#eef3fb"); cell.set_text_props(weight="bold", color=INK)
            cell.set_height(hdr_lines / n)
        else:
            cell.set_height(1 / n)
    fig.text(0.01, 1 - 0.12 / fig_h, title, ha="left", va="top", fontsize=12, weight="bold", color=INK)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08); plt.close(fig)


# =====================================================================================
# 1. LOAD + LABEL
# =====================================================================================
def load(month):
    m = pd.read_csv(DATA / f"modern_pkmn_cards_{month}2025.csv").assign(Era="Modern")
    v = pd.read_csv(DATA / f"vintage_pkmn_cards_{month}2025.csv").assign(Era="Vintage")
    d = pd.concat([m, v], ignore_index=True)
    price_cols = [c for c in d.columns if "Market Price" in c]
    d["Best Market Price"] = d[price_cols].max(axis=1)     # most valuable printing of the card
    return d.dropna(subset=["Best Market Price"]).drop_duplicates("ID").reset_index(drop=True)

raw = load("feb")
raw["High Value"] = np.where(raw["Best Market Price"] >= THRESHOLD, "Yes", "No")

# The sample of raw data shown on the site (unlabelled-for-modelling raw columns + label)
raw.sample(200, random_state=SEED).to_csv(OUT / "raw_sample.csv", index=False)

# =====================================================================================
# 2. FEATURE ENGINEERING  (only printed attributes — no price column ever enters X)
# =====================================================================================
def max_damage(attacks):
    if pd.isna(attacks):
        return 0
    nums = [int(n) for n in re.findall(r"\((\d+)", attacks)]
    return max(nums) if nums else 0

def n_attacks(attacks):
    return 0 if pd.isna(attacks) else len(re.findall(r"\(", attacks))

KEEP_SUBTYPES = ["Basic", "Stage 1", "Stage 2", "V", "VMAX", "VSTAR", "ex", "Tera",
                 "Supporter", "Item", "Pokémon Tool", "Stadium", "Special", "ACE SPEC",
                 "Radiant", "V-UNION", "Ancient", "Future"]

def clean(d):
    c = pd.DataFrame({"ID": d["ID"], "Name": d["Name"]})
    c["Era"] = d["Era"]
    c["Release Year"] = pd.to_datetime(d["Release Date"]).dt.year
    c["Supertype"] = d["Supertype"].str.replace("é", "e")
    c["Rarity"] = d["Rarity"].fillna("Unknown")
    c["Type"] = d["Types"].str.split(",").str[0].fillna("None")
    c["Weakness"] = d["Weaknesses"].str.split(" ").str[0].fillna("None")
    c["HP"] = pd.to_numeric(d["HP"], errors="coerce").fillna(0).astype(int)
    c["Retreat Cost"] = pd.to_numeric(d["Retreat Cost"], errors="coerce").fillna(0).astype(int)
    c["Num Attacks"] = d["Attacks"].apply(n_attacks)
    c["Max Damage"] = d["Attacks"].apply(max_damage)
    c["Subtypes"] = d["Subtypes"].fillna("")
    c["Best Market Price"] = d["Best Market Price"]      # kept for reference, NOT a feature
    c["High Value"] = (d["Best Market Price"] >= THRESHOLD).map({True: "Yes", False: "No"})
    return c

CATEGORICAL = ["Era", "Supertype", "Rarity", "Type", "Weakness"]
NUMERIC = ["Release Year", "HP", "Retreat Cost", "Num Attacks", "Max Damage"]

def encode(c, columns=None):
    """One-hot encode categorical columns; multi-hot encode the comma-separated Subtypes."""
    X = c[NUMERIC].copy()
    oh = pd.get_dummies(c[CATEGORICAL], prefix=CATEGORICAL, dtype=int)
    subs = c["Subtypes"].str.split(r",\s*")
    for s in KEEP_SUBTYPES:
        X[f"Subtype_{s}"] = subs.apply(lambda lst: int(s in lst))
    X = pd.concat([X, oh], axis=1)
    if columns is not None:                     # align a later month to the training columns
        X = X.reindex(columns=columns, fill_value=0)
    return X

cleaned = clean(raw)
X_all = encode(cleaned)
y_all = (cleaned["High Value"] == "Yes").astype(int)
cleaned.to_csv(OUT / "cleaned_feb2025.csv", index=False)

# =====================================================================================
# 3. TRAIN / TEST SPLIT  (stratified, disjoint)
# =====================================================================================
idx_train, idx_test = train_test_split(cleaned.index, test_size=0.20, stratify=y_all,
                                       random_state=SEED)
X_train, X_test = X_all.loc[idx_train], X_all.loc[idx_test]
y_train, y_test = y_all.loc[idx_train], y_all.loc[idx_test]
assert set(idx_train).isdisjoint(idx_test)
assert set(cleaned.loc[idx_train, "ID"]).isdisjoint(cleaned.loc[idx_test, "ID"])

encoded = pd.concat([cleaned[["ID"]], X_all, y_all.rename("High Value (1=Yes)")], axis=1)
encoded.to_csv(OUT / "encoded_feb2025.csv", index=False)
encoded.loc[idx_train].to_csv(OUT / "train_set.csv", index=False)
encoded.loc[idx_test].to_csv(OUT / "test_set.csv", index=False)

# ---- Data Prep images ----------------------------------------------------------------
raw_cols = ["ID", "Name", "Supertype", "Subtypes", "HP", "Types", "Rarity", "Release Date",
            "TCG Market Price USD (Holofoil)", "TCG Market Price USD (Normal)"]
table_png(raw.sample(8, random_state=7)[raw_cols].rename(columns=lambda c: c.replace("TCG Market Price USD", "Market $")),
          FIG / "dp_raw_sample.png",
          "Raw data (sample of the Feb 2025 pull, 10 of 29 columns)", col_width=1.45)
show_clean = ["ID", "Era", "Release Year", "Supertype", "Rarity", "Type", "HP",
              "Max Damage", "Subtypes", "Best Market Price", "High Value"]
table_png(cleaned.sample(8, random_state=7)[show_clean], FIG / "dp_cleaned.png",
          "Cleaned + labelled (High Value = best market price ≥ $5)", col_width=1.3)
ex = cleaned.sample(6, random_state=3)
table_png(ex[["ID", "Era", "Supertype", "Type"]], FIG / "dp_onehot_before.png",
          "Before: categorical text", col_width=1.4, fontsize=9)
oh_cols = ["Era_Modern", "Era_Vintage", "Supertype_Pokemon", "Supertype_Trainer",
           "Type_Fire", "Type_Water", "Type_Psychic", "Type_None"]
table_png(pd.concat([ex[["ID"]], X_all.loc[ex.index, oh_cols]], axis=1),
          FIG / "dp_onehot_after.png", "After: one-hot 0/1 columns (subset shown)",
          col_width=1.25, fontsize=8.5)
tr_show = ["ID", "HP", "Max Damage", "Era_Vintage", "Rarity_Common", "Subtype_V",
           "High Value (1=Yes)"]
table_png(encoded.loc[idx_train].head(8)[tr_show], FIG / "dp_train.png",
          f"Training set — {len(idx_train):,} rows × {X_all.shape[1]} features + label",
          col_width=1.4, fontsize=8.5)
table_png(encoded.loc[idx_test].head(8)[tr_show], FIG / "dp_test.png",
          f"Testing set — {len(idx_test):,} rows (never seen during training)",
          col_width=1.4, fontsize=8.5)

# =====================================================================================
# 4. MODEL — AdaBoost with decision stumps, tuned by 5-fold CV on the TRAINING set only
# =====================================================================================
stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = {"n_estimators": [25, 50, 100, 200, 400, 800], "learning_rate": [0.05, 0.1, 0.25, 0.5, 1.0]}
search = GridSearchCV(AdaBoostClassifier(estimator=stump, random_state=SEED), grid,
                      scoring="accuracy", cv=cv, n_jobs=-1)
search.fit(X_train, y_train)
ada = search.best_estimator_                      # refit on the full training set
best = search.best_params_
print("Best AdaBoost params:", best, "CV acc:", round(search.best_score_, 4))

# CV heatmap: n_estimators x learning_rate
cvres = pd.DataFrame(search.cv_results_)
heat = cvres.pivot(index="param_learning_rate", columns="param_n_estimators",
                   values="mean_test_score")
fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.imshow(heat.values, cmap=BLUES, aspect="auto")
ax.set_xticks(range(heat.shape[1]), heat.columns); ax.set_yticks(range(heat.shape[0]), heat.index)
ax.set_xlabel("n_estimators (boosting rounds)"); ax.set_ylabel("learning_rate"); ax.grid(False)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                color="white" if v > heat.values.mean() else INK)
ax.set_title("5-fold CV accuracy on the training set", loc="left")
fig.colorbar(im, ax=ax, shrink=0.85).outline.set_visible(False)
fig.savefig(FIG / "res_tuning_heatmap.png"); plt.close(fig)

# Validation curve: accuracy as rounds are added (hold 20% of TRAIN out as validation)
X_sub, X_val, y_sub, y_val = train_test_split(X_train, y_train, test_size=0.20,
                                              stratify=y_train, random_state=SEED)
curves = {}
MAX_R = 800
for lr, col in [(0.1, GRAY), (0.5, AQUA), (1.0, BLUE)]:
    m = AdaBoostClassifier(estimator=stump, n_estimators=MAX_R, learning_rate=lr,
                           random_state=SEED).fit(X_sub, y_sub)
    curves[lr] = [accuracy_score(y_val, p) for p in m.staged_predict(X_val)]
fig, ax = plt.subplots(figsize=(8, 4.3))
for lr, col in [(0.1, GRAY), (0.5, AQUA), (1.0, BLUE)]:
    acc = curves[lr]
    ax.plot(range(1, len(acc) + 1), acc, color=col, lw=2, label=f"learning_rate = {lr}")
    ax.annotate(f"lr {lr}: {acc[-1]:.3f}", (len(acc), acc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK2)
base = 1 - y_val.mean()
ax.axhline(base, color=ORANGE, lw=1.5, ls="--")
ax.text(5, base - 0.004, f"always guess 'No' = {base:.3f}", color=INK2, fontsize=9, va="top")
ax.set_xlabel("boosting rounds (stumps added)"); ax.set_ylabel("validation accuracy")
ax.set_xlim(0, MAX_R * 1.13); ax.set_ylim(base - 0.012, None); ax.legend(frameon=False, loc="lower right")
ax.set_title("Validation accuracy as stumps are added", loc="left")
fig.savefig(FIG / "res_staged_accuracy.png"); plt.close(fig)

# =====================================================================================
# 5. DECISION TREE (Part 3 comparison) — same features, same split, tuned the same way
# =====================================================================================
tree_search = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                           {"max_depth": [2, 3, 4, 5, 6, 8, 10, 12, None],
                            "min_samples_leaf": [1, 5, 10, 20]},
                           scoring="accuracy", cv=cv, n_jobs=-1).fit(X_train, y_train)
tree = tree_search.best_estimator_
print("Best tree params:", tree_search.best_params_)

# =====================================================================================
# 6. EVALUATE ON THE UNTOUCHED TEST SET
# =====================================================================================
def metrics(model):
    p = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, p), "balanced_accuracy": balanced_accuracy_score(y_test, p),
            "precision_yes": precision_score(y_test, p), "recall_yes": recall_score(y_test, p),
            "f1_yes": f1_score(y_test, p), "confusion": confusion_matrix(y_test, p).tolist()}, p

m_ada, p_ada = metrics(ada)
m_tree, p_tree = metrics(tree)
baseline = 1 - y_test.mean()
print(classification_report(y_test, p_ada, target_names=["No (<$5)", "Yes (≥$5)"]))

def plot_cm(cm, title, path, color_map):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=color_map); ax.grid(False)
    labels = ["No (< $5)", "Yes (≥ $5)"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.0%} of row)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_title(title, loc="left"); fig.savefig(path); plt.close(fig)

plot_cm(m_ada["confusion"], f"AdaBoost — test accuracy {m_ada['accuracy']:.1%}",
        FIG / "res_cm_adaboost.png", BLUES)
plot_cm(m_tree["confusion"], f"Decision tree — test accuracy {m_tree['accuracy']:.1%}",
        FIG / "res_cm_tree.png",
        LinearSegmentedColormap.from_list("o", ["#fdf3ee", "#f5b393", ORANGE, "#8a3510"]))

# Comparison bars
names = ["Accuracy", "Balanced acc.", "Precision (Yes)", "Recall (Yes)", "F1 (Yes)"]
keys = ["accuracy", "balanced_accuracy", "precision_yes", "recall_yes", "f1_yes"]
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.2))
b1 = ax.bar(x - w / 2 - 0.01, [m_ada[k] for k in keys], w, color=BLUE, label="AdaBoost")
b2 = ax.bar(x + w / 2 + 0.01, [m_tree[k] for k in keys], w, color=ORANGE, label="Decision tree")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                ha="center", fontsize=8.5, color=INK2)
ax.axhline(baseline, color=GRAY, ls="--", lw=1.2)
ax.text(len(names) - 0.5, baseline + 0.012, f"majority-class accuracy {baseline:.3f}",
        ha="right", fontsize=8.5, color=INK2)
ax.set_xticks(x, names); ax.set_ylim(0, 1.08); ax.legend(frameon=False, ncol=2, loc="upper center",
                                                       bbox_to_anchor=(0.5, 1.12))
ax.set_title("Test-set comparison", loc="left", pad=24); ax.grid(axis="x", visible=False)
fig.savefig(FIG / "res_compare.png"); plt.close(fig)

# What the stumps split on: total alpha (stump weight) per feature
alpha_by_feat = {}
for est, a in zip(ada.estimators_, ada.estimator_weights_[:len(ada.estimators_)]):
    f = X_train.columns[est.tree_.feature[0]]
    alpha_by_feat[f] = alpha_by_feat.get(f, 0) + a
top = pd.Series(alpha_by_feat).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(top.index, top.values, color=BLUE, height=0.6)
ax.set_xlabel("sum of α over the stumps that split on this feature")
ax.set_title("Which features the stumps voted with", loc="left"); ax.grid(axis="y", visible=False)
fig.savefig(FIG / "res_feature_alpha.png"); plt.close(fig)

# First few stumps, for the write-up
first_stumps = []
for k, (est, a, e) in enumerate(zip(ada.estimators_[:8], ada.estimator_weights_, ada.estimator_errors_)):
    t = est.tree_
    first_stumps.append({"round": k + 1, "feature": X_train.columns[t.feature[0]],
                         "threshold": round(float(t.threshold[0]), 2),
                         "weighted_error": round(float(e), 4), "alpha": round(float(a), 4)})

# =====================================================================================
# 7. "NEXT MONTH" CHECK — score the March 2025 pull with the February model
# =====================================================================================
mar = clean(load("mar"))
X_mar = encode(mar, columns=X_all.columns)
y_mar = (mar["High Value"] == "Yes").astype(int)
mar_new = ~mar["ID"].isin(cleaned.loc[idx_train, "ID"])   # cards the model never trained on
mar_acc_ada = accuracy_score(y_mar[mar_new], ada.predict(X_mar[mar_new]))
mar_acc_tree = accuracy_score(y_mar[mar_new], tree.predict(X_mar[mar_new]))

# Error analysis by era
test_era = cleaned.loc[idx_test, "Era"]
by_era = {e: {"n": int((test_era == e).sum()),
              "ada": accuracy_score(y_test[test_era == e], p_ada[test_era.values == e]),
              "tree": accuracy_score(y_test[test_era == e], p_tree[test_era.values == e]),
              "base": float(1 - y_test[test_era == e].mean()) if y_test[test_era == e].mean() < .5
                      else float(y_test[test_era == e].mean())}
          for e in ["Modern", "Vintage"]}

results = {
    "n_rows": len(cleaned), "n_features": X_all.shape[1], "n_train": len(idx_train),
    "n_test": len(idx_test), "pos_rate": float(y_all.mean()), "threshold": THRESHOLD,
    "ada_best_params": best, "ada_cv_acc": search.best_score_,
    "tree_best_params": {k: (v if v is not None else "None") for k, v in tree_search.best_params_.items()},
    "tree_cv_acc": tree_search.best_score_, "baseline_acc": baseline,
    "adaboost": m_ada, "tree": m_tree, "first_stumps": first_stumps,
    "staged_final": {str(k): v[-1] for k, v in curves.items()},
    "staged_at": {str(k): {str(r): v[r - 1] for r in [1, 10, 25, 50, 100, 200, 400, 800]}
                  for k, v in curves.items()},
    "march_check": {"n": int(mar_new.sum()), "ada": mar_acc_ada, "tree": mar_acc_tree},
    "by_era": by_era, "top_alpha_features": top.sort_values(ascending=False).round(3).to_dict(),
}
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({k: results[k] for k in ["ada_best_params", "tree_best_params", "baseline_acc",
                                          "march_check", "by_era"]}, indent=1, default=float))
print("AdaBoost:", {k: round(v, 4) for k, v in m_ada.items() if k != "confusion"}, m_ada["confusion"])
print("Tree    :", {k: round(v, 4) for k, v in m_tree.items() if k != "confusion"}, m_tree["confusion"])
