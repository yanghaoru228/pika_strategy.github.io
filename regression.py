"""
Regression tab - Pokemon card analysis
Linear regression (predict log10 market price) and logistic regression
(predict whether a card is "valuable", market price >= $5), plus a Gaussian
Naive Bayes model on the same split for comparison.

Input : the February 2025 snapshot of modern + vintage cards
        (modern_pkmn_cards_feb2025.csv, vintage_pkmn_cards_feb2025.csv)
Output: data/  prepared sample, training set, testing set
        img/   all figures used on the Regression tab
        results.json  every metric quoted on the page

Run:  python regression_pokemon.py --data-dir <folder with the csvs> --out <site folder>
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

SEED = 42
VALUE_THRESHOLD = 5.00          # dollars: label 1 = "valuable"
FEATURES = ["HP", "Retreat_Cost", "Num_Attacks", "Max_Damage",
            "Card_Age_Years", "Rarity_Tier"]

# Rarity names -> an ordinal tier (0 = most common ... 5 = chase cards)
RARITY_TIER = {
    "Common": 0,
    "Uncommon": 1,
    "Rare": 2,
    "Promo": 2,
    "Rare Holo": 3,
    "Rare Holo V": 4, "Rare Holo VMAX": 4, "Rare Holo VSTAR": 4,
    "Double Rare": 4, "Ultra Rare": 4, "Rare Ultra": 4, "Radiant Rare": 4,
    "Amazing Rare": 4, "Shiny Rare": 4, "Rare Shiny": 4, "ACE SPEC Rare": 4,
    "Trainer Gallery Rare Holo": 4,
    "Illustration Rare": 5, "Special Illustration Rare": 5, "Hyper Rare": 5,
    "Rare Secret": 5, "Rare Rainbow": 5, "Shiny Ultra Rare": 5,
}

# Chart palette (validated categorical slots)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


# ----------------------------------------------------------------------------
# 1. Data prep
# ----------------------------------------------------------------------------
def max_damage(attacks):
    """Largest printed damage number in the Attacks text, e.g. 'Absorb (10)' -> 10."""
    if not isinstance(attacks, str):
        return 0
    nums = [int(n) for n in re.findall(r"\((\d+)", attacks)]
    return max(nums) if nums else 0


def num_attacks(attacks):
    if not isinstance(attacks, str):
        return 0
    return len(re.findall(r"\(", attacks))


def load_and_prepare(data_dir):
    modern = pd.read_csv(os.path.join(data_dir, "modern_pkmn_cards_feb2025.csv"))
    vintage = pd.read_csv(os.path.join(data_dir, "vintage_pkmn_cards_feb2025.csv"))
    modern["Era"] = "Modern"
    vintage["Era"] = "Vintage"
    df = pd.concat([modern, vintage], ignore_index=True)
    raw_rows = len(df)

    # Pokemon cards only: Trainer/Energy cards have no HP or attacks
    df = df[df["Supertype"] == "Pokémon"].copy()

    # One market price per card: holofoil if the card has one, else normal,
    # else reverse holofoil (the version collectors usually mean)
    df["Market_Price"] = (df["TCG Market Price USD (Holofoil)"]
                          .fillna(df["TCG Market Price USD (Normal)"])
                          .fillna(df["TCG Market Price USD (Reverse Holofoil)"]))

    df["Rarity_Tier"] = df["Rarity"].map(RARITY_TIER)
    df["Retreat_Cost"] = df["Retreat Cost"]
    df["Num_Attacks"] = df["Attacks"].apply(num_attacks)
    df["Max_Damage"] = df["Attacks"].apply(max_damage)
    rel = pd.to_datetime(df["Release Date"], format="%Y/%m/%d")
    priced = pd.to_datetime(df["TCG Price Date"], format="%Y/%m/%d")
    df["Card_Age_Years"] = ((priced - rel).dt.days / 365.25).round(2)

    df = df.dropna(subset=["Market_Price", "HP", "Rarity_Tier", "Card_Age_Years"])
    df = df[df["Market_Price"] > 0]

    # Targets
    df["Log10_Price"] = np.log10(df["Market_Price"]).round(4)          # linear
    df["Valuable"] = (df["Market_Price"] >= VALUE_THRESHOLD).astype(int)  # logistic

    keep = ["ID", "Name", "Set Name", "Era", "Rarity", "Market_Price"] + FEATURES \
        + ["Log10_Price", "Valuable"]
    df = df[keep].reset_index(drop=True)
    df["HP"] = df["HP"].astype(int)
    df["Rarity_Tier"] = df["Rarity_Tier"].astype(int)
    return df, raw_rows


# ----------------------------------------------------------------------------
# 2. Figures
# ----------------------------------------------------------------------------
def table_image(frame, path, title, max_rows=8, col_width=None):
    shown = frame.head(max_rows).copy()
    for c in shown.columns:
        if shown[c].dtype.kind == "f":
            shown[c] = shown[c].map(lambda v: f"{v:.2f}")
        shown[c] = shown[c].astype(str).str.slice(0, 22)
    ncol = len(shown.columns)
    fig_w = col_width or max(8, 1.25 * ncol)
    fig, ax = plt.subplots(figsize=(fig_w, 0.3 * (len(shown) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=shown.values, colLabels=list(shown.columns),
                   bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#f1f0ec")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK2)
    tbl.auto_set_column_width(list(range(ncol)))
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=4)
    fig.savefig(path)
    plt.close(fig)


def fig_linear_vs_logistic(path):
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # Linear: numeric y
    x = rng.uniform(0, 5, 60)
    y = 0.8 + 0.55 * x + rng.normal(0, 0.35, 60)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(0, 5, 100)
    axes[0].scatter(x, y, s=28, color=BLUE, edgecolor="white", linewidth=1)
    axes[0].plot(xs, b0 + b1 * xs, color=INK, lw=2)
    axes[0].set_title("Linear regression: predict a number")
    axes[0].set_xlabel("Rarity tier (example)")
    axes[0].set_ylabel("log10(price)")
    axes[0].text(0.1, 3.7, "ŷ = b₀ + b₁x", fontsize=12, color=INK)
    # Logistic: 0/1 y
    x2 = rng.uniform(0, 5, 80)
    p = 1 / (1 + np.exp(-(2.2 * (x2 - 2.6))))
    y2 = (rng.uniform(size=80) < p).astype(int)
    jit = rng.uniform(-0.03, 0.03, 80)
    axes[1].scatter(x2, y2 + jit, s=28, c=np.where(y2 == 1, ORANGE, BLUE),
                    edgecolor="white", linewidth=1)
    axes[1].plot(xs, 1 / (1 + np.exp(-(2.2 * (xs - 2.6)))), color=INK, lw=2)
    axes[1].axhline(0.5, color=INK2, lw=1, ls="--")
    axes[1].set_yticks([0, 0.5, 1])
    axes[1].set_title("Logistic regression: predict a probability")
    axes[1].set_xlabel("Rarity tier (example)")
    axes[1].set_ylabel("P(valuable)")
    axes[1].text(0.1, 0.83, "p = σ(b₀ + b₁x)", fontsize=12, color=INK)
    axes[1].text(4.15, 0.07, "0 = not valuable", fontsize=9, color=BLUE)
    axes[1].text(0.05, 0.9 - 0.18, "", fontsize=9)
    axes[1].text(3.85, 0.9, "1 = valuable", fontsize=9, color=ORANGE)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sigmoid(path):
    z = np.linspace(-7, 7, 300)
    s = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.fill_between(z, 0, 1, where=z >= 0, color=ORANGE, alpha=0.08, lw=0)
    ax.fill_between(z, 0, 1, where=z < 0, color=BLUE, alpha=0.08, lw=0)
    ax.plot(z, s, color=INK, lw=2.2)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Linear score  z = b₀ + b₁x₁ + … + bₖxₖ   (log-odds)")
    ax.set_ylabel("σ(z) = probability of class 1")
    ax.set_title("The sigmoid squeezes any score into a probability between 0 and 1")
    ax.text(-6.6, 0.88, "σ(z) = 1 / (1 + e^(−z))", fontsize=13, color=INK)
    ax.text(-6.6, 0.32, "z < 0  →  p < 0.5  →  predict 0\n(not valuable)",
            fontsize=10, color=BLUE)
    ax.text(2.0, 0.32, "z > 0  →  p > 0.5  →  predict 1\n(valuable)",
            fontsize=10, color=ORANGE)
    for zz in (-2, 2):
        ss = 1 / (1 + np.exp(-zz))
        ax.scatter([zz], [ss], s=50, color=INK, zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(f"z = {zz} → p = {ss:.2f}", (zz, ss),
                    xytext=(zz + 0.35, ss - 0.08),
                    fontsize=9.5, color=INK2)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(path)
    plt.close(fig)


def fig_split(path, n_total, n_train, n_test):
    fig, ax = plt.subplots(figsize=(9, 2.1))
    ax.axis("off")
    tot = n_train + n_test
    ax.barh(0, n_train / tot, color=BLUE, height=0.5)
    ax.barh(0, n_test / tot, left=n_train / tot + 0.004, color=ORANGE, height=0.5)
    ax.text(n_train / tot / 2, 0, f"Training set  {n_train:,} cards (80%)",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.text(n_train / tot + n_test / tot / 2, 0, f"Testing\n{n_test:,} (20%)",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.01)
    ax.set_title(f"{n_total:,} labeled cards → shuffled, stratified on 'Valuable', split once (seed 42). "
                 "No card is in both.", fontsize=10.5, loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_pred_vs_actual(path, y_true, y_pred, r2, rmse):
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.scatter(y_true, y_pred, s=14, color=BLUE, alpha=0.45, edgecolor="none")
    lo, hi = -2.1, 3.6
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.6, ls="--")
    ax.text(2.0, 2.45, "perfect\nprediction", fontsize=9, color=INK2, rotation=0)
    ticks = [-2, -1, 0, 1, 2, 3]
    labels = ["$0.01", "$0.10", "$1", "$10", "$100", "$1,000"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual market price (log scale)")
    ax.set_ylabel("Predicted market price (log scale)")
    ax.set_title("Linear regression on the testing set")
    ax.text(-1.95, 3.15, f"R² = {r2:.3f}\nRMSE = {rmse:.3f} log10 units",
            fontsize=10.5, color=INK, va="top")
    fig.savefig(path)
    plt.close(fig)


def fig_residuals(path, y_true, y_pred):
    res = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(res, bins=50, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=INK, lw=1.5, ls="--")
    ax.set_xlabel("Residual (actual − predicted, log10 dollars)")
    ax.set_ylabel("Cards")
    ax.set_title("Residuals are centered on zero but have long tails")
    fig.savefig(path)
    plt.close(fig)


def fig_coefs(path, names, coefs, title, xlabel):
    order = np.argsort(np.abs(coefs))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = np.array(coefs)[order]
    ax.barh(np.array(names)[order], vals,
            color=[BLUE if v >= 0 else ORANGE for v in vals], height=0.55)
    ax.axvline(0, color=INK2, lw=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.004 if v >= 0 else -0.004) * max(abs(vals)) * 10, i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    pad = max(abs(vals)) * 0.25
    ax.set_xlim(min(0, vals.min()) - pad, max(0, vals.max()) + pad)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(path, cm, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(False)
    shades = np.array([["#cde0f6", "#fbd9c9"], ["#fbd9c9", "#cde0f6"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, i), 0.98, 0.98, color=shades[i][j]))
            tag = ["TN", "FP", "FN", "TP"][i * 2 + j]
            ax.text(j + 0.49, i + 0.42, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=INK)
            ax.text(j + 0.49, i + 0.7, tag, ha="center", va="center", fontsize=10, color=INK2)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.49, 1.49], ["Predicted:\nnot valuable", "Predicted:\nvaluable"])
    ax.set_yticks([0.49, 1.49], ["Actual:\nnot valuable", "Actual:\nvaluable"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def fig_roc(path, curves):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for (name, fpr, tpr, auc), col in zip(curves, [BLUE, ORANGE]):
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve, testing set")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig_compare(path, metrics):
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    lr = [metrics["logistic"][k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
    nb = [metrics["naive_bayes"][k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4))
    b1 = ax.bar(x - w / 2 - 0.01, lr, w, color=BLUE, label="Logistic regression")
    b2 = ax.bar(x + w / 2 + 0.01, nb, w, color=ORANGE, label="Gaussian Naive Bayes")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_title("Same features, same split: logistic regression vs Naive Bayes")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    data_out = os.path.join(args.out, "data")
    img = os.path.join(args.out, "img")
    os.makedirs(data_out, exist_ok=True)
    os.makedirs(img, exist_ok=True)

    df, raw_rows = load_and_prepare(args.data_dir)
    df.to_csv(os.path.join(data_out, "pokemon_regression_data.csv"), index=False)

    # ---- Train/test split: ONE split, reused by every model ----
    train, test = train_test_split(df, test_size=0.20, random_state=SEED,
                                   stratify=df["Valuable"])
    assert set(train["ID"]).isdisjoint(set(test["ID"])), "train/test overlap!"
    train.to_csv(os.path.join(data_out, "train_set.csv"), index=False)
    test.to_csv(os.path.join(data_out, "test_set.csv"), index=False)

    X_train, X_test = train[FEATURES].values, test[FEATURES].values

    # ---- Linear regression: numeric target log10(price) ----
    lin = LinearRegression().fit(X_train, train["Log10_Price"])
    yl_pred = lin.predict(X_test)
    yl_true = test["Log10_Price"].values
    rmse = float(np.sqrt(mean_squared_error(yl_true, yl_pred)))
    mae = float(mean_absolute_error(yl_true, yl_pred))
    r2 = float(r2_score(yl_true, yl_pred))
    dollar_mae = float(np.median(np.abs(10 ** yl_true - 10 ** yl_pred)))
    baseline_rmse = float(np.sqrt(np.mean((yl_true - train["Log10_Price"].mean()) ** 2)))

    # ---- Logistic regression: two-class label Valuable ----
    logit = LogisticRegression(max_iter=10000).fit(X_train, train["Valuable"])
    p_test = logit.predict_proba(X_test)[:, 1]
    yc_pred = (p_test >= 0.5).astype(int)          # threshold 0.5 -> class
    yc_true = test["Valuable"].values
    cm = confusion_matrix(yc_true, yc_pred)

    # ---- Gaussian Naive Bayes on the same split, for comparison ----
    nb = GaussianNB().fit(X_train, train["Valuable"])
    nb_p = nb.predict_proba(X_test)[:, 1]
    nb_pred = nb.predict(X_test)
    nb_cm = confusion_matrix(yc_true, nb_pred)

    def cls_metrics(pred, prob):
        return {"accuracy": float(accuracy_score(yc_true, pred)),
                "precision": float(precision_score(yc_true, pred)),
                "recall": float(recall_score(yc_true, pred)),
                "f1": float(f1_score(yc_true, pred)),
                "roc_auc": float(roc_auc_score(yc_true, prob))}

    results = {
        "raw_rows": raw_rows, "n_rows": len(df), "n_train": len(train), "n_test": len(test),
        "valuable_share": float(df["Valuable"].mean()),
        "train_valuable": int(train["Valuable"].sum()), "test_valuable": int(test["Valuable"].sum()),
        "features": FEATURES,
        "linear": {"rmse": rmse, "mae": mae, "r2": r2, "median_abs_dollar_error": dollar_mae,
                   "baseline_rmse": baseline_rmse, "intercept": float(lin.intercept_),
                   "coef": dict(zip(FEATURES, map(float, lin.coef_)))},
        "logistic": {**cls_metrics(yc_pred, p_test), "confusion": cm.tolist(),
                     "intercept": float(logit.intercept_[0]),
                     "coef": dict(zip(FEATURES, map(float, logit.coef_[0]))),
                     "odds_ratio": dict(zip(FEATURES, map(float, np.exp(logit.coef_[0]))))},
        "naive_bayes": {**cls_metrics(nb_pred, nb_p), "confusion": nb_cm.tolist()},
        "majority_baseline_accuracy": float(max(yc_true.mean(), 1 - yc_true.mean())),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Figures ----
    fig_linear_vs_logistic(os.path.join(img, "linear_vs_logistic.png"))
    fig_sigmoid(os.path.join(img, "sigmoid.png"))
    sample_cols = ["ID", "Name", "Era", "Rarity", "Market_Price"] + FEATURES + ["Log10_Price", "Valuable"]
    table_image(df[sample_cols].sample(8, random_state=1), os.path.join(img, "data_sample.png"),
                f"Prepared data sample ({len(df):,} rows × {len(sample_cols)} columns shown)")
    tcols = ["ID", "Name"] + FEATURES + ["Log10_Price", "Valuable"]
    table_image(train[tcols], os.path.join(img, "train_sample.png"),
                f"Training set — first 6 of {len(train):,} rows", max_rows=6)
    table_image(test[tcols], os.path.join(img, "test_sample.png"),
                f"Testing set — first 6 of {len(test):,} rows", max_rows=6)
    fig_split(os.path.join(img, "split.png"), len(df), len(train), len(test))
    fig_pred_vs_actual(os.path.join(img, "pred_vs_actual.png"), yl_true, yl_pred, r2, rmse)
    fig_residuals(os.path.join(img, "residuals.png"), yl_true, yl_pred)
    fig_coefs(os.path.join(img, "linear_coefs.png"), FEATURES, lin.coef_,
              "Linear regression coefficients", "Change in log10(price) per +1 unit of the feature")
    fig_coefs(os.path.join(img, "logistic_coefs.png"), FEATURES, logit.coef_[0],
              "Logistic regression coefficients", "Change in log-odds of 'valuable' per +1 unit")
    fig_confusion(os.path.join(img, "confusion_logistic.png"), cm,
                  "Logistic regression — confusion matrix")
    fig_confusion(os.path.join(img, "confusion_nb.png"), nb_cm,
                  "Naive Bayes — confusion matrix")
    fr, tr, _ = roc_curve(yc_true, p_test)
    fn, tn, _ = roc_curve(yc_true, nb_p)
    fig_roc(os.path.join(img, "roc.png"),
            [("Logistic", fr, tr, results["logistic"]["roc_auc"]),
             ("Naive Bayes", fn, tn, results["naive_bayes"]["roc_auc"])])
    fig_compare(os.path.join(img, "model_compare.png"), results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
