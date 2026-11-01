"""
Naive Bayes on Pokemon TCG cards: predict a card's PRICE TIER from its attributes.

Label (supervised target): Price Tier, from the March 2025 TCGplayer market price
(highest of the Normal / Reverse Holofoil / Holofoil market prices):
    Low  : < $1
    Mid  : $1 to < $10
    High : >= $10

Features: card attributes turned into non-negative COUNTS, as sklearn's
MultinomialNB requires (one-hot indicators + small integer counts).

Usage:
    python nb_pokemon.py <modern_mar2025.csv> <vintage_mar2025.csv> [out_dir]
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)

MODERN, VINTAGE = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "nb_output"
os.makedirs(OUT, exist_ok=True)
SEED = 42
TIERS = ["Low", "Mid", "High"]

# ---------------------------------------------------------------- 1. load + label
mod = pd.read_csv(MODERN);  mod["Era"] = "Modern"
vin = pd.read_csv(VINTAGE); vin["Era"] = "Vintage"
cards = pd.concat([mod, vin], ignore_index=True)

price_cols = ["TCG Market Price USD (Normal)",
              "TCG Market Price USD (Reverse Holofoil)",
              "TCG Market Price USD (Holofoil)"]
cards["Market Price"] = cards[price_cols].max(axis=1)

# Only LABELED rows can be used for supervised learning: drop cards with no price
labeled = cards.dropna(subset=["Market Price"]).copy()
labeled["Price Tier"] = pd.cut(labeled["Market Price"], [0, 1, 10, np.inf],
                               right=False, labels=TIERS).astype(str)
print(f"All cards: {len(cards)}   labeled (have a price): {len(labeled)}")
print(labeled["Price Tier"].value_counts())

# ---------------------------------------------------------------- 2. features -> counts
def n_attacks(s):
    return 0 if pd.isna(s) or s == "" else len(str(s).split("), "))

def max_damage(s):
    if pd.isna(s):
        return 0
    nums = [int("".join(ch for ch in part if ch.isdigit()) or 0)
            for part in str(s).split("(")[1:]]
    return max(nums) if nums else 0

X = pd.DataFrame(index=labeled.index)
# one-hot categorical attributes (0/1 counts)
for col, prefix in [("Era", "era"), ("Supertype", "super"), ("Types", "type"),
                    ("Rarity", "rarity")]:
    X = X.join(pd.get_dummies(labeled[col].fillna("None"), prefix=prefix).astype(int))
# subtypes can hold several tokens ("Basic, V") -> multi-hot
sub = labeled["Subtypes"].fillna("None").str.get_dummies(sep=", ").add_prefix("sub_")
X = X.join(sub)
# release year (set age) one-hot
X = X.join(pd.get_dummies(labeled["Release Date"].str[:4], prefix="year").astype(int))
# HP bins one-hot (raw HP would swamp the other counts in a multinomial model)
hp_bin = pd.cut(labeled["HP"].fillna(0), [-1, 0, 60, 100, 150, 220, 1000],
                labels=["none", "1-60", "61-100", "101-150", "151-220", "221+"])
X = X.join(pd.get_dummies(hp_bin, prefix="hp").astype(int))
# damage bins one-hot
dmg = labeled["Attacks"].apply(max_damage)
dmg_bin = pd.cut(dmg, [-1, 0, 50, 120, 200, 10000],
                 labels=["0", "1-50", "51-120", "121-200", "201+"])
X = X.join(pd.get_dummies(dmg_bin, prefix="dmg").astype(int))
# small integer counts
X["n_attacks"] = labeled["Attacks"].apply(n_attacks)
X["retreat_cost"] = labeled["Retreat Cost"].fillna(0).astype(int)
X.columns = [c.replace(" ", "_") for c in X.columns]
y = labeled["Price Tier"]
assert (X.values >= 0).all(), "MultinomialNB needs non-negative features"
print("Feature matrix:", X.shape)

# save the raw labeled sample and the prepared numeric sample
raw_cols = ["ID", "Name", "Era", "Supertype", "Subtypes", "HP", "Types", "Attacks",
            "Retreat Cost", "Set Name", "Release Date", "Rarity", "Market Price",
            "Price Tier"]
labeled[raw_cols].to_csv(f"{OUT}/labeled_cards_raw.csv", index=False)
prepared = X.copy(); prepared.insert(0, "Price Tier", y); prepared.insert(0, "ID", labeled["ID"])
prepared.to_csv(f"{OUT}/prepared_numeric_data.csv", index=False)

# ---------------------------------------------------------------- 3. train / test split
ids = labeled["ID"]
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.30, random_state=SEED, stratify=y)
assert set(X_train.index).isdisjoint(X_test.index), "train and test must be disjoint"
train_df = X_train.copy(); train_df.insert(0, "Price Tier", y_train); train_df.insert(0, "ID", id_train)
test_df = X_test.copy();   test_df.insert(0, "Price Tier", y_test);   test_df.insert(0, "ID", id_test)
train_df.to_csv(f"{OUT}/train_set.csv", index=False)
test_df.to_csv(f"{OUT}/test_set.csv", index=False)
print(f"Train: {len(X_train)}  Test: {len(X_test)}  overlap: "
      f"{len(set(X_train.index) & set(X_test.index))}")

# ---------------------------------------------------------------- 4. models
mnb = MultinomialNB(alpha=1.0).fit(X_train, y_train)
pred = mnb.predict(X_test)
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=TIERS)
print("\nMultinomial NB accuracy:", round(acc, 4))
print(classification_report(y_test, pred, labels=TIERS, digits=3))

bnb = BernoulliNB(alpha=1.0, binarize=0.5).fit(X_train, y_train)
bpred = bnb.predict(X_test)
bacc = accuracy_score(y_test, bpred)
bcm = confusion_matrix(y_test, bpred, labels=TIERS)
print("Bernoulli NB accuracy:", round(bacc, 4))

baseline = (y_test == y_train.value_counts().idxmax()).mean()
print("Majority-class baseline:", round(baseline, 4))

# accuracy by era
era_test = labeled.loc[X_test.index, "Era"]
era_acc = {e: float(accuracy_score(y_test[era_test == e], pred[(era_test == e).values]))
           for e in ["Modern", "Vintage"]}

# most indicative features per class: log P(f|class) - mean log P(f|other classes)
flp = pd.DataFrame(mnb.feature_log_prob_, index=mnb.classes_, columns=X.columns)
top = {}
for c in TIERS:
    others = [o for o in TIERS if o != c]
    diff = flp.loc[c] - flp.loc[others].mean()
    top[c] = diff.sort_values(ascending=False).head(8).round(3).to_dict()

p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=TIERS)
metrics = dict(accuracy=acc, bernoulli_accuracy=bacc, baseline=baseline,
               n_cards=len(cards), n_labeled=len(labeled), n_features=X.shape[1],
               n_train=len(X_train), n_test=len(X_test),
               class_counts=y.value_counts().reindex(TIERS).to_dict(),
               train_counts=y_train.value_counts().reindex(TIERS).to_dict(),
               test_counts=y_test.value_counts().reindex(TIERS).to_dict(),
               confusion_matrix=cm.tolist(), bernoulli_confusion_matrix=bcm.tolist(),
               per_class={t: dict(precision=float(p[i]), recall=float(r[i]),
                                  f1=float(f[i]), support=int(s[i]))
                          for i, t in enumerate(TIERS)},
               era_accuracy=era_acc, top_features=top,
               priors=dict(zip(mnb.classes_, np.exp(mnb.class_log_prior_).round(4))))
with open(f"{OUT}/metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2, default=float)

# ---------------------------------------------------------------- 5. figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TIER_COL = {"Low": "#2a78d6", "Mid": "#eb6834", "High": "#1baf7a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

def table_png(df, path, title, col_w=None):
    df = df.astype(object).where(df.notna(), "").astype(str).apply(lambda c: c.str.slice(0, 24))
    fig_w = max(8, 1.45 * df.shape[1]); fig_h = 0.3 * (len(df) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
    t.auto_set_column_width(list(range(df.shape[1])))
    for (r_, c_), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r_ == 0:
            cell.set_facecolor("#eef3fb"); cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(SURF); cell.set_text_props(color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=11, weight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

sample_raw = pd.concat([
    labeled[labeled.Era == "Modern"][raw_cols].sample(7, random_state=SEED),
    labeled[labeled.Era == "Vintage"][raw_cols].sample(3, random_state=SEED)]).drop(columns=["Attacks"])
sample_raw["Market Price"] = sample_raw["Market Price"].map("${:,.2f}".format)
table_png(sample_raw, f"{OUT}/img_raw_sample.png", "Labeled raw data (10 random cards)")

show_cols = ["ID", "Price Tier", "era_Vintage", "super_Pokémon", "type_Fire",
             "rarity_Common", "rarity_Rare_Holo", "sub_Basic", "sub_V", "hp_61-100",
             "n_attacks", "retreat_cost"]
show_cols = [c for c in show_cols if c in train_df.columns]
table_png(prepared.loc[sample_raw.index[:8], show_cols], f"{OUT}/img_prepared_sample.png",
          f"Prepared numeric data (8 rows, 12 of {X.shape[1]+2} columns)")
table_png(train_df[show_cols].head(6), f"{OUT}/img_train_sample.png",
          f"Training set: {len(train_df)} rows (first 6 shown)")
table_png(test_df[show_cols].head(6), f"{OUT}/img_test_sample.png",
          f"Testing set: {len(test_df)} rows (first 6 shown)")

# class balance in train vs test
fig, ax = plt.subplots(figsize=(7, 2.2))
xs = np.arange(2); left = np.zeros(2)
for t in TIERS:
    vals = np.array([metrics["train_counts"][t] / len(y_train), metrics["test_counts"][t] / len(y_test)])
    ax.barh(xs, vals, left=left, color=TIER_COL[t], label=t, height=0.55, edgecolor=SURF, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(left[i] + v / 2, xs[i], f"{t}\n{v:.0%}", ha="center", va="center", color="white",
                fontsize=9, weight="bold")
    left += vals
ax.set_yticks(xs, [f"Train (n={len(y_train)})", f"Test (n={len(y_test)})"])
ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Stratified split keeps the same tier mix in both sets", loc="left", color=INK, weight="bold")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
fig.savefig(f"{OUT}/img_split_balance.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# confusion matrices
cmap = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc2ef", "#2a78d6", "#103d73"])
def cm_png(m, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    rowpct = m / m.sum(axis=1, keepdims=True)
    ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=1)  # shade by row %, so each class reads fairly
    for i in range(3):
        for j in range(3):
            pct = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]}\n({pct:.0%} of row)", ha="center", va="center",
                    color="white" if pct > 0.45 else INK, fontsize=10)
    ax.set_xticks(range(3), TIERS); ax.set_yticks(range(3), TIERS)
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("Actual tier")
    ax.set_title(title, loc="left", color=INK, weight="bold")
    for s_ in ax.spines.values(): s_.set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
cm_png(cm, f"{OUT}/img_confusion_multinomial.png", f"Multinomial NB  (accuracy {acc:.1%})")
cm_png(bcm, f"{OUT}/img_confusion_bernoulli.png", f"Bernoulli NB  (accuracy {bacc:.1%})")

# accuracy comparison
fig, ax = plt.subplots(figsize=(7, 2.8))
names = ["Majority-class guess", "Bernoulli NB", "Multinomial NB"]
vals = [baseline, bacc, acc]; cols = ["#b9b8b2", "#eb6834", "#2a78d6"]
ax.barh(names, vals, color=cols, height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.1%}", va="center", color=INK, fontsize=10)
ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.set_title("Test-set accuracy", loc="left", color=INK, weight="bold")
fig.savefig(f"{OUT}/img_accuracy.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# per-class precision / recall
fig, ax = plt.subplots(figsize=(7, 3))
w = 0.36; xs = np.arange(3)
ax.bar(xs - w/2 - 0.01, p, w, color="#2a78d6", label="Precision")
ax.bar(xs + w/2 + 0.01, r, w, color="#eb6834", label="Recall")
for i in range(3):
    ax.text(i - w/2 - 0.01, p[i] + 0.02, f"{p[i]:.2f}", ha="center", fontsize=9, color=INK)
    ax.text(i + w/2 + 0.01, r[i] + 0.02, f"{r[i]:.2f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(xs, TIERS); ax.set_ylim(0, 1.1)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.set_title("Multinomial NB precision and recall by tier", loc="left", color=INK, weight="bold")
fig.savefig(f"{OUT}/img_precision_recall.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# indicative features
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=False)
for ax, t in zip(axes, TIERS):
    items = list(top[t].items())[:6][::-1]
    ax.barh([k for k, _ in items], [v for _, v in items], color=TIER_COL[t], height=0.6)
    ax.set_title(f"{t} tier", loc="left", color=INK, weight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
fig.suptitle("Features most characteristic of each tier (log-probability lift vs other tiers)",
             x=0.01, ha="left", color=INK, weight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/img_top_features.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("\nDone. Outputs in", OUT)
