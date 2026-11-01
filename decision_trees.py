"""
Decision Trees on Pokémon TCG card prices (Python / scikit-learn)

Question: from a card's attributes alone (rarity, era, age, HP, attacks,
card type, where it sits in its set, how popular the Pokémon is), can a
decision tree predict which PRICE TIER the card falls in?

    Low  : market price <= $1
    Mid  : $1  < price <= $10
    High : price > $10

Inputs  : modern_pkmn_cards_feb2025.csv, vintage_pkmn_cards_feb2025.csv
          (+ the two mar2025 files for an out-of-time check)
Outputs : out/  (cleaned data, train/test sets, figures, metrics.json)

sklearn trees need numeric input, so every feature below is numeric
(ordinal codes, 0/1 flags, counts, or measurements).
"""
import json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

OUT = "out"
os.makedirs(OUT, exist_ok=True)
SEED = 42
CLASSES = ["Low", "Mid", "High"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue / orange / aqua
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

PRICE_COLS = ["TCG Market Price USD (Normal)",
              "TCG Market Price USD (Reverse Holofoil)",
              "TCG Market Price USD (Holofoil)"]

# Rarity -> ordinal tier (0 = most common ... 6 = chase cards)
RARITY_RANK = {
    "Common": 0, "Uncommon": 1, "Rare": 2, "Promo": 2, "Rare Holo": 3,
    "Rare Holo V": 4, "Rare Holo VMAX": 4, "Rare Holo VSTAR": 4, "Double Rare": 4,
    "ACE SPEC Rare": 4, "Radiant Rare": 4, "Amazing Rare": 4, "Rare Shiny": 4,
    "Shiny Rare": 4,
    "Rare Ultra": 5, "Ultra Rare": 5, "Trainer Gallery Rare Holo": 5,
    "Illustration Rare": 5, "Shiny Ultra Rare": 5,
    "Rare Rainbow": 6, "Rare Secret": 6, "Special Illustration Rare": 6,
    "Hyper Rare": 6,
}
RULE_BOX = re.compile(r"\b(V|VMAX|VSTAR|ex|EX|GX|Radiant)\b")


# ---------------------------------------------------------------- 1. load
def load(month):
    m = pd.read_csv(f"modern_pkmn_cards_{month}2025.csv").assign(Era="Modern")
    v = pd.read_csv(f"vintage_pkmn_cards_{month}2025.csv").assign(Era="Vintage")
    return pd.concat([m, v], ignore_index=True)


def price_tier(p):
    return pd.cut(p, [0, 1, 10, np.inf], labels=CLASSES)


# ------------------------------------------------------- 2. feature engineering
def max_damage(attacks):
    if not isinstance(attacks, str):
        return 0
    nums = [int(n) for n in re.findall(r"\((\d+)", attacks)]
    return max(nums) if nums else 0


def build_features(df):
    d = df.copy()
    d["price"] = d[PRICE_COLS].max(axis=1)           # best-printing market price
    d["card_no"] = pd.to_numeric(d["ID"].str.split("-").str[-1], errors="coerce")
    d["set_max_no"] = d.groupby("Set Name")["card_no"].transform("max")
    rel = pd.to_datetime(d["Release Date"])
    priced = pd.to_datetime(d["TCG Price Date"]).fillna(pd.Timestamp("2025-02-22"))
    base_name = d["Name"].str.replace(r"\s+(V|VMAX|VSTAR|ex|EX|GX|δ)$", "", regex=True)

    f = pd.DataFrame(index=d.index)
    f["rarity_rank"]   = d["Rarity"].map(RARITY_RANK).fillna(-1).astype(int)
    f["is_vintage"]    = (d["Era"] == "Vintage").astype(int)
    f["card_age_yrs"]  = ((priced - rel).dt.days / 365.25).round(2)
    f["hp"]            = d["HP"].fillna(0)
    f["retreat_cost"]  = d["Retreat Cost"].fillna(0)
    f["n_attacks"]     = d["Attacks"].fillna("").apply(lambda s: 0 if s == "" else s.count("(") )
    f["max_damage"]    = d["Attacks"].apply(max_damage)
    f["is_pokemon"]    = (d["Supertype"] == "Pokémon").astype(int)
    f["is_trainer"]    = (d["Supertype"] == "Trainer").astype(int)
    f["is_supporter"]  = d["Subtypes"].str.contains("Supporter", na=False).astype(int)
    f["is_rule_box"]   = d["Subtypes"].fillna("").apply(lambda s: int(bool(RULE_BOX.search(s))))
    f["set_position"]  = (d["card_no"] / d["set_max_no"]).fillna(0).round(3)  # ~1 = secret/chase slots
    f["n_printings"]   = d[PRICE_COLS].notna().sum(axis=1)
    f["pokemon_popularity"] = base_name.map(base_name.value_counts())       # # of cards of that Pokémon
    f["pokedex_no"]    = d["Pokedex Number"].fillna(0)
    meta = d[["ID", "Name", "Set Name", "Rarity", "Era", "price"]]
    return meta, f


feb = load("feb")
meta, X_all = build_features(feb)
mask = meta["price"].notna()                       # ONLY labeled rows are usable
meta, X_all = meta[mask].reset_index(drop=True), X_all[mask].reset_index(drop=True)
y_all = price_tier(meta["price"]).astype(str)

clean = pd.concat([meta, X_all, y_all.rename("price_tier")], axis=1)
clean.to_csv(f"{OUT}/cards_labeled_clean.csv", index=False)
feb.head(200).to_csv(f"{OUT}/raw_sample_200.csv", index=False)
print("raw rows:", len(feb), "| labeled rows:", len(clean))
print(y_all.value_counts())

# -------------------------------------------------- 3. train / test split
X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
    X_all, y_all, meta, test_size=0.25, random_state=SEED, stratify=y_all)
assert set(m_train["ID"]).isdisjoint(set(m_test["ID"]))       # disjoint check
pd.concat([m_train, X_train, y_train.rename("price_tier")], axis=1).to_csv(f"{OUT}/train_set.csv", index=False)
pd.concat([m_test,  X_test,  y_test.rename("price_tier")],  axis=1).to_csv(f"{OUT}/test_set.csv",  index=False)
print("train:", len(X_train), "test:", len(X_test))

# ------------------------------------------------------------ 4. three trees
NO_RARITY = [c for c in X_all.columns if c not in ("rarity_rank",)]
SPECS = {
    "tree1_gini_d3": dict(criterion="gini", max_depth=3, features=list(X_all.columns),
                          title="Tree 1 · Gini, depth 3, all features"),
    "tree2_entropy_d6": dict(criterion="entropy", max_depth=6, min_samples_leaf=10,
                             features=list(X_all.columns),
                             title="Tree 2 · Entropy, depth 6, min 10 per leaf"),
    "tree3_gini_no_rarity": dict(criterion="gini", max_depth=4, features=NO_RARITY,
                                 title="Tree 3 · Gini, depth 4, rarity removed"),
    "tree4_unpruned": dict(criterion="gini", max_depth=None, features=list(X_all.columns),
                           title="Tree 4 · Gini, no depth limit (overfit check)"),
}
results, models = {}, {}
for key, s in SPECS.items():
    feats = s["features"]
    clf = DecisionTreeClassifier(criterion=s["criterion"], max_depth=s["max_depth"],
                                 min_samples_leaf=s.get("min_samples_leaf", 1),
                                 random_state=SEED)
    clf.fit(X_train[feats], y_train)
    pred = clf.predict(X_test[feats])
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    results[key] = {
        "title": s["title"], "criterion": s["criterion"], "max_depth": s["max_depth"],
        "depth": int(clf.get_depth()), "leaves": int(clf.get_n_leaves()),
        "train_acc": round(accuracy_score(y_train, clf.predict(X_train[feats])), 4),
        "test_acc": round(accuracy_score(y_test, pred), 4),
        "confusion": cm.tolist(),
        "report": classification_report(y_test, pred, labels=CLASSES, output_dict=True),
        "importances": dict(sorted(zip(feats, clf.feature_importances_.round(4).tolist()),
                                   key=lambda t: -t[1])),
    }
    models[key] = (clf, feats)
    with open(f"{OUT}/{key}_rules.txt", "w") as fh:
        fh.write(export_text(clf, feature_names=feats, max_depth=6))
    print(f"{key}: test acc {results[key]['test_acc']:.3f}  train acc {results[key]['train_acc']:.3f}")

baseline = y_test.value_counts(normalize=True).max()
results["baseline_majority"] = round(float(baseline), 4)

# Out-of-time check: Feb-trained Tree 2 predicts March tiers for the TEST cards
mar_meta, mar_X = build_features(load("mar"))
mar_y = price_tier(mar_meta["price"]).astype(str)
mar = pd.concat([mar_meta[["ID"]], mar_X, mar_y.rename("tier")], axis=1)
mar = mar[mar_meta["price"].notna().values & mar["ID"].isin(m_test["ID"]).values]
clf2, f2 = models["tree2_entropy_d6"]
results["march_out_of_time_acc_tree2"] = round(accuracy_score(mar["tier"], clf2.predict(mar[f2])), 4)
both = m_test.merge(mar_meta[["ID", "price"]], on="ID", suffixes=("_feb", "_mar")).dropna()
results["tier_changed_feb_to_mar_pct"] = round(float(
    (price_tier(both.price_feb).astype(str) != price_tier(both.price_mar).astype(str)).mean()), 4)

# ------------------------------------------------------------------ 5. figures
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def draw_tree(key, figsize, depth=None, fontsize=8):
    clf, feats = models[key]
    fig, ax = plt.subplots(figsize=figsize)
    arts = plot_tree(clf, feature_names=feats, class_names=list(clf.classes_), filled=True,
                     rounded=True, impurity=True, proportion=False, fontsize=fontsize,
                     max_depth=depth, ax=ax)
    # recolor with the page's class palette (sklearn's default colors are arbitrary)
    order = list(clf.classes_)
    vals = clf.tree_.value[:, 0, :]
    for a in arts:
        if a.get_bbox_patch() is None:
            continue
        m = re.search(r"value = \[([^\]]+)\]", a.get_text())
        if not m:
            continue
        v = np.array([float(x) for x in re.split(r",\s*", m.group(1).strip())])
        k = order[int(v.argmax())]
        purity = (v.max() / v.sum() - 1 / 3) / (2 / 3)
        base = np.array(matplotlib.colors.to_rgb(CLASS_COLORS[CLASSES.index(k)]))
        col = 1 - (1 - base) * (0.15 + 0.6 * purity)
        a.get_bbox_patch().set_facecolor(col)
        a.get_bbox_patch().set_edgecolor(INK2)
    ax.set_title(SPECS[key]["title"], loc="left", fontsize=12, color=INK)
    save(fig, key)

draw_tree("tree1_gini_d3", (18, 7), fontsize=9)
draw_tree("tree2_entropy_d6", (26, 10), depth=3, fontsize=8)      # top 3 levels shown
SPECS["tree2_full"] = dict(SPECS["tree2_entropy_d6"]); models["tree2_full"] = models["tree2_entropy_d6"]
draw_tree("tree2_full", (60, 14), fontsize=6)
draw_tree("tree3_gini_no_rarity", (24, 9), fontsize=8)

# Confusion matrices (single-hue sequential ramp)
blues = LinearSegmentedColormap.from_list("b", ["#f2f6fc", "#2a78d6", "#0d3a73"])
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, key in zip(axes, ["tree1_gini_d3", "tree2_entropy_d6", "tree3_gini_no_rarity"]):
    cm = np.array(results[key]["confusion"])
    ax.imshow(cm, cmap=blues)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("Actual tier")
    ax.set_title(f"{SPECS[key]['title'].split(' · ')[0]}  ·  accuracy {results[key]['test_acc']:.1%}",
                 loc="left", fontsize=11, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
save(fig, "confusion_matrices")

# Train vs test accuracy
keys = ["tree1_gini_d3", "tree2_entropy_d6", "tree3_gini_no_rarity", "tree4_unpruned"]
labels = ["Tree 1\nGini d3", "Tree 2\nEntropy d6", "Tree 3\nno rarity", "Tree 4\nunpruned"]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(keys)); w = 0.36
tr = [results[k]["train_acc"] for k in keys]; te = [results[k]["test_acc"] for k in keys]
ax.bar(x - w / 2 - 0.01, tr, w, color="#9cc1ee", label="Training accuracy")
ax.bar(x + w / 2 + 0.01, te, w, color="#2a78d6", label="Testing accuracy")
for i, v in enumerate(te):
    ax.text(x[i] + w / 2, v + 0.012, f"{v:.1%}", ha="center", fontsize=9, color=INK)
ax.axhline(baseline, color=INK2, lw=1, ls="--", label=f"Always guess 'Low' ({baseline:.1%})")
ax.set_xticks(x, labels); ax.set_ylim(0, 1.18); ax.set_ylabel("Accuracy")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
save(fig, "accuracy_train_vs_test")

# Feature importances (Tree 2 and Tree 3)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
fig.subplots_adjust(wspace=0.55)
for ax, key in zip(axes, ["tree2_entropy_d6", "tree3_gini_no_rarity"]):
    imp = {k: v for k, v in results[key]["importances"].items() if v > 0.005}
    names = list(imp)[::-1]; vals = [imp[n] for n in names]
    ax.barh(names, vals, color="#2a78d6", height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_title(SPECS[key]["title"].split(",")[0] + " — feature importance", loc="left",
                 fontsize=11, color=INK)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, "feature_importance")

# Decision regions on two features: what axis-aligned splits look like
two = ["rarity_rank", "card_age_yrs"]
small = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train[two], y_train)
xx, yy = np.meshgrid(np.linspace(-1.5, 6.5, 400), np.linspace(-0.5, 27, 400))
zz = small.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
zi = np.vectorize(CLASSES.index)(zz)
fig, ax = plt.subplots(figsize=(8, 5))
ax.contourf(xx, yy, zi, levels=[-.5, .5, 1.5, 2.5],
            colors=[c + "33" for c in CLASS_COLORS])
rng = np.random.default_rng(1)
for c, col in zip(CLASSES, CLASS_COLORS):
    s = X_train[y_train == c].sample(min(350, (y_train == c).sum()), random_state=1)
    ax.scatter(s.rarity_rank + rng.uniform(-.3, .3, len(s)), s.card_age_yrs, s=12,
               color=col, edgecolor="white", linewidth=0.5, label=c, alpha=.9)
ax.set_xlabel("Rarity tier (0 = Common … 6 = Secret / Special Illustration)")
ax.set_ylabel("Card age (years)")
ax.legend(title="Price tier", frameon=False, loc="upper left")
ax.set_title("A depth-3 tree cuts the space into axis-aligned rectangles", loc="left",
             fontsize=11, color=INK)
save(fig, "decision_regions")

# Gini vs entropy curve (two-class)
p = np.linspace(0.001, 0.999, 300)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(p, 1 - p**2 - (1 - p)**2, color="#2a78d6", lw=2)
ax.plot(p, -(p * np.log2(p) + (1 - p) * np.log2(1 - p)), color="#eb6834", lw=2)
ax.text(0.5, 0.52, "Gini = 1 − Σ pᵢ²  (max 0.5)", ha="center", color=INK, fontsize=9)
ax.text(0.5, 1.03, "Entropy = −Σ pᵢ log₂ pᵢ  (max 1.0)", ha="center", color=INK, fontsize=9)
ax.set_xlabel("Proportion of class A in a node (two-class case)"); ax.set_ylabel("Impurity")
ax.set_ylim(0, 1.12); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Both measures are 0 for a pure node and peak at a 50/50 mix", loc="left",
             fontsize=11, color=INK)
save(fig, "gini_vs_entropy")

# Table images of the data at each stage
def table_png(df, name, title, colw=None):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == float:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:g}")
    df = df.fillna("").astype(str).replace("nan", "").apply(lambda col: col.str.slice(0, 26))
    fig, ax = plt.subplots(figsize=(min(1.15 * len(df.columns) + 1, 22), 0.27 * len(df) + 0.5))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="upper left", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
    t.auto_set_column_width(list(range(len(df.columns))))
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#eef3fb"); cell.set_text_props(weight="bold", color=INK)
        elif "price_tier" in df.columns and c == list(df.columns).index("price_tier"):
            k = cell.get_text().get_text()
            if k in CLASSES:
                cell.set_facecolor(CLASS_COLORS[CLASSES.index(k)] + "30")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    save(fig, name)

raw_cols = ["ID", "Name", "Supertype", "Subtypes", "HP", "Attacks", "Set Name",
            "Release Date", "Rarity", "TCG Market Price USD (Normal)",
            "TCG Market Price USD (Reverse Holofoil)", "TCG Market Price USD (Holofoil)"]
table_png(feb[raw_cols].sample(10, random_state=3).rename(columns=lambda c: c.replace("TCG Market Price USD ", "Mkt $ ")),
          "raw_sample", "Raw data (10 random rows of 7,213, 12 of 27 columns shown)")
show = ["Name", "rarity_rank", "is_vintage", "card_age_yrs", "hp", "max_damage",
        "is_rule_box", "set_position", "pokemon_popularity", "price", "price_tier"]
table_png(clean[show].sample(10, random_state=7), "clean_sample",
          "Cleaned, labeled, all-numeric features (Name/price kept only for reference)")
table_png(pd.concat([m_train, X_train, y_train.rename("price_tier")], axis=1)[show].head(8),
          "train_sample", f"Training set — first 8 of {len(X_train):,} rows")
table_png(pd.concat([m_test, X_test, y_test.rename("price_tier")], axis=1)[show].head(8),
          "test_sample", f"Testing set — first 8 of {len(X_test):,} rows")

# Label distribution across the split
fig, ax = plt.subplots(figsize=(7, 3.2))
for i, (nm, ys) in enumerate([("All labeled", y_all), ("Training", y_train), ("Testing", y_test)]):
    left = 0
    for c, col in zip(CLASSES, CLASS_COLORS):
        share = (ys == c).mean()
        ax.barh(i, share, left=left, color=col, edgecolor="white", linewidth=2, height=0.6)
        ax.text(left + share / 2, i, f"{c} {share:.0%}" if share > 0.15 else f"{share:.0%}", ha="center", va="center",
                color="white", fontsize=8.5, weight="bold")
        left += share
ax.set_yticks(range(3), [f"All labeled (n={len(y_all):,})", f"Training (n={len(y_train):,})",
                          f"Testing (n={len(y_test):,})"])
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Stratified split keeps the same tier mix in both sets", loc="left",
             fontsize=11, color=INK)
save(fig, "split_balance")

results["n_raw"] = int(len(feb)); results["n_labeled"] = int(len(clean))
results["n_train"] = int(len(X_train)); results["n_test"] = int(len(X_test))
results["label_counts"] = y_all.value_counts().to_dict()
json.dump(results, open(f"{OUT}/metrics.json", "w"), indent=2)
print(json.dumps({k: (v["test_acc"] if isinstance(v, dict) and "test_acc" in v else v)
                  for k, v in results.items()}, indent=1))
