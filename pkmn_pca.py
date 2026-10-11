import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = Path("D:/CU Boulder/Courses/CSCI 5621 ML for DS/Data/")
OUT_DIR = Path("D:/CU Boulder/Courses/CSCI 5621 ML for DS/pca/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and prepare (same steps as the clustering script)
# ---------------------------------------------------------------------------
PRICE_COLS = ["TCG Market Price USD (Holofoil)", "TCG Market Price USD (Normal)",
              "TCG Market Price USD (Reverse Holofoil)"]


def load_era(era):
    feb = pd.read_csv(DATA_DIR / f"{era}_pkmn_cards_feb2025.csv")
    mar = pd.read_csv(DATA_DIR / f"{era}_pkmn_cards_mar2025.csv")
    feb["price_feb"] = feb[PRICE_COLS].max(axis=1)
    mar["price_mar"] = mar[PRICE_COLS].max(axis=1)
    df = feb.merge(mar[["ID", "price_mar"]], on="ID", how="left")
    df["era"] = era
    return df


def attack_stats(s):
    if pd.isna(s):
        return 0, 0
    dmg = re.findall(r"\(([^)]*)\)", s)
    nums = [int(m) for d in dmg for m in re.findall(r"\d+", d)]
    return len(dmg), max(nums) if nums else 0


cards = pd.concat([load_era("modern"), load_era("vintage")], ignore_index=True)
cards = cards[cards["Supertype"] == "Pokémon"].dropna(subset=["HP", "price_feb", "price_mar"])
cards = cards[(cards["price_feb"] > 0) & (cards["price_mar"] > 0)].copy()
cards[["n_attacks", "max_damage"]] = cards["Attacks"].apply(lambda s: pd.Series(attack_stats(s)))
cards["log_price"] = np.log1p(cards["price_mar"])
cards["price_change_pct"] = (cards["price_mar"] / cards["price_feb"] - 1).clip(-1, 1)
cards["card_age_yrs"] = (pd.Timestamp("2025-03-01")
                         - pd.to_datetime(cards["Release Date"])).dt.days / 365.25

FEATURES = ["HP", "Retreat Cost", "n_attacks", "max_damage",
            "log_price", "price_change_pct", "card_age_yrs"]
LABELS = ["HP", "Retreat cost", "# attacks", "Max damage",
          "Log price", "Price change", "Card age"]
X = StandardScaler().fit_transform(cards[FEATURES])

# ---------------------------------------------------------------------------
# 2. PCA with all components
# ---------------------------------------------------------------------------
pca = PCA().fit(X)
scores = pca.transform(X)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
eig = pca.explained_variance_
pcs = [f"PC{i+1}" for i in range(len(evr))]

print(f"Cards: {len(cards)}\n")
print(pd.DataFrame({"eigenvalue": eig, "var_explained": evr, "cumulative": cum},
                   index=pcs).round(3).to_string(), "\n")
n_kaiser = int((eig > 1).sum())
n_80 = int(np.argmax(cum >= 0.80) + 1)
print(f"Kaiser rule (eigenvalue > 1): keep {n_kaiser} components")
print(f"Components for 80% of variance: {n_80}\n")

loadings = pd.DataFrame(pca.components_.T * np.sqrt(eig), index=LABELS, columns=pcs)
print("Loadings (correlation of each feature with each PC):")
print(loadings.iloc[:, :4].round(2).to_string(), "\n")

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
# Scree plot
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(1, len(evr) + 1)
ax.bar(x, evr * 100, color="#4C72B0", label="Individual")
ax.plot(x, cum * 100, "o-", color="#DD8452", label="Cumulative")
ax.axhline(80, ls="--", c="gray", lw=1)
ax.text(len(evr) + 0.4, 80, "80%", va="center", color="gray")
for xi, v in zip(x, evr * 100):
    ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.set(xticks=x, xticklabels=pcs, ylabel="% of variance explained",
       title="Scree plot: variance explained by each principal component", ylim=(0, 105))
ax.legend(loc="center right")
fig.tight_layout(); fig.savefig(OUT_DIR / "pca_scree.png", dpi=130); plt.close(fig)

# Loadings heatmap (first 4 PCs)
k = 4
L = loadings.iloc[:, :k]
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(L.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set(xticks=range(k), xticklabels=[f"{p}\n({evr[i]*100:.0f}%)" for i, p in enumerate(L.columns)],
       yticks=range(len(LABELS)), yticklabels=LABELS, title="PCA loadings (feature–component correlation)")
for i in range(L.shape[0]):
    for j in range(k):
        v = L.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.55 else "black", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(OUT_DIR / "pca_loadings.png", dpi=130); plt.close(fig)

# Biplot: PC1 vs PC2, points by era, arrows = loadings
fig, ax = plt.subplots(figsize=(9, 7))
for era, color in [("modern", "#4C72B0"), ("vintage", "#C44E52")]:
    m = (cards["era"] == era).values
    ax.scatter(scores[m, 0], scores[m, 1], s=6, alpha=0.35, c=color,
               label=f"{era.title()} ({m.sum():,})")
scale = 3.2
for i, lab in enumerate(LABELS):
    dx, dy = loadings.iloc[i, 0] * scale, loadings.iloc[i, 1] * scale
    ax.arrow(0, 0, dx, dy, color="black", width=0.015, head_width=0.12, length_includes_head=True)
    ax.text(dx * 1.12, dy * 1.12, lab, fontsize=10, ha="center", va="center", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
ax.set(xlabel=f"PC1 ({evr[0]*100:.0f}% of variance)", ylabel=f"PC2 ({evr[1]*100:.0f}% of variance)",
       title="PCA biplot: cards on PC1 and PC2, with feature loadings")
ax.legend(loc="upper right", markerscale=3)
fig.tight_layout(); fig.savefig(OUT_DIR / "pca_biplot.png", dpi=130); plt.close(fig)

# Cards at the extremes of each PC (helps name the components)
for j in range(2):
    s = pd.Series(scores[:, j], index=cards.index)
    print(f"PC{j+1} highest:", ", ".join(cards.loc[s.nlargest(5).index, "Name"]))
    print(f"PC{j+1} lowest: ", ", ".join(cards.loc[s.nsmallest(5).index, "Name"]))

out = cards[["ID", "Name", "Set Name", "era", "Rarity", *FEATURES, "price_feb", "price_mar"]].copy()
for j in range(k):
    out[f"PC{j+1}"] = scores[:, j]
out.to_csv(OUT_DIR / "card_pca_scores.csv", index=False)
print(f"\nSaved results to {OUT_DIR.resolve()}")
