import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

DATA_DIR = Path("D:/CU Boulder/Courses/CSCI 5621 ML for DS/Data/")
OUT_DIR = Path("D:/CU Boulder/Courses/CSCI 5621 ML for DS/clustering/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and merge sets data
# ---------------------------------------------------------------------------
PRICE_COLS = [
    "TCG Market Price USD (Holofoil)",
    "TCG Market Price USD (Normal)",
    "TCG Market Price USD (Reverse Holofoil)",
]


def market_price(df):
    """One price per card: the highest available market price across print variants."""
    return df[PRICE_COLS].max(axis=1)


def load_era(era):
    feb = pd.read_csv(DATA_DIR / f"{era}_pkmn_cards_feb2025.csv")
    mar = pd.read_csv(DATA_DIR / f"{era}_pkmn_cards_mar2025.csv")
    feb["price_feb"] = market_price(feb)
    mar["price_mar"] = market_price(mar)
    df = feb.merge(mar[["ID", "price_mar"]], on="ID", how="left")
    df["era"] = era
    return df


cards = pd.concat([load_era("modern"), load_era("vintage")], ignore_index=True)

# ---------------------------------------------------------------------------
# 2. Data preprocessing (Pokémon cards only: Trainers/Energy have no HP/attacks)
# ---------------------------------------------------------------------------
cards = cards[cards["Supertype"] == "Pokémon"].copy()
cards = cards.dropna(subset=["HP", "price_feb", "price_mar"])
cards = cards[(cards["price_feb"] > 0) & (cards["price_mar"] > 0)]


def attack_stats(s):
    """'Find a Friend (), Line Force (50+)' -> (n_attacks=2, max_damage=50)."""
    if pd.isna(s):
        return 0, 0
    dmg = re.findall(r"\(([^)]*)\)", s)
    nums = [int(m) for d in dmg for m in re.findall(r"\d+", d)]
    return len(dmg), max(nums) if nums else 0


cards[["n_attacks", "max_damage"]] = cards["Attacks"].apply(
    lambda s: pd.Series(attack_stats(s)))
cards["log_price"] = np.log1p(cards["price_mar"])
cards["price_change_pct"] = (cards["price_mar"] / cards["price_feb"] - 1).clip(-1, 1)
cards["card_age_yrs"] = (pd.Timestamp("2025-03-01")
                         - pd.to_datetime(cards["Release Date"])).dt.days / 365.25
cards["is_vintage"] = (cards["era"] == "vintage").astype(int)

FEATURES = ["HP", "Retreat Cost", "n_attacks", "max_damage",
            "log_price", "price_change_pct", "card_age_yrs"]
scaler = StandardScaler()
X = scaler.fit_transform(cards[FEATURES])
print(f"Cards clustered: {len(cards)}  |  features: {FEATURES}\n")

# ---------------------------------------------------------------------------
# 3. K-means
# ---------------------------------------------------------------------------
ks = range(2, 11)
inertias, sils = [], []
sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X), min(3000, len(X)), replace=False)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X[sample_idx], km.labels_[sample_idx]))
    print(f"k={k:2d}  inertia={km.inertia_:10.1f}  silhouette={sils[-1]:.3f}")

best_k = list(ks)[int(np.argmax(sils))]
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
cards["kmeans_cluster"] = kmeans.labels_
print(f"\nChosen k = {best_k} (highest silhouette)\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertias, "o-"); ax[0].set(title="Elbow", xlabel="k", ylabel="Inertia")
ax[1].plot(list(ks), sils, "o-"); ax[1].set(title="Silhouette", xlabel="k", ylabel="Score")
ax[1].axvline(best_k, ls="--", c="gray")
fig.tight_layout(); fig.savefig(OUT_DIR / "kmeans_k_selection.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Hierarchical clustering with cosine distance
# ---------------------------------------------------------------------------
# Cosine distance = 1 - cosine similarity. pdist computes it directly; the
# explicit check below shows it matches sklearn's cosine_similarity.
cos_dist = pdist(X, metric="cosine")
check = 1 - cosine_similarity(X[:5])
assert np.allclose(check[np.triu_indices(5, 1)], pdist(X[:5], metric="cosine"))

Z = linkage(cos_dist, method="average")
cards["hclust_cluster"] = fcluster(Z, t=best_k, criterion="maxclust") - 1
hc_sil = silhouette_score(X[sample_idx], cards["hclust_cluster"].values[sample_idx],
                          metric="cosine")
print(f"Hierarchical (cosine, average linkage), {best_k} clusters: "
      f"cosine silhouette={hc_sil:.3f}\n")

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True, ax=ax,
           color_threshold=Z[-(best_k - 1), 2])
ax.set(title="Hierarchical clustering dendrogram (cosine distance, average linkage)",
       xlabel="Cluster (leaf count)", ylabel="Cosine distance")
fig.tight_layout(); fig.savefig(OUT_DIR / "hclust_dendrogram.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Compare and describe clusters
# ---------------------------------------------------------------------------
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
pcs = pca_2d.transform(X)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for a, col, title in [(ax[0], "kmeans_cluster", "K-means (Euclidean)"),
                      (ax[1], "hclust_cluster", "Hierarchical (cosine)")]:
    a.scatter(pcs[:, 0], pcs[:, 1], c=cards[col], cmap="tab10", s=6, alpha=0.6)
    a.set(title=title, xlabel="PC1", ylabel="PC2")
# --- K-means centroids on the PCA plot ---
centroids_2d = pca_2d.transform(kmeans.cluster_centers_)   # project 7-D centroids onto PC1/PC2
ax[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=250,
              c="red", edgecolors="black", linewidths=1.5, zorder=5, label="Centroid")
for i, (cx, cy) in enumerate(centroids_2d):
    ax[0].annotate(f"C{i}", (cx, cy), xytext=(8, 8), textcoords="offset points",
                   fontsize=11, weight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))
ax[0].legend(loc="upper right")

# Centroids in original units (printed and saved)
centroids_orig = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                              columns=FEATURES).round(2)
centroids_orig.index.name = "cluster"
print("K-means centroids (original units):")
print(centroids_orig.to_string(), "\n")
centroids_orig.to_csv(OUT_DIR / "kmeans_centroids.csv")
fig.tight_layout(); fig.savefig(OUT_DIR / "clusters_pca.png", dpi=130); plt.close(fig)

for col in ["kmeans_cluster", "hclust_cluster"]:
    prof = cards.groupby(col).agg(
        n=("ID", "size"), HP=("HP", "mean"), retreat=("Retreat Cost", "mean"),
        attacks=("n_attacks", "mean"), max_dmg=("max_damage", "mean"),
        median_price=("price_mar", "median"), pct_change=("price_change_pct", "mean"),
        age_yrs=("card_age_yrs", "mean"), pct_vintage=("is_vintage", "mean"))
    print(f"Cluster profiles — {col}\n{prof.round(2).to_string()}\n")

print("Cross-tab (rows = k-means, cols = hierarchical):")
print(pd.crosstab(cards["kmeans_cluster"], cards["hclust_cluster"]))

cards[["ID", "Name", "Set Name", "era", "Rarity", *FEATURES, "price_feb", "price_mar",
       "kmeans_cluster", "hclust_cluster"]].to_csv(OUT_DIR / "card_clusters.csv", index=False)
print(f"\nSaved results to {OUT_DIR.resolve()}")
