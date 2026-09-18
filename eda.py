"""
Exploratory Data Analysis
==================================================

Date Files
    - vintage_feb2025.csv / vintage_mar2025.csv 
    - modern_feb2025.csv  / modern_mar2025.csv    

The vintage and modern files each carry the same set of card IDs across the Feb and Mar of 2025 
only the TCG market price columns are expected to change between two data files

Steps performed in this EDA:

    1. Loads and combines all four files into one dataframe
    2. Profiles data quality (shape, dtypes, missing values)
    3. Produces descriptive statistics for numeric fields
    4. Explores categorical breakdowns (supertype, rarity, type, set)
    5. Analyzes price distributions, by rarity and by print variant
    6. Checks correlations between HP / retreat cost / price
    7. Compares vintage vs. modern cards
    8. Compares Feb -> Mar snapshots to find the biggest price movers

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = "output"
FILES = {
    ("Vintage", "2025-02"): "vintage_feb2025.csv",
    ("Vintage", "2025-03"): "vintage_mar2025.csv",
    ("Modern", "2025-02"): "modern_feb2025.csv",
    ("Modern", "2025-03"): "modern_mar2025.csv",
}

PRICE_COLS = {
    "Normal": "TCG Market Price USD (Normal)",
    "Reverse Holofoil": "TCG Market Price USD (Reverse Holofoil)",
    "Holofoil": "TCG Market Price USD (Holofoil)",
}

# --- Color palette (validated categorical order + chart chrome, light mode) --
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

SUMMARY_LINES = []  # collected markdown lines for the final report


def log(md_line):
    """Print to console and stash for the markdown summary report."""
    print(md_line)
    SUMMARY_LINES.append(md_line)


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\n![{name}]({name})")
    return path


# ----------------------------------------------------------------------------
# 1. Load & combine
# ----------------------------------------------------------------------------

def load_data():
    frames = []
    for (era, snapshot), fname in FILES.items():
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        df["era"] = era
        df["snapshot"] = snapshot
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["Release Date"] = pd.to_datetime(df["Release Date"], format="%Y/%m/%d", errors="coerce")
    df["Release Year"] = df["Release Date"].dt.year

    # Best available market price across the three print variants (a card
    # typically stocks only one or two of Normal/Reverse Holofoil/Holofoil).
    df["best_price"] = df[list(PRICE_COLS.values())].max(axis=1, skipna=True)

    # Which variant that best price came from (for variant-level breakdowns)
    def variant_of(row):
        vals = {v: row[c] for v, c in PRICE_COLS.items()}
        vals = {v: p for v, p in vals.items() if pd.notna(p)}
        if not vals:
            return np.nan
        return max(vals, key=vals.get)

    df["price_variant"] = df.apply(variant_of, axis=1)
    return df


def long_price_table(df):
    """One row per (card, snapshot, variant) with a non-null market price."""
    parts = []
    for variant, col in PRICE_COLS.items():
        sub = df[["ID", "Name", "era", "snapshot", "Rarity", col]].dropna(subset=[col]).copy()
        sub = sub.rename(columns={col: "market_price"})
        sub["variant"] = variant
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# 2. Basic overview
# ----------------------------------------------------------------------------

def basic_overview(df):
    log("\n## 1. Overview\n")
    log(f"- **Total rows (all snapshots combined):** {len(df):,}")
    log(f"- **Unique cards:** {df['ID'].nunique():,} "
        f"({df.loc[df.era=='Vintage','ID'].nunique():,} vintage, "
        f"{df.loc[df.era=='Modern','ID'].nunique():,} modern)")
    log(f"- **Columns:** {df.shape[1]}")
    log(f"- **Sets represented:** {df['Set Name'].nunique()}")
    log(f"- **Release date range:** {df['Release Date'].min().date()} to {df['Release Date'].max().date()}")
    log(f"- **Snapshots:** {', '.join(sorted(df['snapshot'].unique()))}")


# ----------------------------------------------------------------------------
# 3. Missing values
# ----------------------------------------------------------------------------

def missing_value_analysis(df):
    log("\n## 2. Missing values\n")
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    log("| Column | % missing |")
    log("|---|---|")
    for col, pct in miss.items():
        log(f"| {col} | {pct:.1f}% |")

    log("\nNote: `HP`, `Types`, `Attacks`, `Weaknesses`, `Pokedex Number` are only "
        "defined for Pokemon cards, not Trainer/Energy cards, which is why they "
        "run ~17-20% missing rather than indicating bad data. The three "
        "`TCG Market Price` columns are each only populated for the print "
        "variant (Normal / Reverse Holofoil / Holofoil) that actually exists "
        "for that card — most cards are missing at least one of the three by design.")

    fig, ax = plt.subplots(figsize=(8, max(3, len(miss) * 0.3)))
    ax.barh(miss.index[::-1], miss.values[::-1], color=CATEGORICAL[0])
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    ax.xaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "01_missing_values.png")


# ----------------------------------------------------------------------------
# 4. Descriptive stats
# ----------------------------------------------------------------------------

def descriptive_stats(df):
    log("\n## 3. Descriptive statistics (numeric fields)\n")
    num_cols = ["HP", "Retreat Cost", "best_price"]
    desc = df[num_cols].describe().T
    log("| Field | count | mean | std | min | 25% | 50% | 75% | max |")
    log("|---|---|---|---|---|---|---|---|---|")
    for idx, row in desc.iterrows():
        log(f"| {idx} | {row['count']:.0f} | {row['mean']:.2f} | {row['std']:.2f} | "
            f"{row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |")
    log("\n`best_price` is heavily right-skewed (median ${:.2f} vs. max ${:.2f}) — "
        "typical of collectible card pricing, where a small number of cards "
        "carry disproportionate value. Charts below use a log scale for price."
        .format(df["best_price"].median(), df["best_price"].max()))


# ----------------------------------------------------------------------------
# 5. Categorical distributions
# ----------------------------------------------------------------------------

def categorical_distributions(df):
    log("\n## 4. Categorical breakdowns\n")

    # Supertype
    st = df["Supertype"].value_counts()
    log("**Supertype:** " + ", ".join(f"{k} ({v:,})" for k, v in st.items()))

    # Types (Pokemon only)
    types = df["Types"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(st.index, st.values, color=CATEGORICAL[:len(st)])
    axes[0].set_title("Cards by supertype")
    axes[0].set_ylabel("count")
    axes[0].yaxis.grid(True, linewidth=0.6)
    axes[0].set_axisbelow(True)

    top_types = types.head(10)
    axes[1].barh(top_types.index[::-1], top_types.values[::-1], color=CATEGORICAL[0])
    axes[1].set_title("Top 10 Pokemon types")
    axes[1].set_xlabel("count")
    axes[1].xaxis.grid(True, linewidth=0.6)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "02_supertype_and_types.png")

    # Rarity
    rar = df["Rarity"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rar.index[::-1], rar.values[::-1], color=CATEGORICAL[2])
    ax.set_title("Top 15 rarities by card count")
    ax.set_xlabel("count")
    ax.xaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "03_rarity_counts.png")

    # Top sets
    sets = df.drop_duplicates("ID")["Set Name"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sets.index[::-1], sets.values[::-1], color=CATEGORICAL[3])
    ax.set_title("Top 15 sets by number of unique cards")
    ax.set_xlabel("unique cards")
    ax.xaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "04_top_sets.png")

    log(f"\n- {df['Rarity'].nunique()} distinct rarities across {df['Set Name'].nunique()} sets.")
    log(f"- Most common rarity: **{rar.index[0]}** ({rar.iloc[0]:,} cards).")
    log(f"- Most common type (Pokemon cards): **{types.index[0]}** ({types.iloc[0]:,} cards).")


# ----------------------------------------------------------------------------
# 6. Price distributions
# ----------------------------------------------------------------------------

def price_distribution(df, long_prices):
    log("\n## 5. Price distributions\n")

    one_snap = df[df["snapshot"] == "2025-03"].dropna(subset=["best_price"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, era in enumerate(["Vintage", "Modern"]):
        vals = np.log10(one_snap.loc[one_snap.era == era, "best_price"])
        ax.hist(vals, bins=30, alpha=0.65, color=CATEGORICAL[i], label=era)
    ax.set_xlabel("log10(price, USD)")
    ax.set_ylabel("count")
    ax.set_title("Price distribution (Mar 2025 snapshot) — log scale")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "05_price_distribution_log.png")

    # Price by variant (boxplot)
    mar_long = long_prices[long_prices.snapshot == "2025-03"]
    fig, ax = plt.subplots(figsize=(7, 5))
    variants = ["Normal", "Reverse Holofoil", "Holofoil"]
    data = [np.log10(mar_long.loc[mar_long.variant == v, "market_price"]) for v in variants]
    bp = ax.boxplot(data, tick_labels=variants, patch_artist=True, widths=0.5,
                     medianprops=dict(color=INK_PRIMARY, linewidth=1.5))
    for patch, color in zip(bp["boxes"], CATEGORICAL[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK_SECONDARY)
    ax.set_ylabel("log10(price, USD)")
    ax.set_title("Price by print variant (Mar 2025)")
    ax.yaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "06_price_by_variant.png")

    for v in variants:
        med = mar_long.loc[mar_long.variant == v, "market_price"].median()
        log(f"- Median **{v}** price: ${med:.2f}")

    # Top 15 most valuable cards
    top_cards = one_snap.sort_values("best_price", ascending=False).head(15)
    log("\n**Top 15 most valuable cards (Mar 2025 snapshot):**\n")
    log("| Card | Set | Rarity | Era | Price (USD) |")
    log("|---|---|---|---|---|")
    for _, r in top_cards.iterrows():
        log(f"| {r['Name']} | {r['Set Name']} | {r['Rarity']} | {r['era']} | ${r['best_price']:.2f} |")

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [f"{r['Name']} ({r['Set Name']})" for _, r in top_cards.iterrows()]
    ax.barh(labels[::-1], top_cards["best_price"].values[::-1], color=CATEGORICAL[5])
    ax.set_xlabel("price (USD)")
    ax.set_title("Top 15 most valuable cards (Mar 2025)")
    ax.xaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "07_top_valuable_cards.png")


def price_by_rarity(df):
    log("\n## 6. Price by rarity\n")
    one_snap = df[df["snapshot"] == "2025-03"].dropna(subset=["best_price"])
    top_rarities = one_snap["Rarity"].value_counts().head(10).index.tolist()
    med_order = (one_snap[one_snap.Rarity.isin(top_rarities)]
                 .groupby("Rarity")["best_price"].median()
                 .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [np.log10(one_snap.loc[one_snap.Rarity == r, "best_price"]) for r in med_order]
    bp = ax.boxplot(data, tick_labels=med_order, patch_artist=True, widths=0.55,
                     medianprops=dict(color=INK_PRIMARY, linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(CATEGORICAL[0])
        patch.set_alpha(0.7)
        patch.set_edgecolor(INK_SECONDARY)
    ax.set_ylabel("log10(price, USD)")
    ax.set_title("Price by rarity (10 most common rarities, Mar 2025)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.yaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "08_price_by_rarity.png")

    log(f"- Highest median price among common rarities: **{med_order[0]}** "
        f"(${one_snap.loc[one_snap.Rarity==med_order[0],'best_price'].median():.2f})")
    log(f"- Lowest median price among common rarities: **{med_order[-1]}** "
        f"(${one_snap.loc[one_snap.Rarity==med_order[-1],'best_price'].median():.2f})")


# ----------------------------------------------------------------------------
# 7. Correlations
# ----------------------------------------------------------------------------

def correlation_analysis(df):
    log("\n## 7. Correlations\n")
    one_snap = df[(df["snapshot"] == "2025-03") & (df["Supertype"] == "Pokémon")].copy()
    one_snap["log_price"] = np.log10(one_snap["best_price"])
    num = one_snap[["HP", "Retreat Cost", "log_price"]].dropna()
    corr = num.corr()

    log("Correlation matrix (Pokemon cards only, Mar 2025 snapshot; price is log10-transformed):\n")
    log("| | " + " | ".join(corr.columns) + " |")
    log("|---" * (len(corr.columns) + 1) + "|")
    for idx, row in corr.iterrows():
        log(f"| {idx} | " + " | ".join(f"{v:.2f}" for v in row) + " |")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(corr.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", SEQ_BLUE), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color=INK_PRIMARY if abs(corr.values[i, j]) < 0.6 else "white", fontsize=9)
    ax.set_title("HP / Retreat Cost / log(price) correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig(fig, "09_correlation_heatmap.png")

    # HP vs price scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(one_snap["HP"], one_snap["best_price"], s=14, alpha=0.35, color=CATEGORICAL[0])
    ax.set_yscale("log")
    ax.set_xlabel("HP")
    ax.set_ylabel("price (USD, log scale)")
    ax.set_title("HP vs. price (Pokemon cards, Mar 2025)")
    ax.yaxis.grid(True, linewidth=0.6, which="both")
    ax.set_axisbelow(True)
    savefig(fig, "10_hp_vs_price.png")

    hp_corr = corr.loc["HP", "log_price"]
    rc_corr = corr.loc["Retreat Cost", "log_price"]
    log(f"\n- HP vs. log(price): r = {hp_corr:.2f} — {'a weak' if abs(hp_corr) < 0.3 else 'a moderate'} relationship.")
    log(f"- Retreat Cost vs. log(price): r = {rc_corr:.2f} — {'a weak' if abs(rc_corr) < 0.3 else 'a moderate'} relationship.")
    log("- In collectible card pricing, rarity/scarcity and character popularity typically "
        "drive price far more than in-game stats like HP, which is consistent with the weak "
        "correlations here.")


# ----------------------------------------------------------------------------
# 8. Vintage vs Modern
# ----------------------------------------------------------------------------

def vintage_vs_modern(df):
    log("\n## 8. Vintage vs. Modern\n")
    one_snap = df[df["snapshot"] == "2025-03"].dropna(subset=["best_price"])
    summary = one_snap.groupby("era")["best_price"].agg(["count", "median", "mean", "max"])
    log("| Era | # priced cards | median price | mean price | max price |")
    log("|---|---|---|---|---|")
    for era, row in summary.iterrows():
        log(f"| {era} | {row['count']:.0f} | ${row['median']:.2f} | ${row['mean']:.2f} | ${row['max']:.2f} |")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, era in enumerate(["Vintage", "Modern"]):
        sub = one_snap[one_snap.era == era]
        axes[0].hist(np.log10(sub["best_price"]), bins=25, alpha=0.65, color=CATEGORICAL[i], label=era)
    axes[0].set_xlabel("log10(price, USD)")
    axes[0].set_title("Price distribution by era")
    axes[0].legend(frameon=False)
    axes[0].yaxis.grid(True, linewidth=0.6)
    axes[0].set_axisbelow(True)

    yearly = df.dropna(subset=["Release Year"]).drop_duplicates("ID").groupby(
        df["Release Year"].astype("Int64"))["ID"].count()
    axes[1].bar(yearly.index.astype(str), yearly.values, color=CATEGORICAL[6])
    axes[1].set_title("Unique cards by release year")
    axes[1].set_ylabel("count")
    plt.setp(axes[1].get_xticklabels(), rotation=90)
    axes[1].yaxis.grid(True, linewidth=0.6)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "11_vintage_vs_modern.png")


# ----------------------------------------------------------------------------
# 9. Feb -> Mar price change
# ----------------------------------------------------------------------------

def price_change_analysis(df):
    log("\n## 9. Price change: Feb 2025 -> Mar 2025\n")
    feb = df[df.snapshot == "2025-02"][["ID", "Name", "Set Name", "Rarity", "era", "best_price"]]
    mar = df[df.snapshot == "2025-03"][["ID", "best_price"]]
    merged = feb.merge(mar, on="ID", suffixes=("_feb", "_mar")).dropna(subset=["best_price_feb", "best_price_mar"])
    merged = merged[merged["best_price_feb"] > 0]
    merged["pct_change"] = (merged["best_price_mar"] - merged["best_price_feb"]) / merged["best_price_feb"] * 100
    merged["abs_change"] = merged["best_price_mar"] - merged["best_price_feb"]

    log(f"- {len(merged):,} cards had a priced value in both snapshots.")
    log(f"- Median % change: {merged['pct_change'].median():.1f}%")
    log(f"- Cards that increased in price: {(merged['pct_change'] > 0).sum():,} "
        f"({(merged['pct_change'] > 0).mean()*100:.1f}%)")
    log(f"- Cards that decreased in price: {(merged['pct_change'] < 0).sum():,} "
        f"({(merged['pct_change'] < 0).mean()*100:.1f}%)")
    log(f"- Unchanged: {(merged['pct_change'] == 0).sum():,}")

    fig, ax = plt.subplots(figsize=(8, 5))
    clipped = merged["pct_change"].clip(-100, 100)
    ax.hist(clipped, bins=40, color=CATEGORICAL[0])
    ax.axvline(0, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_xlabel("% change in price (Feb -> Mar 2025, clipped to ±100%)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of Feb -> Mar price changes")
    ax.yaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "12_price_change_distribution.png")

    top_gain = merged.sort_values("pct_change", ascending=False).head(10)
    top_loss = merged.sort_values("pct_change", ascending=True).head(10)

    log("\n**Top 10 gainers (by % change):**\n")
    log("| Card | Set | Feb price | Mar price | % change |")
    log("|---|---|---|---|---|")
    for _, r in top_gain.iterrows():
        log(f"| {r['Name']} | {r['Set Name']} | ${r['best_price_feb']:.2f} | ${r['best_price_mar']:.2f} | +{r['pct_change']:.1f}% |")

    log("\n**Top 10 decliners (by % change):**\n")
    log("| Card | Set | Feb price | Mar price | % change |")
    log("|---|---|---|---|---|")
    for _, r in top_loss.iterrows():
        log(f"| {r['Name']} | {r['Set Name']} | ${r['best_price_feb']:.2f} | ${r['best_price_mar']:.2f} | {r['pct_change']:.1f}% |")

    combined = pd.concat([top_gain.assign(direction="gain"), top_loss.assign(direction="loss")])
    combined = combined.sort_values("pct_change")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [STATUS_CRITICAL if v < 0 else STATUS_GOOD for v in combined["pct_change"]]
    labels = [f"{r['Name']} ({r['Set Name']})" for _, r in combined.iterrows()]
    ax.barh(labels, combined["pct_change"], color=colors)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_xlabel("% change (Feb -> Mar 2025)")
    ax.set_title("Biggest movers: top 10 gainers & top 10 decliners")
    ax.xaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    savefig(fig, "13_biggest_movers.png")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    long_prices = long_price_table(df)

    log("# Pokemon TCG Card Data — EDA Summary Report\n")
    basic_overview(df)
    missing_value_analysis(df)
    descriptive_stats(df)
    categorical_distributions(df)
    price_distribution(df, long_prices)
    price_by_rarity(df)
    correlation_analysis(df)
    vintage_vs_modern(df)
    price_change_analysis(df)

    report_path = os.path.join(OUTPUT_DIR, "EDA_SUMMARY.md")
    with open(report_path, "w") as f:
        f.write("\n".join(SUMMARY_LINES))
    print(f"\n\nDone. Charts + summary written to '{OUTPUT_DIR}/' (report: {report_path})")


if __name__ == "__main__":
    main()
