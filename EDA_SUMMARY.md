# Pokemon TCG Card Data — EDA Summary Report


## 1. Overview

- **Total rows (all snapshots combined):** 14,426
- **Unique cards:** 7,213 (868 vintage, 6,345 modern)
- **Columns:** 32
- **Sets represented:** 45
- **Release date range:** 1999-01-09 to 2025-01-17
- **Snapshots:** 2025-02, 2025-03

## 2. Missing values

| Column | % missing |
|---|---|
| TCG Market Price USD (Holofoil) | 56.1% |
| TCG Low Price USD (Holofoil) | 56.1% |
| TCG High Price USD (Holofoil) | 56.1% |
| TCG Market Price USD (Normal) | 51.1% |
| TCG Low Price USD (Normal) | 51.1% |
| TCG High Price USD (Normal) | 51.1% |
| TCG Market Price USD (Reverse Holofoil) | 48.0% |
| TCG Low Price USD (Reverse Holofoil) | 48.0% |
| TCG High Price USD (Reverse Holofoil) | 48.0% |
| Weaknesses | 20.2% |
| Pokedex Number | 18.3% |
| Attacks | 16.9% |
| HP | 16.8% |
| Types | 16.8% |
| Artist | 11.1% |
| price_variant | 7.2% |
| best_price | 7.2% |
| Subtypes | 1.7% |
| Rarity | 0.4% |
| TCG Player URL | 0.3% |
| TCG Price Date | 0.3% |

Note: `HP`, `Types`, `Attacks`, `Weaknesses`, `Pokedex Number` are only defined for Pokemon cards, not Trainer/Energy cards, which is why they run ~17-20% missing rather than indicating bad data. The three `TCG Market Price` columns are each only populated for the print variant (Normal / Reverse Holofoil / Holofoil) that actually exists for that card — most cards are missing at least one of the three by design.

![01_missing_values.png](01_missing_values.png)

## 3. Descriptive statistics (numeric fields)

| Field | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| HP | 11996 | 127.23 | 76.56 | 10.00 | 70.00 | 100.00 | 170.00 | 340.00 |
| Retreat Cost | 14426 | 1.38 | 1.04 | 0.00 | 1.00 | 1.00 | 2.00 | 4.00 |
| best_price | 13382 | 8.26 | 47.87 | 0.05 | 0.17 | 0.37 | 2.65 | 1747.73 |

`best_price` is heavily right-skewed (median $0.37 vs. max $1747.73) — typical of collectible card pricing, where a small number of cards carry disproportionate value. Charts below use a log scale for price.

## 4. Categorical breakdowns

**Supertype:** Pokémon (11,976), Trainer (2,222), Energy (228)

![02_supertype_and_types.png](02_supertype_and_types.png)

![03_rarity_counts.png](03_rarity_counts.png)

![04_top_sets.png](04_top_sets.png)

- 23 distinct rarities across 45 sets.
- Most common rarity: **Common** (3,580 cards).
- Most common type (Pokemon cards): **Grass** (1,694 cards).

## 5. Price distributions


![05_price_distribution_log.png](05_price_distribution_log.png)

![06_price_by_variant.png](06_price_by_variant.png)
- Median **Normal** price: $0.07
- Median **Reverse Holofoil** price: $0.18
- Median **Holofoil** price: $2.34

**Top 15 most valuable cards (Mar 2025 snapshot):**

| Card | Set | Rarity | Era | Price (USD) |
|---|---|---|---|---|
| Umbreon VMAX | Evolving Skies | Rare Rainbow | Modern | $1727.65 |
| Umbreon ex | Prismatic Evolutions | Special Illustration Rare | Modern | $1594.56 |
| Charizard | Legendary Collection | Rare Holo | Vintage | $1199.00 |
| Rayquaza VMAX | Evolving Skies | Rare Rainbow | Modern | $761.77 |
| Gengar VMAX | Fusion Strike | Rare Rainbow | Modern | $717.19 |
| Giratina V | Lost Origin | Rare Ultra | Modern | $709.09 |
| Lucky Stadium | Wizards Black Star Promos | Promo | Vintage | $605.66 |
| Charizard | SWSH Black Star Promos | Promo | Modern | $599.98 |
| Pokémon Center | Wizards Black Star Promos | Promo | Vintage | $539.50 |
| Sylveon ex | Prismatic Evolutions | Special Illustration Rare | Modern | $474.70 |
| Pikachu ex | Surging Sparks | Special Illustration Rare | Modern | $459.38 |
| Leafeon VMAX | Evolving Skies | Rare Rainbow | Modern | $445.74 |
| Espeon ex | Prismatic Evolutions | Special Illustration Rare | Modern | $428.26 |
| Sylveon VMAX | Evolving Skies | Rare Rainbow | Modern | $422.06 |
| Greninja ex | Twilight Masquerade | Special Illustration Rare | Modern | $410.29 |

![07_top_valuable_cards.png](07_top_valuable_cards.png)

## 6. Price by rarity


![08_price_by_rarity.png](08_price_by_rarity.png)
- Highest median price among common rarities: **Illustration Rare** ($12.06)
- Lowest median price among common rarities: **Common** ($0.16)

## 7. Correlations

Correlation matrix (Pokemon cards only, Mar 2025 snapshot; price is log10-transformed):

| | HP | Retreat Cost | log_price |
|---|---|---|---|
| HP | 1.00 | 0.39 | 0.41 |
| Retreat Cost | 0.39 | 1.00 | 0.07 |
| log_price | 0.41 | 0.07 | 1.00 |

![09_correlation_heatmap.png](09_correlation_heatmap.png)

![10_hp_vs_price.png](10_hp_vs_price.png)

- HP vs. log(price): r = 0.41 — a moderate relationship.
- Retreat Cost vs. log(price): r = 0.07 — a weak relationship.
- In collectible card pricing, rarity/scarcity and character popularity typically drive price far more than in-game stats like HP, which is consistent with the weak correlations here.

## 8. Vintage vs. Modern

| Era | # priced cards | median price | mean price | max price |
|---|---|---|---|---|
| Modern | 6299 | $0.31 | $6.41 | $1727.65 |
| Vintage | 392 | $5.45 | $38.93 | $1199.00 |

![11_vintage_vs_modern.png](11_vintage_vs_modern.png)

## 9. Price change: Feb 2025 -> Mar 2025

- 6,691 cards had a priced value in both snapshots.
- Median % change: 0.0%
- Cards that increased in price: 3,305 (49.4%)
- Cards that decreased in price: 2,115 (31.6%)
- Unchanged: 1,271

![12_price_change_distribution.png](12_price_change_distribution.png)

**Top 10 gainers (by % change):**

| Card | Set | Feb price | Mar price | % change |
|---|---|---|---|---|
| Grimer | Legendary Collection | $0.52 | $42.81 | +8132.7% |
| Oricorio | Crown Zenith Galarian Gallery | $1.45 | $9.40 | +548.3% |
| Iron Moth | Paradox Rift | $7.47 | $26.89 | +260.0% |
| Charcadet | Obsidian Flames | $0.11 | $0.36 | +227.3% |
| Antique Root Fossil | Stellar Crown | $0.29 | $0.79 | +172.4% |
| Charcadet | Surging Sparks | $0.26 | $0.70 | +169.2% |
| Darmanitan | Obsidian Flames | $0.16 | $0.42 | +162.5% |
| Milotic ex | Surging Sparks | $1.28 | $3.33 | +160.2% |
| Palafin | Paldean Fates | $1.18 | $2.87 | +143.2% |
| Mimikyu δ | SWSH Black Star Promos | $1.98 | $4.66 | +135.4% |

**Top 10 decliners (by % change):**

| Card | Set | Feb price | Mar price | % change |
|---|---|---|---|---|
| Nidoran ♂ | Legendary Collection | $36.98 | $0.58 | -98.4% |
| Festival Grounds | Twilight Masquerade | $0.88 | $0.30 | -65.9% |
| Spearow | 151 | $2.49 | $1.17 | -53.0% |
| Bulbasaur | 151 | $0.49 | $0.24 | -51.0% |
| Monferno | Twilight Masquerade | $0.17 | $0.09 | -47.1% |
| Meowth | Shrouded Fable | $0.17 | $0.09 | -47.1% |
| Houndour | Obsidian Flames | $34.44 | $18.64 | -45.9% |
| Litwick | Twilight Masquerade | $0.25 | $0.14 | -44.0% |
| Rhydon | Legendary Collection | $79.99 | $44.98 | -43.8% |
| Sawk | Brilliant Stars | $0.23 | $0.13 | -43.5% |

![13_biggest_movers.png](13_biggest_movers.png)