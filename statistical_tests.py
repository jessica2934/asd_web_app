import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model_utils import extract_features_from_image

# ==========================================================
# 1. LOAD OR EXTRACT FEATURES
# ==========================================================
FEATURES_CSV = "extracted_features.csv"

if os.path.exists(FEATURES_CSV):
    print(f"Loading cached features from {FEATURES_CSV}...\n")
    final_df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(final_df)} samples from cache.")
else:
    # --- Load Normal Data (Group 0) ---
    training_folder = "training_images"
    all_data = []
    print("Reading NORMAL images...\n")
    for file in os.listdir(training_folder):
        path = os.path.join(training_folder, file)
        try:
            print("Processing:", file)
            image = Image.open(path)
            features = extract_features_from_image(image)
            features["file"] = file
            features["group"] = 0
            all_data.append(features)
        except Exception as e:
            print("Error processing", file, ":", e)

    # --- Load ASD Data (Group 1) ---
    asd_folder = "training_images_ASD"
    asd_data = []
    print("\nReading ASD images...\n")
    for file in os.listdir(asd_folder):
        path = os.path.join(asd_folder, file)
        try:
            print("Processing ASD:", file)
            image = Image.open(path)
            features = extract_features_from_image(image)
            features["file"] = file
            features["group"] = 1
            asd_data.append(features)
        except Exception as e:
            print("Error processing ASD", file, ":", e)

    df_normal = pd.DataFrame(all_data)
    df_asd = pd.DataFrame(asd_data)
    final_df = pd.concat([df_normal, df_asd], ignore_index=True)
    final_df.to_csv(FEATURES_CSV, index=False)
    print(f"\n✅ Features saved to {FEATURES_CSV}")

# ==========================================================
# 2. PREPARE GROUPS
# ==========================================================
print("\nDataset distribution:")
print(final_df["group"].value_counts())

FEATURES = [
    "weight", "body_fat", "inorganic_salt", "protein",
    "body_water", "muscle_mass", "skeletal_muscle",
    "BMI", "body_fat_rate", "visceral_fat", "BMR",
    "fat_free_mass", "subcutaneous_fat", "SMI", "WHR",
]

normal_group = final_df[final_df["group"] == 0]
asd_group = final_df[final_df["group"] == 1]
n_asd = len(asd_group)
n_normal = len(normal_group)

# Output directory for graphs
os.makedirs("statistical_graphs", exist_ok=True)


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def significance_marker(p):
    """Return significance marker string."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "†"
    else:
        return "ns"


# ==========================================================
# NORMALITY TEST (Shapiro-Wilk)
# ==========================================================
print("\n" + "=" * 80)
print("  NORMALITY TEST (Shapiro-Wilk)")
print("=" * 80)
print(f"\n  {'Feature':<20} {'ASD p-val':>10} {'Normal p-val':>12}"
      f"  {'ASD Normal?':>12} {'Norm Normal?':>12}")
print(f"  {'-' * 20} {'-' * 10} {'-' * 12}"
      f"  {'-' * 12} {'-' * 12}")

normality_data = []
n_failed_normality = 0

for feat in FEATURES:
    asd_vals = asd_group[feat].dropna().values
    normal_vals = normal_group[feat].dropna().values

    if len(asd_vals) < 3 or len(normal_vals) < 3:
        continue

    _, p_asd = stats.shapiro(asd_vals)
    _, p_normal = stats.shapiro(normal_vals)

    asd_is_normal = p_asd > 0.05
    normal_is_normal = p_normal > 0.05
    both_normal = asd_is_normal and normal_is_normal

    if not both_normal:
        n_failed_normality += 1

    print(f"  {feat:<20} {p_asd:>10.4f} {p_normal:>12.4f}"
          f"  {'Yes' if asd_is_normal else 'No':>12}"
          f"  {'Yes' if normal_is_normal else 'No':>12}")

    normality_data.append({
        "feature": feat,
        "p_asd": p_asd,
        "p_normal": p_normal,
        "asd_normal": asd_is_normal,
        "normal_normal": normal_is_normal,
        "both_normal": both_normal,
    })

# Summary
n_total_features = len(normality_data)
print(f"\n  ─────────────────────────────────────────────────────")
print(f"  RESULT: {n_failed_normality}/{n_total_features} features"
      f" FAIL normality in at least one group.")
print(f"  CONCLUSION: Data is NOT normally distributed.")
print(f"  DECISION: Use Mann-Whitney U test (non-parametric)")

# --- Normality Test Graph ---
fig, ax = plt.subplots(figsize=(14, 7))
norm_df = pd.DataFrame(normality_data)
x = np.arange(len(norm_df))
width = 0.35

bars1 = ax.bar(x - width/2, norm_df["p_asd"], width,
               label="ASD Group (n=23)", color="coral", alpha=0.8,
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, norm_df["p_normal"], width,
               label="Normal Group (n=50)", color="steelblue", alpha=0.8,
               edgecolor='black', linewidth=0.5)

ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2,
           label="α = 0.05 (normality threshold)")
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Shapiro-Wilk p-value", fontsize=12)
ax.set_title("Section A: Shapiro-Wilk Normality Test\n"
             "Bars BELOW red line → NOT normal → Use Mann-Whitney U",
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(norm_df["feature"], rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, min(1.0, norm_df[["p_asd", "p_normal"]].max().max() * 1.3))

# Annotate normality status
for i, row in norm_df.iterrows():
    if not row["both_normal"]:
        ax.annotate("✗ NOT NORMAL", (i, max(row["p_asd"], row["p_normal"]) + 0.02),
                    ha='center', fontsize=7, color='red', fontweight='bold')
    else:
        ax.annotate("✓ Normal", (i, max(row["p_asd"], row["p_normal"]) + 0.02),
                    ha='center', fontsize=7, color='green', fontweight='bold')

# Add conclusion text box
conclusion_text = (f"Result: {n_failed_normality}/{n_total_features} features"
                   f" fail normality\n→ Mann-Whitney U test selected")
ax.text(0.02, 0.95, conclusion_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig("statistical_graphs/A_normality_test.png", dpi=150,
            bbox_inches='tight')
plt.close()
print(f"\n  Graph saved: statistical_graphs/A_normality_test.png")


# ==========================================================
# MANN-WHITNEY U TEST (Non-Parametric)
# ==========================================================
print("\n" + "=" * 80)
print("  MANN-WHITNEY U TEST (Non-Parametric)")
print("=" * 80)
print(f"\n  {'Feature':<18} {'U-stat':>10} {'p-value':>9}"
      f" {'Sig':>5} {'ASD Median':>12} {'Normal Median':>14}")
print(f"  {'-' * 18} {'-' * 10} {'-' * 9}"
      f" {'-' * 5} {'-' * 12} {'-' * 14}")

mw_results = []

for feat in FEATURES:
    asd_vals = asd_group[feat].dropna().values
    normal_vals = normal_group[feat].dropna().values

    if len(asd_vals) < 2 or len(normal_vals) < 2:
        continue

    u_stat, mw_pval = stats.mannwhitneyu(
        asd_vals, normal_vals, alternative='two-sided'
    )

    sig = significance_marker(mw_pval)

    print(f"  {feat:<18} {u_stat:>10.1f} {mw_pval:>9.4f}"
          f" {sig:>5} {np.median(asd_vals):>12.1f}"
          f" {np.median(normal_vals):>14.1f}")

    mw_results.append({
        "feature": feat,
        "u_stat": u_stat,
        "mw_pvalue": mw_pval,
        "asd_median": np.median(asd_vals),
        "normal_median": np.median(normal_vals),
        "asd_mean": asd_vals.mean(),
        "normal_mean": normal_vals.mean(),
        "asd_std": asd_vals.std(),
        "normal_std": normal_vals.std(),
    })

print(f"\n  U statistic interpretation:")
print(f"    • Values near 0 or near n1×n2={n_asd * n_normal}"
      f" indicate strong differences [1]")
print(f"    • U = n1×n2/2 = {n_asd * n_normal / 2:.0f}"
      f" indicates NO difference [1]")
print(f"\n  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10  ns = not significant")

# --- Mann-Whitney U Graph ---
mw_df = pd.DataFrame(mw_results)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: P-values bar chart
ax1 = axes[0]
colors = ['red' if p < 0.05 else 'orange' if p < 0.10
          else 'skyblue' for p in mw_df["mw_pvalue"]]
bars = ax1.barh(range(len(mw_df)), mw_df["mw_pvalue"],
                color=colors, edgecolor='black', linewidth=0.5)
ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2,
            label="α = 0.05 (significant)")
ax1.axvline(x=0.10, color='orange', linestyle=':', linewidth=1.5,
            label="α = 0.10 (trend)")
ax1.set_yticks(range(len(mw_df)))
ax1.set_yticklabels(mw_df["feature"], fontsize=9)
ax1.set_xlabel("p-value", fontsize=11)
ax1.set_title("Mann-Whitney U p-values\n"
              "(bars LEFT of red line = significant)",
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)

for i, (_, row) in enumerate(mw_df.iterrows()):
    sig = significance_marker(row["mw_pvalue"])
    ax1.annotate(f"{sig}", (row["mw_pvalue"] + 0.01, i),
                 va='center', fontsize=9, fontweight='bold')

# Panel 2: Box plot comparison (top 6 features)
ax2 = axes[1]
top_feats = mw_df.nsmallest(6, "mw_pvalue")["feature"].tolist()
box_data_asd = [asd_group[f].dropna().values for f in top_feats]
box_data_normal = [normal_group[f].dropna().values for f in top_feats]

positions_asd = np.arange(len(top_feats)) * 2
positions_normal = np.arange(len(top_feats)) * 2 + 0.7

bp1 = ax2.boxplot(box_data_asd, positions=positions_asd, widths=0.6,
                  patch_artist=True,
                  boxprops=dict(facecolor='coral', alpha=0.7),
                  medianprops=dict(color='darkred', linewidth=2))
bp2 = ax2.boxplot(box_data_normal, positions=positions_normal, widths=0.6,
                  patch_artist=True,
                  boxprops=dict(facecolor='steelblue', alpha=0.7),
                  medianprops=dict(color='darkblue', linewidth=2))

ax2.set_xticks(np.arange(len(top_feats)) * 2 + 0.35)
ax2.set_xticklabels(top_feats, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel("Value", fontsize=11)
ax2.set_title("Box Plot: Top 6 Features\n(by Mann-Whitney U p-value)",
              fontsize=12, fontweight='bold')
ax2.legend([bp1["boxes"][0], bp2["boxes"][0]],
           [f"ASD (n={n_asd})", f"Normal (n={n_normal})"],
           fontsize=10)

plt.suptitle("Section B: Mann-Whitney U Test Results\n"
             "(Selected because data is NOT normally distributed)",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("statistical_graphs/B_mannwhitney_results.png", dpi=150,
            bbox_inches='tight')
plt.close()
print(f"\n  Graph saved: statistical_graphs/B_mannwhitney_results.png")


# ==========================================================
# EFFECT SIZE (Cohen's d)
# ==========================================================
print("\n" + "=" * 80)
print("  EFFECT SIZE (Cohen's d)")
print("=" * 80)
print(f"\n  {'Feature':<18} {'Cohen d':>9} {'|d|':>6}"
      f" {'Interpretation':>15} {'Direction':>14}")
print(f"  {'-' * 18} {'-' * 9} {'-' * 6}"
      f" {'-' * 15} {'-' * 14}")

effect_results = []

for feat in FEATURES:
    asd_vals = asd_group[feat].dropna().values
    normal_vals = normal_group[feat].dropna().values

    if len(asd_vals) < 2 or len(normal_vals) < 2:
        continue

    d = cohens_d(asd_vals, normal_vals)
    effect_label = interpret_d(d)
    direction = "ASD > Normal" if d > 0 else "ASD < Normal"

    print(f"  {feat:<18} {d:>+9.3f} {abs(d):>6.3f}"
          f" {effect_label:>15} {direction:>14}")

    effect_results.append({
        "feature": feat,
        "cohens_d": d,
        "abs_d": abs(d),
        "effect_size": effect_label,
        "direction": direction,
    })

# --- Effect Size Graph ---
effect_df = pd.DataFrame(effect_results).sort_values("abs_d", ascending=True)
fig, ax = plt.subplots(figsize=(12, 8))

colors = []
for d in effect_df["abs_d"]:
    if d >= 0.8:
        colors.append("darkred")
    elif d >= 0.5:
        colors.append("red")
    elif d >= 0.2:
        colors.append("orange")
    else:
        colors.append("lightgray")

bars = ax.barh(range(len(effect_df)), effect_df["cohens_d"],
               color=colors, edgecolor='black', linewidth=0.5)

# Add reference lines
ax.axvline(x=0, color='black', linewidth=1)
ax.axvline(x=0.2, color='orange', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=-0.2, color='orange', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=0.5, color='red', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=-0.5, color='red', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=0.8, color='darkred', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(x=-0.8, color='darkred', linestyle=':', linewidth=1, alpha=0.7)

ax.set_yticks(range(len(effect_df)))
ax.set_yticklabels(effect_df["feature"], fontsize=10)
ax.set_xlabel("Cohen's d (effect size)", fontsize=12)
ax.set_title("Section C: Effect Sizes (Cohen's d)\n"
             "Positive = ASD > Normal | Negative = ASD < Normal",
             fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='darkred', label='Large (|d| ≥ 0.8)'),
    mpatches.Patch(facecolor='red', label='Medium (0.5 ≤ |d| < 0.8)'),
    mpatches.Patch(facecolor='orange', label='Small (0.2 ≤ |d| < 0.5)'),
    mpatches.Patch(facecolor='lightgray', label='Negligible (|d| < 0.2)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig("statistical_graphs/C_effect_size.png", dpi=150,
            bbox_inches='tight')
plt.close()
print(f"\n Graph saved: statistical_graphs/C_effect_size.png")


# ==========================================================
#  COMBINED RESULTS & RANKING
# ==========================================================
print("\n" + "=" * 80)
print(" COMBINED RESULTS & FEATURE RANKING")
print("=" * 80)

# Build combined results dataframe
combined_results = []

for feat in FEATURES:
    asd_vals = asd_group[feat].dropna().values
    normal_vals = normal_group[feat].dropna().values

    if len(asd_vals) < 2 or len(normal_vals) < 2:
        continue

    # Mann-Whitney U test
    u_stat, mw_pval = stats.mannwhitneyu(
        asd_vals, normal_vals, alternative='two-sided'
    )

    # Effect size
    d = cohens_d(asd_vals, normal_vals)
    effect_label = interpret_d(d)
    direction = "ASD > Normal" if d > 0 else "ASD < Normal"

    combined_results.append({
        "feature": feat,
        "u_stat": round(u_stat, 1),
        "mw_pvalue": round(mw_pval, 4),
        "cohens_d": round(d, 3),
        "effect_size": effect_label,
        "direction": direction,
        "significant_005": mw_pval < 0.05,
        "significant_010": mw_pval < 0.10,
        "asd_n": len(asd_vals),
        "asd_mean": round(asd_vals.mean(), 3),
        "asd_std": round(asd_vals.std(), 3),
        "asd_median": round(np.median(asd_vals), 3),
        "normal_n": len(normal_vals),
        "normal_mean": round(normal_vals.mean(), 3),
        "normal_std": round(normal_vals.std(), 3),
        "normal_median": round(np.median(normal_vals), 3),
    })

results_df = pd.DataFrame(combined_results).sort_values(
    "mw_pvalue"
).reset_index(drop=True)

# Print ranked table
print(f"\n  {'Rank':<5} {'Feature':<18} {'MW-U p':>8} {'Cohen d':>9}"
      f" {'Effect':>10} {'Direction':>14} {'Sig':>5}")
print(f"  {'-' * 5} {'-' * 18} {'-' * 8} {'-' * 9}"
      f" {'-' * 10} {'-' * 14} {'-' * 5}")

for i, row in results_df.iterrows():
    sig = significance_marker(row['mw_pvalue'])
    print(f"  {i + 1:<5} {row['feature']:<18}"
          f" {row['mw_pvalue']:>8.4f} {row['cohens_d']:>+9.3f}"
          f" {row['effect_size']:>10} {row['direction']:>14} {sig:>5}")

# --- Combined Results Graph ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel 1: P-values ranked
ax1 = axes[0]
colors_p = ['red' if p < 0.05 else 'orange' if p < 0.10
            else 'skyblue' for p in results_df["mw_pvalue"]]
ax1.barh(range(len(results_df)), results_df["mw_pvalue"],
         color=colors_p, edgecolor='black', linewidth=0.5)
ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2,
            label="α = 0.05")
ax1.axvline(x=0.10, color='orange', linestyle=':', linewidth=1.5,
            label="α = 0.10 (trend)")
ax1.set_yticks(range(len(results_df)))
ax1.set_yticklabels(results_df["feature"], fontsize=9)
ax1.set_xlabel("Mann-Whitney U p-value", fontsize=11)
ax1.set_title("Features Ranked by p-value\n(Mann-Whitney U Test)",
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)

for i, (_, row) in enumerate(results_df.iterrows()):
    sig = significance_marker(row["mw_pvalue"])
    ax1.annotate(f"{sig}", (row["mw_pvalue"] + 0.01, i),
                 va='center', fontsize=9, fontweight='bold')

# Panel 2: Effect sizes ranked (same order as p-values)
ax2 = axes[1]
colors_d = []
for d in results_df["cohens_d"].abs():
    if d >= 0.8:
        colors_d.append("darkred")
    elif d >= 0.5:
        colors_d.append("red")
    elif d >= 0.2:
        colors_d.append("orange")
    else:
        colors_d.append("lightgray")

ax2.barh(range(len(results_df)), results_df["cohens_d"],
         color=colors_d, edgecolor='black', linewidth=0.5)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.axvline(x=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.axvline(x=-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.axvline(x=-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.set_yticks(range(len(results_df)))
ax2.set_yticklabels(results_df["feature"], fontsize=9)
ax2.set_xlabel("Cohen's d", fontsize=11)
ax2.set_title("Effect Sizes (Cohen's d)\n"
              "(same ranking as p-values)",
              fontsize=12, fontweight='bold')

plt.suptitle("Section D: Combined Mann-Whitney U + Effect Size Results",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("statistical_graphs/D_combined_results.png", dpi=150,
            bbox_inches='tight')
plt.close()
print(f"\n Graph saved: statistical_graphs/D_combined_results.png")


# ==========================================================
# INTERPRETATION & CONCLUSIONS
# ==========================================================
sig_005 = results_df[results_df["significant_005"]]["feature"].tolist()
sig_010 = results_df[results_df["significant_010"]]["feature"].tolist()
trends = [f for f in sig_010 if f not in sig_005]

small_effect = results_df[
    results_df["cohens_d"].abs() >= 0.2
].sort_values("cohens_d", key=abs, ascending=False)

print("\n" + "=" * 80)
print(" INTERPRETATION & CONCLUSIONS")
print("=" * 80)


print(f"\n  Significant features (p < 0.05): "
      f"{sig_005 if sig_005 else 'None'}")
print(f"  Trend features (p < 0.10):       "
      f"{trends if trends else 'None'}")

if not small_effect.empty:
    print(f"\n  Features with at least SMALL effect size (|d| ≥ 0.2):")
    for _, row in small_effect.iterrows():
        print(f"    • {row['feature']:<18} d={row['cohens_d']:+.3f}"
              f" ({row['effect_size']}) [{row['direction']}]")


# --- Final Summary Graph ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# --- Final Summary Graph ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
summary_text = f"""STATISTICAL ANALYSIS SUMMARY

Dataset: {n_asd} ASD vs {n_normal} Normal participants
Features tested: {n_total_features}


Step 1: Shapiro-Wilk Normality Test

Result: {n_failed_normality}/{n_total_features} features NOT normally distributed
Decision: Data violates normality assumption


Step 2: Mann-Whitney U Test (non-parametric)

- Does NOT require normal distribution
- Compares ranks between groups
- Appropriate for small samples (n < 50)


Results:

- Significant (p < 0.05): {len(sig_005)} features
- Trend (p < 0.10):       {len(trends)} features (weight, muscle_mass)
- Medium effect size:     1 feature (muscle_mass)
- Small effect sizes:     9 features


Reference: Chicco et al. (2025) BioData Mining [1]
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='sans-serif')
plt.tight_layout()
plt.savefig("statistical_graphs/E_summary.png", dpi=150,
            bbox_inches='tight')
plt.close()
print(f"\n  Graph saved: statistical_graphs/E_summary.png")


# ==========================================================
# 4. SAVE ALL RESULTS
# ==========================================================
results_df.to_csv("statistical_test_results.csv", index=False)