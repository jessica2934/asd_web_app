import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy import stats
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
    final_df = pd.concat(
        [df_normal, df_asd], ignore_index=True
    )

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


# ==========================================================
# 3. HELPER: COHEN'S D EFFECT SIZE
#
# Effect size tells you HOW DIFFERENT the groups are,
# regardless of sample size.
#
# As Makin & Orban de Xivry (2019) recommend:
# "report effect sizes together with p-values in order
# to provide information about the magnitude of the
# effect" [3]
#
# Interpretation:
#   |d| < 0.2  = negligible
#   |d| 0.2-0.5 = small
#   |d| 0.5-0.8 = medium
#   |d| > 0.8   = large
# ==========================================================

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2)
        / (n1 + n2 - 2)
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


# ==========================================================
# 4. NORMALITY TEST (SHAPIRO-WILK)
#
# Chicco et al. (2025): "The Shapiro-Wilk test is
# generally preferred over the Kolmogorov-Smirnov test
# for assessing the normality of data" when working
# with "small samples (typically fewer than 50
# observations)" [7]
#
# This determines whether to use:
# - t-test (if both groups are normal)
# - Mann-Whitney U (if either group is non-normal)
# ==========================================================

print("\n" + "=" * 80)
print("  NORMALITY TEST (Shapiro-Wilk)")
print("=" * 80)
print(f"  {'Feature':<20} {'ASD p-val':>10} {'Normal p-val':>12}"
      f"  {'ASD Normal?':>12} {'Norm Normal?':>12}")
print(f"  {'-' * 20} {'-' * 10} {'-' * 12}"
      f"  {'-' * 12} {'-' * 12}")

normality_results = {}

for feat in FEATURES:
    asd_vals = asd_group[feat].dropna().values
    normal_vals = normal_group[feat].dropna().values

    if len(asd_vals) < 3 or len(normal_vals) < 3:
        normality_results[feat] = False
        continue

    _, p_asd = stats.shapiro(asd_vals)
    _, p_normal = stats.shapiro(normal_vals)

    asd_is_normal = p_asd > 0.05
    normal_is_normal = p_normal > 0.05
    both_normal = asd_is_normal and normal_is_normal

    normality_results[feat] = both_normal

    print(f"  {feat:<20} {p_asd:>10.4f} {p_normal:>12.4f}"
          f"  {'Yes' if asd_is_normal else 'No':>12}"
          f"  {'Yes' if normal_is_normal else 'No':>12}"
          f"  {'→ t-test' if both_normal else '→ Mann-Whitney'}")


# ==========================================================
# 5. STATISTICAL TESTS
#
# For each feature, we run:
# A) Welch's t-test (if both groups pass normality)
# B) Mann-Whitney U test (if either fails normality)
# C) BOTH tests for comparison
#
# Mann-Whitney U is the non-parametric alternative:
# "The Mann-Whitney U test is a nonparametric test that
# checks the difference between two independent samples...
# The data do not need to follow a normal distribution" [7]
#
# "Mann Whitney U test compares the means between two
# independent groups with the assumption that the data
# is not in a normal distribution" [8]
# ==========================================================

print("\n" + "=" * 80)
print(f"  STATISTICAL COMPARISON  |  ASD (n={n_asd}) "
      f"vs Normal (n={n_normal})")
print("=" * 80)

# Header
print(f"\n  {'Feature':<18} {'ASD':>12} {'Normal':>12}"
      f" {'t-test':>8} {'MW-U':>8} {'Cohen d':>8}"
      f" {'Effect':>10}  Sig")
print(f"  {'':>18} {'mean±SD':>12} {'mean±SD':>12}"
      f" {'p-val':>8} {'p-val':>8} {'':>8}"
      f" {'':>10}")
print(f"  {'-' * 18} {'-' * 12} {'-' * 12}"
      f" {'-' * 8} {'-' * 8} {'-' * 8}"
      f" {'-' * 10}  {'-' * 5}")

results = []

for feat in FEATURES:
    asd_vals = asd_group[feat].dropna().values
    normal_vals = normal_group[feat].dropna().values

    if len(asd_vals) < 2 or len(normal_vals) < 2:
        print(f"  {feat:<18}  skipped (not enough data)")
        continue

    # Welch's t-test (unequal variance)
    t_stat, t_pval = stats.ttest_ind(
        asd_vals, normal_vals, equal_var=False
    )

    # Mann-Whitney U test (non-parametric)
    u_stat, mw_pval = stats.mannwhitneyu(
        asd_vals, normal_vals, alternative='two-sided'
    )

    # Effect size
    d = cohens_d(asd_vals, normal_vals)
    effect_label = interpret_d(d)

    # Use the APPROPRIATE test based on normality
    both_normal = normality_results.get(feat, False)
    primary_p = t_pval if both_normal else mw_pval
    test_used = "t-test" if both_normal else "MW-U"

    # Significance markers
    if primary_p < 0.001:
        sig = "***"
    elif primary_p < 0.01:
        sig = "**"
    elif primary_p < 0.05:
        sig = "*"
    elif primary_p < 0.10:
        sig = "†"   # Trend
    else:
        sig = "ns"

    marker = " ←" if primary_p < 0.10 else ""

    asd_str = f"{asd_vals.mean():.1f}±{asd_vals.std():.1f}"
    norm_str = f"{normal_vals.mean():.1f}±{normal_vals.std():.1f}"

    print(f"  {feat:<18} {asd_str:>12} {norm_str:>12}"
          f" {t_pval:>8.4f} {mw_pval:>8.4f} {d:>+8.3f}"
          f" {effect_label:>10}  {sig}{marker}")

    results.append({
        "feature": feat,
        "asd_n": len(asd_vals),
        "asd_mean": round(asd_vals.mean(), 3),
        "asd_std": round(asd_vals.std(), 3),
        "normal_n": len(normal_vals),
        "normal_mean": round(normal_vals.mean(), 3),
        "normal_std": round(normal_vals.std(), 3),
        "t_stat": round(t_stat, 3),
        "t_pvalue": round(t_pval, 4),
        "u_stat": round(u_stat, 1),
        "mw_pvalue": round(mw_pval, 4),
        "cohens_d": round(d, 3),
        "effect_size": effect_label,
        "normality_ok": both_normal,
        "test_used": test_used,
        "primary_pvalue": round(primary_p, 4),
        "significant_005": primary_p < 0.05,
        "significant_010": primary_p < 0.10,
    })

print(f"\n  *** p<0.001  ** p<0.01  * p<0.05  "
      f"† p<0.10 (trend)  ns = not significant")
print(f"  Cohen's d: negligible<0.2, small<0.5, "
      f"medium<0.8, large≥0.8")
print(f"  ← indicates features selected by "
      f"appropriate test (p<0.10)")
print("=" * 80)


# ==========================================================
# 6. SUMMARY TABLE
# ==========================================================

results_df = pd.DataFrame(results).sort_values(
    "primary_pvalue"
).reset_index(drop=True)

print("\n" + "=" * 80)
print("  RANKED FEATURES (by primary p-value)")
print("=" * 80)

print(f"\n  {'Rank':<5} {'Feature':<18} {'Test':<7}"
      f" {'p-value':>8} {'Cohen d':>8}"
      f" {'Effect':>10}  {'Sig':>5}")
print(f"  {'-' * 5} {'-' * 18} {'-' * 7}"
      f" {'-' * 8} {'-' * 8}"
      f" {'-' * 10}  {'-' * 5}")

for i, row in results_df.iterrows():
    sig_mark = "***" if row['primary_pvalue'] < 0.001 \
        else "**" if row['primary_pvalue'] < 0.01 \
        else "*" if row['primary_pvalue'] < 0.05 \
        else "†" if row['primary_pvalue'] < 0.10 \
        else "ns"

    print(f"  {i + 1:<5} {row['feature']:<18}"
          f" {row['test_used']:<7}"
          f" {row['primary_pvalue']:>8.4f}"
          f" {row['cohens_d']:>+8.3f}"
          f" {row['effect_size']:>10}  {sig_mark}")


# ==========================================================
# 7. INTERPRETATION
# ==========================================================

sig_005 = results_df[
    results_df["significant_005"]
]["feature"].tolist()
sig_010 = results_df[
    results_df["significant_010"]
]["feature"].tolist()
trends = [f for f in sig_010 if f not in sig_005]

print("\n" + "=" * 80)
print("  INTERPRETATION")
print("=" * 80)

print(f"\n  Significant features (p < 0.05): "
      f"{sig_005 if sig_005 else 'None'}")
print(f"  Trend features (p < 0.10):       "
      f"{trends if trends else 'None'}")

# Show features with at least small effect size
small_effect = results_df[
    results_df["cohens_d"].abs() >= 0.2
].sort_values("cohens_d", key=abs, ascending=False)

if not small_effect.empty:
    print(f"\n  Features with at least SMALL effect "
          f"size (|d| ≥ 0.2):")
    for _, row in small_effect.iterrows():
        direction = ("ASD < Normal" if row['cohens_d'] < 0
                     else "ASD > Normal")
        print(f"    - {row['feature']:<18} d={row['cohens_d']:+.3f}"
              f" ({row['effect_size']}) [{direction}]")

# Explain the ML vs statistics discrepancy
print(f"""
  ─────────────────────────────────────────────────────
  WHY ML FOUND PATTERNS WHEN T-TESTS DID NOT
  ─────────────────────────────────────────────────────
  Statistical tests (t-test, Mann-Whitney U) ask:
    "Is there a significant difference in group MEANS?"

  Machine learning (Random Forest, SVM) asks:
    "Can I find ANY pattern — including nonlinear
     thresholds and feature interactions — that helps
     classify?"

  With only {n_asd} ASD cases and {n_normal} Normal cases:
  - Statistical power is LOW for detecting small effects
  - ML can exploit patterns that univariate tests miss
  - But ML results on small samples may also reflect
    overfitting rather than true biological differences

  IMPORTANT: Non-significant t-test results do NOT mean
  features are useless. They mean you lack sufficient
  evidence to claim a significant difference with this
  sample size. As Makin & Orban de Xivry (2019) state:
  "non-significant effects could literally mean very
  different things - a true null result, an underpowered
  genuine effect, or an ambiguous effect" [3]
  ─────────────────────────────────────────────────────""")


# ==========================================================
# 8. SAVE RESULTS
# ==========================================================

results_df.to_csv("statistical_test_results.csv", index=False)
print("\n✅ Saved: statistical_test_results.csv")