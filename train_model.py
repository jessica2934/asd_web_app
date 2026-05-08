import os
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
from model_utils import extract_features_from_image
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_curve, auc,
    classification_report, roc_auc_score,
    balanced_accuracy_score
)
from mlxtend.feature_selection import (
    ExhaustiveFeatureSelector as EFS
)

# =========================
# 1. LOAD NORMAL DATA (GROUP 0)
# =========================
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

df = pd.DataFrame(all_data)

# =========================
# 2. LOAD ASD DATA (GROUP 1)
# =========================
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

asd_df = pd.DataFrame(asd_data)

# =========================
# COMBINE DATA
# =========================
final_df = pd.concat([df, asd_df], ignore_index=True)

print("\nDataset distribution:")
print(final_df["group"].value_counts())

# =========================
# 3. PREPARE DATA
# =========================
X = final_df.drop(columns=["file", "group"], errors="ignore")
y = final_df["group"].astype(int)

feature_names = X.columns.tolist()
n_positive = int(y.sum())
n_negative = int(len(y) - y.sum())
n_total = len(y)
n_features = X.shape[1]

# Allow 3 to 5 features
min_features = 3
max_features = 5

total_combos = sum(
    comb(n_features, k)
    for k in range(min_features, max_features + 1)
)

print(f"\nTotal samples: {n_total}")
print(f"Positive cases (ASD): {n_positive}")
print(f"Negative cases (Normal): {n_negative}")
print(f"Number of features: {n_features}")
print(f"Feature range: {min_features} to {max_features}")
print(f"Total combinations to evaluate: {total_combos}")
print(f"Imbalance ratio: {n_negative / n_positive:.2f}:1")

# Save feature means for imputation
feature_means = X.mean()
joblib.dump(feature_means, "feature_means.pkl")
X = X.fillna(feature_means)


# ==========================================================
# 4. DEFINE MODELS WITH HYPERPARAMETER GRIDS
#
# GridSearchCV "exhaustively generates candidates from a
# grid of parameter values specified with the param_grid
# parameter" [1]
#
# "It is possible and recommended to search the
# hyper-parameter space for the best cross validation
# score." [1]
# ==========================================================

def get_model_configs():
    """
    Define model pipelines and their hyperparameter grids.

    scikit-learn recommends: "a small subset of those
    parameters can have a large impact on the predictive
    or computation performance of the model while others
    can be left to their default values" [1]
    """

    configs = {}

    # --- Firth-approximated Logistic Regression ---
    configs['Firth LR'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs',
                max_iter=10000,
                class_weight='balanced'
            ))
        ]),
        'param_grid': {
            'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    }

    # --- SVM with RBF kernel ---
    # "It is highly recommended to scale your data" [1]
    configs['SVM (RBF)'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ]),
        'param_grid': {
            'clf__C': [0.1, 1.0, 10.0, 100.0],
            'clf__gamma': ['scale', 0.01, 0.1, 1.0]
        }
    }

    # --- Random Forest ---
    configs['Random Forest'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500,
                class_weight=None,
                random_state=42
            ))
        ]),
        'param_grid': {
            'clf__max_depth': [2, 3, 5, None],
            'clf__min_samples_leaf': [3, 5, 10],
            'clf__min_samples_split': [2, 5, 10]
        }
    }

    return configs


# ==========================================================
# 5. EXHAUSTIVE FEATURE SEARCH (3 to 5 features)
# ==========================================================

print("\n" + "=" * 60)
print(f"EXHAUSTIVE FEATURE SEARCH "
      f"({min_features} to {max_features} features)")
print(f"Evaluating {total_combos} combinations...")
print("=" * 60)

scaler_search = StandardScaler()
X_scaled = pd.DataFrame(
    scaler_search.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Use Firth LR for feature search (fast and stable)
search_clf = LogisticRegression(
    solver='lbfgs',
    max_iter=10000,
    class_weight='balanced',
    C=0.1
)

efs = EFS(
    search_clf,
    min_features=min_features,
    max_features=max_features,
    scoring='roc_auc',
    cv=StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    ),
    print_progress=True,
    n_jobs=-1
)

efs = efs.fit(X_scaled, y)

print(f"\nBest AUC score: {efs.best_score_:.3f}")
print(f"Best features: {efs.best_feature_names_}")

# Show top 10
results_df = pd.DataFrame.from_dict(
    efs.get_metric_dict()
).T
results_df.sort_values(
    'avg_score', ascending=False, inplace=True
)

print("\n===== TOP 10 FEATURE COMBINATIONS BY AUC =====")
for i, (idx, row) in enumerate(
    results_df.head(10).iterrows()
):
    print(f"  {i+1}. {row['feature_names']} -> "
          f"AUC = {row['avg_score']:.3f} "
          f"± {row['std_dev']:.3f}")

selected_features = list(efs.best_feature_names_)
print(f"\nSelected features ({len(selected_features)}): "
      f"{selected_features}")
joblib.dump(selected_features, "selected_features.pkl")


# ==========================================================
# 6. HYPERPARAMETER TUNING WITH GridSearchCV
#
# "GridSearchCV exhaustively generates candidates from a
# grid of parameter values... the best combination is
# retained." [1]
#
# "For some applications, other scoring functions are
# better suited (for example in unbalanced classification,
# the accuracy score is often uninformative)" [1]
#
# We use StratifiedKFold because it "returns stratified
# folds: each set contains approximately the same
# percentage of samples of each target class" [3]
# ==========================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

X_selected = X[selected_features]
model_configs = get_model_configs()
tuned_models = {}

inner_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

for model_name, config in model_configs.items():
    print(f"\n--- Tuning {model_name} ---")

    grid_search = GridSearchCV(
        estimator=config['pipeline'],
        param_grid=config['param_grid'],
        scoring='roc_auc',
        cv=inner_cv,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_selected, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    tuned_models[model_name] = {
        'grid_search': grid_search,
        'best_params': best_params,
        'best_cv_auc': best_score,
        'pipeline': grid_search.best_estimator_
    }

    print(f"  Best params: {best_params}")
    print(f"  Best CV AUC: {best_score:.3f}")

    # Show top 3 parameter combinations
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values(
        'rank_test_score'
    )
    print(f"  Top 3 configurations:")
    for j, (_, row) in enumerate(
        cv_results.head(3).iterrows()
    ):
        print(f"    {j+1}. {row['params']} -> "
              f"AUC={row['mean_test_score']:.3f} "
              f"± {row['std_test_score']:.3f}")


# ==========================================================
# 7. LOOCV EVALUATION WITH TUNED HYPERPARAMETERS
#
# "LeaveOneOut is a simple cross-validation. Each learning
# set is created by taking all the samples except one" [3]
# ==========================================================

print("\n" + "=" * 60)
print("LEAVE-ONE-OUT CROSS-VALIDATION (Tuned Models)")
print("=" * 60)

loo = LeaveOneOut()
all_results = {}

for model_name, tuned in tuned_models.items():
    print(f"\n--- {model_name} "
          f"(params: {tuned['best_params']}) ---")

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    # Rebuild pipeline with best params for each LOO fold
    best_pipeline = tuned['pipeline']

    for train_idx, test_idx in loo.split(X_selected):
        X_tr = X_selected.iloc[train_idx]
        X_te = X_selected.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        # Clone the pipeline with best hyperparameters
        from sklearn.base import clone
        pipe = clone(best_pipeline)
        pipe.fit(X_tr, y_tr)

        y_true_all.append(y_te.values[0])
        y_pred_all.append(pipe.predict(X_te)[0])
        y_prob_all.append(
            pipe.predict_proba(X_te)[0, 1]
        )

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    loocv_acc = accuracy_score(y_true_all, y_pred_all)
    loocv_bal_acc = balanced_accuracy_score(
        y_true_all, y_pred_all
    )
    loocv_auc = roc_auc_score(y_true_all, y_prob_all)

    cm = confusion_matrix(y_true_all, y_pred_all)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens = spec = tn = fp = fn = tp = 0

    all_results[model_name] = {
        'accuracy': loocv_acc,
        'balanced_accuracy': loocv_bal_acc,
        'auc': loocv_auc,
        'sensitivity': sens,
        'specificity': spec,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_prob': y_prob_all,
        'cm': cm,
        'best_params': tuned['best_params'],
    }

    print(f"  Accuracy:          {loocv_acc:.3f}")
    print(f"  Balanced Accuracy: {loocv_bal_acc:.3f}")
    print(f"  AUC:               {loocv_auc:.3f}")
    print(f"  Sensitivity:       {sens:.3f}")
    print(f"  Specificity:       {spec:.3f}")


# ==========================================================
# 8. MODEL COMPARISON TABLE
# ==========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON (LOOCV with Tuned Hyperparameters)")
print("=" * 60)

print(f"\n{'Model':<18} {'Acc':>6} {'BalAcc':>8} "
      f"{'AUC':>6} {'Sens':>6} {'Spec':>6}")
print("-" * 56)

for name, res in all_results.items():
    print(f"{name:<18} "
          f"{res['accuracy']:>6.3f} "
          f"{res['balanced_accuracy']:>8.3f} "
          f"{res['auc']:>6.3f} "
          f"{res['sensitivity']:>6.3f} "
          f"{res['specificity']:>6.3f}")

best_name = max(
    all_results, key=lambda k: all_results[k]['auc']
)
best_result = all_results[best_name]

print(f"\n✅ Best model: {best_name} "
      f"(AUC = {best_result['auc']:.3f})")
print(f"   Best params: {best_result['best_params']}")


# ==========================================================
# 9. ABLATION STUDY
#
# An ablation study systematically removes components
# to measure their individual contribution.
#
# We perform three types of ablation:
#
# A) Feature ablation: Remove one feature at a time
#    from the selected set to measure each feature's
#    contribution to the best model's performance.
#
# B) Hyperparameter ablation: Compare tuned vs default
#    hyperparameters to measure the impact of tuning.
#
# C) Model component ablation: Compare with/without
#    scaling and with/without class balancing.
# ==========================================================

print("\n" + "=" * 60)
print("ABLATION STUDY")
print("=" * 60)

best_create_pipeline = model_configs[best_name]
best_tuned_pipeline = tuned_models[best_name]['pipeline']


# --- A) FEATURE ABLATION ---
print("\n--- A) Feature Ablation (Leave-One-Feature-Out) ---")
print(f"    Base features: {selected_features}")
print(f"    Base AUC: {best_result['auc']:.3f}\n")

feature_ablation_results = []

for remove_feat in selected_features:
    # Create subset without this feature
    ablated_features = [
        f for f in selected_features if f != remove_feat
    ]
    X_ablated = X[ablated_features]

    y_true_abl = []
    y_prob_abl = []

    for train_idx, test_idx in loo.split(X_ablated):
        X_tr = X_ablated.iloc[train_idx]
        X_te = X_ablated.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        pipe = clone(best_tuned_pipeline)
        pipe.fit(X_tr, y_tr)

        y_true_abl.append(y_te.values[0])
        y_prob_abl.append(
            pipe.predict_proba(X_te)[0, 1]
        )

    abl_auc = roc_auc_score(
        np.array(y_true_abl), np.array(y_prob_abl)
    )
    auc_drop = best_result['auc'] - abl_auc

    feature_ablation_results.append({
        'removed_feature': remove_feat,
        'remaining_features': ablated_features,
        'auc_without': abl_auc,
        'auc_drop': auc_drop,
    })

    importance = "CRITICAL" if auc_drop > 0.05 \
        else "Important" if auc_drop > 0.02 \
        else "Minor" if auc_drop > 0 \
        else "Redundant"

    print(f"  Remove '{remove_feat}': "
          f"AUC={abl_auc:.3f} "
          f"(Δ={auc_drop:+.3f}) [{importance}]")

# Sort by importance
feature_ablation_results.sort(
    key=lambda x: x['auc_drop'], reverse=True
)
print(f"\n  Most important feature: "
      f"'{feature_ablation_results[0]['removed_feature']}' "
      f"(removing it drops AUC by "
      f"{feature_ablation_results[0]['auc_drop']:.3f})")


# --- B) HYPERPARAMETER ABLATION ---
print("\n--- B) Hyperparameter Ablation "
      "(Tuned vs Default) ---")

# Default (untuned) pipeline
default_configs = {
    'Firth LR': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='lbfgs', max_iter=10000,
            class_weight='balanced', C=1.0
        ))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced',
            probability=True, random_state=42
        ))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=500, max_depth=3,
            min_samples_leaf=5,
            class_weight=None, random_state=42
        ))
    ]),
}

# Run LOOCV with default params for best model
default_pipe = default_configs[best_name]
y_true_def = []
y_prob_def = []

for train_idx, test_idx in loo.split(X_selected):
    X_tr = X_selected.iloc[train_idx]
    X_te = X_selected.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]

    pipe = clone(default_pipe)
    pipe.fit(X_tr, y_tr)
    y_true_def.append(y_te.values[0])
    y_prob_def.append(pipe.predict_proba(X_te)[0, 1])

default_auc = roc_auc_score(
    np.array(y_true_def), np.array(y_prob_def)
)
tuning_improvement = best_result['auc'] - default_auc

print(f"  {best_name} with DEFAULT params: "
      f"AUC={default_auc:.3f}")
print(f"  {best_name} with TUNED params:   "
      f"AUC={best_result['auc']:.3f}")
print(f"  Improvement from tuning: "
      f"{tuning_improvement:+.3f}")

if tuning_improvement > 0.02:
    print("  → Hyperparameter tuning provided "
          "meaningful improvement")
elif tuning_improvement > 0:
    print("  → Hyperparameter tuning provided "
          "marginal improvement")
else:
    print("  → Hyperparameter tuning did not improve "
          "performance (default params may be sufficient)")


# --- C) MODEL COMPONENT ABLATION ---
print("\n--- C) Component Ablation ---")

# Test without scaling
print("  Testing without StandardScaler...")
clf_only = clone(
    best_tuned_pipeline.named_steps['clf']
)
no_scaler_pipe = Pipeline([
    ('clf', clf_only)
])

y_true_ns = []
y_prob_ns = []

for train_idx, test_idx in loo.split(X_selected):
    X_tr = X_selected.iloc[train_idx]
    X_te = X_selected.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]

    pipe = clone(no_scaler_pipe)
    pipe.fit(X_tr, y_tr)
    y_true_ns.append(y_te.values[0])
    y_prob_ns.append(pipe.predict_proba(X_te)[0, 1])

no_scaler_auc = roc_auc_score(
    np.array(y_true_ns), np.array(y_prob_ns)
)
scaler_impact = best_result['auc'] - no_scaler_auc

print(f"    With scaler:    AUC={best_result['auc']:.3f}")
print(f"    Without scaler: AUC={no_scaler_auc:.3f}")
print(f"    Scaler impact:  {scaler_impact:+.3f}")


# ==========================================================
# 10. ABLATION SUMMARY TABLE
# ==========================================================

print("\n" + "=" * 60)
print("ABLATION SUMMARY")
print("=" * 60)

print(f"\n{'Component':<35} {'AUC':>6} {'Impact':>8}")
print("-" * 52)
print(f"{'Full model (tuned)':<35} "
      f"{best_result['auc']:>6.3f} {'baseline':>8}")
print(f"{'Default hyperparameters':<35} "
      f"{default_auc:>6.3f} "
      f"{-tuning_improvement:>+8.3f}")
print(f"{'Without scaler':<35} "
      f"{no_scaler_auc:>6.3f} "
      f"{-scaler_impact:>+8.3f}")

for abl in feature_ablation_results:
    label = f"Without '{abl['removed_feature']}'"
    print(f"{label:<35} "
          f"{abl['auc_without']:>6.3f} "
          f"{-abl['auc_drop']:>+8.3f}")


# ==========================================================
# 11. 5-FOLD CV FOR BEST MODEL
#    Shows ONLY Training and Testing Accuracy
# ==========================================================

print("\n" + "=" * 60)
print(f"5-FOLD CV — {best_name} (Tuned)")
print("=" * 60)

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

train_acc_list = []
val_acc_list = []

for fold, (train_idx, val_idx) in enumerate(
    outer_cv.split(X_selected, y), 1
):
    X_tr = X_selected.iloc[train_idx]
    X_val = X_selected.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    fold_pipe = clone(best_tuned_pipeline)
    fold_pipe.fit(X_tr, y_tr)

    train_pred = fold_pipe.predict(X_tr)
    train_acc_list.append(accuracy_score(y_tr, train_pred))

    val_pred = fold_pipe.predict(X_val)
    val_acc_list.append(accuracy_score(y_val, val_pred))

    print(f"Fold {fold}: "
          f"Training Acc={train_acc_list[-1]:.3f}, "
          f"Testing Acc={val_acc_list[-1]:.3f}")

print(f"\nTraining Accuracy:  {np.mean(train_acc_list):.3f} "
      f"± {np.std(train_acc_list):.3f}")
print(f"Testing Accuracy:   {np.mean(val_acc_list):.3f} "
      f"± {np.std(val_acc_list):.3f}")

overfit_gap = np.mean(train_acc_list) - np.mean(val_acc_list)
print(f"\nOverfit gap: {overfit_gap:.3f}")
if overfit_gap > 0.10:
    print("⚠️  WARNING: Possible overfitting detected!")
elif overfit_gap > 0.05:
    print("⚠️  CAUTION: Moderate overfit gap.")
else:
    print("✅ Overfit gap is acceptable.")


# ==========================================================
# 12. PLOTS
# ==========================================================

# --- 5-Fold CV: Training vs Testing Accuracy ---
plt.figure(figsize=(10, 5))
folds = range(1, 6)
plt.plot(folds, train_acc_list, 'o-', color='blue',
         linewidth=2, markersize=8,
         label="Training Accuracy")
plt.plot(folds, val_acc_list, 's-', color='red',
         linewidth=2, markersize=8,
         label="Testing Accuracy")
plt.axhline(y=np.mean(train_acc_list), color='blue',
            linestyle='--', alpha=0.5,
            label=f"Mean Train: "
                  f"{np.mean(train_acc_list):.3f}")
plt.axhline(y=np.mean(val_acc_list), color='red',
            linestyle='--', alpha=0.5,
            label=f"Mean Test: "
                  f"{np.mean(val_acc_list):.3f}")
plt.title(f"5-Fold Cross Validation — {best_name} (Tuned)\n"
          f"Training vs Testing Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(folds)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cv_performance.png", dpi=150)
plt.show()

# --- Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['accuracy', 'balanced_accuracy', 'auc',
           'sensitivity', 'specificity']
metric_labels = ['Accuracy', 'Bal. Acc', 'AUC',
                 'Sensitivity', 'Specificity']
x_pos = np.arange(len(metrics))
width = 0.25

for i, (name, res) in enumerate(all_results.items()):
    values = [res[m] for m in metrics]
    ax.bar(x_pos + i * width, values, width, label=name)

ax.set_ylabel('Score')
ax.set_title('Model Comparison (LOOCV — Tuned Hyperparameters)')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(metric_labels)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# --- ROC Curves ---
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, (name, res) in enumerate(all_results.items()):
    fpr_curve, tpr_curve, _ = roc_curve(
        res['y_true'], res['y_prob']
    )
    roc_auc_val = auc(fpr_curve, tpr_curve)
    plt.plot(fpr_curve, tpr_curve, color=colors[i], lw=2,
             label=f"{name} (AUC={roc_auc_val:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1,
         label="Random (AUC=0.500)")
plt.title("ROC Curves — All Models (LOOCV, Tuned)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# --- Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(
    best_result['cm'], annot=True, fmt="d",
    xticklabels=["Normal", "ASD"],
    yticklabels=["Normal", "ASD"],
    cmap="Blues"
)
plt.title(f"Confusion Matrix — {best_name} (LOOCV, Tuned)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# --- Feature Ablation Bar Chart ---
plt.figure(figsize=(10, 5))
feat_names = [r['removed_feature']
              for r in feature_ablation_results]
auc_drops = [r['auc_drop']
             for r in feature_ablation_results]
bar_colors = ['#dc3545' if d > 0.05
              else '#ffc107' if d > 0.02
              else '#198754' if d > 0
              else '#6c757d'
              for d in auc_drops]

bars = plt.barh(feat_names, auc_drops, color=bar_colors)
plt.xlabel("AUC Drop When Feature Removed")
plt.ylabel("Removed Feature")
plt.title(f"Feature Ablation Study — {best_name}")
plt.axvline(x=0, color='black', linewidth=0.5)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("feature_ablation.png", dpi=150)
plt.show()

# --- LOOCV Classification Report ---
print(f"\nLOOCV Classification Report ({best_name}):")
print(classification_report(
    best_result['y_true'], best_result['y_pred'],
    target_names=["Normal", "ASD"],
    zero_division=0
))


# ==========================================================
# 13. TRAIN FINAL MODEL ON ALL DATA
# ==========================================================

print("\n" + "=" * 60)
print(f"FINAL MODEL TRAINING — {best_name} (Tuned)")
print("=" * 60)

final_pipeline = clone(best_tuned_pipeline)
final_pipeline.fit(X_selected, y)

clf = final_pipeline.named_steps['clf']
if hasattr(clf, 'coef_'):
    print(f"\nModel coefficients:")
    for fname, coef in zip(selected_features, clf.coef_[0]):
        direction = "↑ ASD" if coef > 0 else "↓ ASD"
        print(f"  {fname}: {coef:.4f} ({direction})")
    print(f"  intercept: {clf.intercept_[0]:.4f}")
elif hasattr(clf, 'feature_importances_'):
    print(f"\nFeature importances:")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"  {selected_features[idx]}: "
              f"{importances[idx]:.4f}")


# ==========================================================
# 14. SAVE ALL ARTIFACTS
# ==========================================================

asd_profile = final_df[
    final_df["group"] == 1
][selected_features].mean()
normal_profile = final_df[
    final_df["group"] == 0
][selected_features].mean()

joblib.dump(asd_profile, "asd_profile.pkl")
joblib.dump(normal_profile, "normal_profile.pkl")
joblib.dump(final_pipeline, "model.pkl")

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print(f"  model.pkl              - {best_name} (tuned)")
print("  selected_features.pkl  - Feature names")
print("  feature_means.pkl      - For imputation")
print("  asd_profile.pkl        - ASD group means")
print("  normal_profile.pkl     - Normal group means")
print("  model_comparison.png   - Model comparison chart")
print("  roc_curve.png          - ROC curves all models")
print("  confusion_matrix.png   - Best model CM")
print("  cv_performance.png     - 5-fold CV plot")
print("  feature_ablation.png   - Feature ablation chart")


# ==========================================================
# 15. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("MODEL ASSESSMENT")
print("=" * 60)
print(f"  Best Algorithm:    {best_name}")
print(f"  Best Params:       {best_result['best_params']}")
print(f"  Features ({len(selected_features)}):    "
      f"{selected_features}")
print(f"  LOOCV AUC:         {best_result['auc']:.3f}")
print(f"  LOOCV Sensitivity: "
      f"{best_result['sensitivity']:.3f}")
print(f"  LOOCV Specificity: "
      f"{best_result['specificity']:.3f}")
print(f"  Training Accuracy: "
      f"{np.mean(train_acc_list):.3f}")
print(f"  Testing Accuracy:  "
      f"{np.mean(val_acc_list):.3f}")

if best_result['auc'] >= 0.80:
    print("\n✅ GOOD: AUC ≥ 0.80")
elif best_result['auc'] >= 0.70:
    print("\n⚠️  ACCEPTABLE: AUC 0.70-0.80")
elif best_result['auc'] >= 0.60:
    print("\n⚠️  WEAK: AUC 0.60-0.70")
else:
    print("\n❌ INSUFFICIENT: AUC < 0.60")