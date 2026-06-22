import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, GridSearchCV,
    permutation_test_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_curve, auc,
    classification_report, roc_auc_score,
    balanced_accuracy_score, f1_score
)
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from model_utils import extract_features_from_image

# ==========================================================
# CONFIGURATION
# ==========================================================
FEATURES_CSV = "extracted_features.csv"
TRAINING_FOLDER = "training_images"
ASD_FOLDER = "training_images_ASD"
GRAPHS_DIR = "model_graphs"
RANDOM_STATE = 42
MIN_FEATURES = 3
MAX_FEATURES = 5
MAX_OVERFIT_GAP = 0.15
N_PERMUTATIONS = 100

os.makedirs(GRAPHS_DIR, exist_ok=True)


# 1. LOAD DATA

def load_data():
    if os.path.exists(FEATURES_CSV):
        print(f"Loading cached features from {FEATURES_CSV}...\n")
        df = pd.read_csv(FEATURES_CSV)
        print(f"Loaded {len(df)} samples from cache.")
        return df

    all_data = []
    print("Reading NORMAL images...\n")
    for file in os.listdir(TRAINING_FOLDER):
        path = os.path.join(TRAINING_FOLDER, file)
        try:
            print("Processing:", file)
            image = Image.open(path)
            features = extract_features_from_image(image)
            features["file"] = file
            features["group"] = 0
            all_data.append(features)
        except Exception as e:
            print("Error:", file, e)

    print("\nReading ASD images...\n")
    for file in os.listdir(ASD_FOLDER):
        path = os.path.join(ASD_FOLDER, file)
        try:
            print("Processing ASD:", file)
            image = Image.open(path)
            features = extract_features_from_image(image)
            features["file"] = file
            features["group"] = 1
            all_data.append(features)
        except Exception as e:
            print("Error:", file, e)

    df = pd.DataFrame(all_data)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"\n✅ Features saved to {FEATURES_CSV}")
    return df


# 2. MODEL DEFINITIONS 

def get_model_configs():
    configs = {}

    # 1. Logistic Regression — simple, interpretable baseline
    configs['Logistic Regression'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=10000,
                class_weight='balanced'
            ))
        ]),
        'param_grid': {
            'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

    # 2. SVM (RBF) — strong nonlinear classifier
    configs['SVM (RBF)'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', class_weight='balanced',
                probability=True, random_state=RANDOM_STATE
            ))
        ]),
        'param_grid': {
            'clf__C': [0.1, 1.0, 10.0, 100.0],
            'clf__gamma': ['scale', 0.01, 0.1, 1.0]
        }
    }

    # 3. Random Forest — robust ensemble
    configs['Random Forest'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=500, class_weight='balanced',
                random_state=RANDOM_STATE
            ))
        ]),
        'param_grid': {
            'clf__max_depth': [2, 3, 5],
            'clf__min_samples_leaf': [3, 5, 10],
            'clf__min_samples_split': [2, 5]
        }
    }

    # 4. Gradient Boosting — strong for small datasets
    configs['Gradient Boosting'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        'param_grid': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [1, 2, 3],
            'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'clf__min_samples_leaf': [3, 5, 10],
            'clf__subsample': [0.7, 0.8, 1.0]
        }
    }

    # 5. KNN — OVERFITTING FIX
    configs['KNN'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
        ]),
        'param_grid': {
            'clf__n_neighbors': [5, 7, 9, 11, 13],
            'clf__weights': ['uniform'],
            'clf__metric': ['euclidean', 'manhattan']
        }
    }

    return configs


# 3. FEATURE SELECTION

def run_feature_selection(X, y):
    n_features = X.shape[1]
    total_combos = sum(
        comb(n_features, k)
        for k in range(MIN_FEATURES, MAX_FEATURES + 1)
    )

    print("\n" + "=" * 60)
    print(f"  FEATURE SELECTION")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )

    search_clf = LogisticRegression(
        solver='lbfgs', max_iter=10000,
        class_weight='balanced', C=0.1
    )

    efs = EFS(
        search_clf,
        min_features=MIN_FEATURES,
        max_features=MAX_FEATURES,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True,
                           random_state=RANDOM_STATE),
        print_progress=True,
        n_jobs=-1
    )
    efs = efs.fit(X_scaled, y)

    selected = list(efs.best_feature_names_)
    print(f"\nBest AUC: {efs.best_score_:.3f}")
    print(f"Selected features: {selected}")

    # Show top 10
    results_df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
    results_df['n_features'] = results_df['feature_names'].apply(len)
    results_df.sort_values('avg_score', ascending=False, inplace=True)

    print("\n  Top 3 combinations by feature set size:")
    for n in [3, 4, 5]:
        subset = results_df[results_df['n_features'] == n].head(3)
        print(f"\n  {n}-feature combinations:")
        for i, (_, row) in enumerate(subset.iterrows()):
            print(f"    {i+1}. {row['feature_names']} -> "
                  f"AUC={row['avg_score']:.3f} ± {row['std_dev']:.3f}")

    return selected


# 4. HYPERPARAMETER TUNING

def tune_models(model_configs, X_selected, y):
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING")
    print("=" * 60)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE)
    tuned = {}

    for name, config in model_configs.items():
        print(f"\n--- {name} ---")
        grid = GridSearchCV(
            config['pipeline'], config['param_grid'],
            scoring='roc_auc', cv=inner_cv,
            n_jobs=-1, refit=True
        )
        grid.fit(X_selected, y)
        tuned[name] = {
            'pipeline': grid.best_estimator_,
            'best_params': grid.best_params_,
            'best_cv_auc': grid.best_score_
        }
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV AUC: {grid.best_score_:.3f}")

    return tuned


# 5. LOOCV EVALUATION

def evaluate_loocv(tuned_models, X_selected, y):
    print("\n" + "=" * 60)
    print("  LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 60)

    loo = LeaveOneOut()
    results = {}

    for name, tuned in tuned_models.items():
        print(f"\n--- {name} ---")
        y_true, y_pred, y_prob = [], [], []

        for train_idx, test_idx in loo.split(X_selected):
            X_tr = X_selected.iloc[train_idx]
            X_te = X_selected.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]

            pipe = clone(tuned['pipeline'])
            pipe.fit(X_tr, y_tr)

            y_true.append(y_te.values[0])
            y_pred.append(pipe.predict(X_te)[0])
            y_prob.append(pipe.predict_proba(X_te)[0, 1])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'cm': cm,
            'best_params': tuned['best_params'],
        }

        r = results[name]
        print(f"  Acc={r['accuracy']:.3f}  BalAcc={r['balanced_accuracy']:.3f}  "
              f"AUC={r['auc']:.3f}  F1={r['f1']:.3f}  "
              f"Sens={r['sensitivity']:.3f}  Spec={r['specificity']:.3f}")

    return results


# 6. OVERFITTING GUARD

def check_overfitting(tuned_models, X_selected, y, all_results):
    print("\n" + "=" * 60)
    print("  OVERFITTING GUARD")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=RANDOM_STATE)
    overfit_report = []

    print(f"\n  {'Model':<22} {'AUC':>6} {'Train':>7} {'Test':>6} "
          f"{'Gap':>6} {'Status':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*10}")

    for name, tuned in tuned_models.items():
        train_accs, test_accs = [], []

        for train_idx, val_idx in cv.split(X_selected, y):
            X_tr = X_selected.iloc[train_idx]
            X_val = X_selected.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = clone(tuned['pipeline'])
            model.fit(X_tr, y_tr)
            train_accs.append(accuracy_score(y_tr, model.predict(X_tr)))
            test_accs.append(accuracy_score(y_val, model.predict(X_val)))

        gap = np.mean(train_accs) - np.mean(test_accs)
        is_overfit = gap > MAX_OVERFIT_GAP
        status = "OVERFIT" if is_overfit else "OK"

        print(f"  {name:<22} {all_results[name]['auc']:>6.3f} "
              f"{np.mean(train_accs):>7.3f} {np.mean(test_accs):>6.3f} "
              f"{gap:>6.3f} {status:>10}")

        overfit_report.append({
            'name': name,
            'auc': all_results[name]['auc'],
            'train_acc': np.mean(train_accs),
            'test_acc': np.mean(test_accs),
            'gap': gap,
            'is_overfit': is_overfit,
        })

    # Auto-select best non-overfitting model
    sorted_by_auc = sorted(overfit_report,
                           key=lambda x: x['auc'], reverse=True)

    best = None
    for entry in sorted_by_auc:
        if not entry['is_overfit']:
            best = entry
            break

    if best is None:
        # All overfit — pick highest AUC anyway
        best = sorted_by_auc[0]
        print(f"\n  All models show overfitting.")
        print(f"  Selecting highest AUC: {best['name']}")
    else:
        if best['name'] != sorted_by_auc[0]['name']:
            print(f"\n  Highest AUC model ({sorted_by_auc[0]['name']}) "
                  f"overfits (gap={sorted_by_auc[0]['gap']:.3f})")
            print(f"  Auto-selected: {best['name']} "
                  f"(AUC={best['auc']:.3f}, gap={best['gap']:.3f})")
        else:
            print(f"\n  Best model ({best['name']}) has acceptable "
                  f"overfit gap ({best['gap']:.3f})")

    return best['name'], overfit_report


# 7. PERMUTATION TEST

def run_permutation_test(pipeline, X_selected, y, model_name):
    print("\n" + "=" * 60)
    print("  PERMUTATION TEST")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=RANDOM_STATE)

    score, perm_scores, p_value = permutation_test_score(
        clone(pipeline), X_selected, y,
        scoring='roc_auc', cv=cv,
        n_permutations=N_PERMUTATIONS,
        n_jobs=-1, random_state=RANDOM_STATE
    )

    print(f"\n  True CV AUC:            {score:.3f}")
    print(f"  Permutation mean AUC:   {perm_scores.mean():.3f} "
          f"± {perm_scores.std():.3f}")
    print(f"  p-value:                {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n  SIGNIFICANT (p={p_value:.4f} < 0.05)")
        print(f"     The model found REAL signal in the data.")
    else:
        print(f"\n  NOT SIGNIFICANT (p={p_value:.4f} ≥ 0.05)")
        print(f"     Results may be due to chance.")

    return score, perm_scores, p_value


# 8. ABLATION STUDY

def run_ablation(pipeline, selected_features, X, y, base_auc):
    print("\n" + "=" * 60)
    print("  FEATURE ABLATION")
    print("=" * 60)

    loo = LeaveOneOut()
    results = []

    for feat in selected_features:
        ablated = [f for f in selected_features if f != feat]
        X_abl = X[ablated]
        y_true, y_prob = [], []

        for train_idx, test_idx in loo.split(X_abl):
            pipe = clone(pipeline)
            pipe.fit(X_abl.iloc[train_idx], y.iloc[train_idx])
            y_true.append(y.iloc[test_idx].values[0])
            y_prob.append(pipe.predict_proba(X_abl.iloc[test_idx])[0, 1])

        abl_auc = roc_auc_score(y_true, y_prob)
        drop = base_auc - abl_auc
        importance = ("CRITICAL" if drop > 0.05
                      else "Important" if drop > 0.02
                      else "Minor" if drop > 0
                      else "Redundant")

        results.append({'feature': feat, 'auc': abl_auc, 'drop': drop})
        print(f"  Remove '{feat}': AUC={abl_auc:.3f} "
              f"(Δ={drop:+.3f}) [{importance}]")

    return sorted(results, key=lambda x: x['drop'], reverse=True)



# 9. VISUALIZATIONS

def generate_plots(all_results, best_name, train_accs, val_accs,
                   val_aucs, ablation_results, perm_scores, true_score,
                   p_value, overfit_report):

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]['auc'], reverse=True)

    # --- Plot 1: Model Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['accuracy', 'balanced_accuracy', 'auc', 'f1',
               'sensitivity', 'specificity']
    labels = ['Accuracy', 'Bal.Acc', 'AUC', 'F1', 'Sens', 'Spec']
    x = np.arange(len(metrics))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

    for i, (name, res) in enumerate(sorted_results):
        vals = [res[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i])

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison (LOOCV)')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/01_model_comparison.png", dpi=150)
    plt.close()

    # --- Plot 2: ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_roc = plt.cm.Set1(np.linspace(0, 1, len(sorted_results)))
    for i, (name, res) in enumerate(sorted_results):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title("ROC Curves (LOOCV)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/02_roc_curves.png", dpi=150)
    plt.close()

    # --- Plot 3: Confusion Matrix (best model) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(all_results[best_name]['cm'], annot=True, fmt="d",
                xticklabels=["Normal", "ASD"],
                yticklabels=["Normal", "ASD"],
                cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/03_confusion_matrix.png", dpi=150)
    plt.close()

    # --- Plot 4: 5-Fold CV ---
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = range(1, 6)
    ax.plot(folds, train_accs, 'o-', color='blue', lw=2,
            label=f"Train ({np.mean(train_accs):.3f})")
    ax.plot(folds, val_accs, 's-', color='red', lw=2,
            label=f"Test ({np.mean(val_accs):.3f})")
    ax.axhline(np.mean(train_accs), color='blue', ls='--', alpha=0.4)
    ax.axhline(np.mean(val_accs), color='red', ls='--', alpha=0.4)
    ax.set_title(f"5-Fold CV — {best_name}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/04_cv_performance.png", dpi=150)
    plt.close()

    # --- Plot 5: Permutation Test ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_scores, bins=20, color='lightcoral',
            edgecolor='black', alpha=0.7, label='Null distribution')
    ax.axvline(true_score, color='blue', lw=3,
               label=f'True score = {true_score:.3f}')
    ax.axvline(perm_scores.mean(), color='red', lw=2, ls='--',
               label=f'Null mean = {perm_scores.mean():.3f}')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Permutation Test (p={p_value:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/05_permutation_test.png", dpi=150)
    plt.close()

    # --- Plot 6: Overfitting Comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r['name'] for r in overfit_report]
    gaps = [r['gap'] for r in overfit_report]
    colors_of = ['red' if r['is_overfit'] else 'green'
                 for r in overfit_report]
    ax.barh(names, gaps, color=colors_of, alpha=0.7,
            edgecolor='black', lw=0.5)
    ax.axvline(MAX_OVERFIT_GAP, color='red', ls='--', lw=2,
               label=f'Threshold ({MAX_OVERFIT_GAP})')
    ax.set_xlabel('Overfit Gap (Train - Test Accuracy)')
    ax.set_title('Overfitting Analysis\nGreen=OK, Red=Overfitting')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/06_overfit_analysis.png", dpi=150)
    plt.close()

    # --- Plot 7: Feature Ablation ---
    fig, ax = plt.subplots(figsize=(8, 4))
    feat_names = [r['feature'] for r in ablation_results]
    drops = [r['drop'] for r in ablation_results]
    bar_colors = ['#dc3545' if d > 0.05
                  else '#ffc107' if d > 0.02
                  else '#198754' if d > 0
                  else '#6c757d' for d in drops]
    ax.barh(feat_names, drops, color=bar_colors, edgecolor='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('AUC Drop')
    ax.set_title(f'Feature Ablation — {best_name}')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/07_feature_ablation.png", dpi=150)
    plt.close()


# MAIN PIPELINE

def main():
    # Load data
    final_df = load_data()
    print("\nDataset distribution:")
    print(final_df["group"].value_counts())

    X = final_df.drop(columns=["file", "group"], errors="ignore")
    y = final_df["group"].astype(int)

    feature_means = X.mean()
    joblib.dump(feature_means, "feature_means.pkl")
    X = X.fillna(feature_means)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\nSamples: {len(y)} ({n_pos} ASD, {n_neg} Normal)")
    print(f"Features: {X.shape[1]}")

    # Feature selection
    selected_features = run_feature_selection(X, y)
    joblib.dump(selected_features, "selected_features.pkl")
    X_selected = X[selected_features]

    # Tune models
    model_configs = get_model_configs()
    tuned_models = tune_models(model_configs, X_selected, y)

    # LOOCV evaluation
    all_results = evaluate_loocv(tuned_models, X_selected, y)

    # Model comparison table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1]['auc'], reverse=True)
    print(f"\n  {'Rank':<5}{'Model':<22}{'Acc':>6}{'BalAcc':>7}"
          f"{'AUC':>6}{'F1':>5}{'Sens':>6}{'Spec':>6}")
    print(f"  {'-'*5}{'-'*22}{'-'*6}{'-'*7}{'-'*6}{'-'*5}{'-'*6}{'-'*6}")
    for i, (name, res) in enumerate(sorted_results):
        print(f"  {i+1:<5}{name:<22}{res['accuracy']:>6.3f}"
              f"{res['balanced_accuracy']:>7.3f}{res['auc']:>6.3f}"
              f"{res['f1']:>5.3f}{res['sensitivity']:>6.3f}"
              f"{res['specificity']:>6.3f}")

    # Overfitting guard — auto-select best generalizable model
    best_name, overfit_report = check_overfitting(
        tuned_models, X_selected, y, all_results
    )
    best_result = all_results[best_name]
    best_pipeline = tuned_models[best_name]['pipeline']

    # Permutation test
    true_score, perm_scores, p_value = run_permutation_test(
        best_pipeline, X_selected, y, best_name
    )

    # Ablation study
    ablation_results = run_ablation(
        best_pipeline, selected_features, X, y, best_result['auc']
    )

    # 5-Fold CV
    print("\n" + "=" * 60)
    print(f"  5-FOLD CV — {best_name}")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_accs, val_accs, val_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y), 1):
        model = clone(best_pipeline)
        model.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
        train_accs.append(accuracy_score(
            y.iloc[train_idx], model.predict(X_selected.iloc[train_idx])))
        val_accs.append(accuracy_score(
            y.iloc[val_idx], model.predict(X_selected.iloc[val_idx])))
        val_aucs.append(roc_auc_score(
            y.iloc[val_idx], model.predict_proba(X_selected.iloc[val_idx])[:, 1]))
        print(f"  Fold {fold}: Train={train_accs[-1]:.3f} "
              f"Test={val_accs[-1]:.3f} AUC={val_aucs[-1]:.3f}")

    print(f"\n  Train: {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f}")
    print(f"  Test:  {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")
    print(f"  AUC:   {np.mean(val_aucs):.3f} ± {np.std(val_aucs):.3f}")

    # Visualizations
    generate_plots(all_results, best_name, train_accs, val_accs,
                   val_aucs, ablation_results, perm_scores,
                   true_score, p_value, overfit_report)

    # Train final model and save
    print("\n" + "=" * 60)
    print("  SAVING FINAL MODEL")
    print("=" * 60)
    final_model = clone(best_pipeline)
    final_model.fit(X_selected, y)

    asd_profile = final_df[final_df["group"] == 1][selected_features].mean()
    normal_profile = final_df[final_df["group"] == 0][selected_features].mean()

    joblib.dump(final_model, "model.pkl")
    joblib.dump(asd_profile, "asd_profile.pkl")
    joblib.dump(normal_profile, "normal_profile.pkl")

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Dataset: {len(y)} samples ({n_pos} ASD, {n_neg} Normal)")
    print(f"  Features: {selected_features}")
    print(f"  Best Model: {best_name}")
    print(f"  LOOCV AUC: {best_result['auc']:.3f}")
    print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.3f}")
    print(f"  Sensitivity: {best_result['sensitivity']:.3f}")
    print(f"  Specificity: {best_result['specificity']:.3f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  5-Fold Test Acc: {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")

    if best_result['auc'] >= 0.80:
        print("\n  GOOD: AUC ≥ 0.80")
    elif best_result['auc'] >= 0.70:
        print("\n  ACCEPTABLE: AUC 0.70-0.80")
    else:
        print("\n  WEAK: AUC < 0.70")

    print(f"\n  Files saved: model.pkl, selected_features.pkl,")
    print(f"  feature_means.pkl, asd_profile.pkl, normal_profile.pkl")
    print(f"  Graphs: {GRAPHS_DIR}/")

    # Classification report
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(
        best_result['y_true'], best_result['y_pred'],
        target_names=["Normal", "ASD"], zero_division=0
    ))


if __name__ == "__main__":
    main()