# ============================================================
# CSEMDSME01 - Model Engineering
# Task 3: Machine Learning and the Problem of Imbalanced Learning
# Dataset: KDD Cup 1999
# Author: Ganesh Lakshmana
# ============================================================

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, cross_validate, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    make_scorer
)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
from scipy.stats import randint, uniform
import os

print("All imports successful!")
print(f"Scikit-learn, imbalanced-learn, XGBoost ready.")

# ============================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================

# KDD Cup 1999 column names (official)
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

print("\n--- Loading KDD Cup 1999 dataset ---")
print("Attempting to download from UCI KDD Archive...")

# Try to load - user should place kddcup.data.gz in same folder
# or we download it
import urllib.request
import gzip
import shutil

DATA_URL = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
DATA_FILE = "kddcup.data_10_percent.gz"
CSV_FILE  = "kddcup.data_10_percent"

print(f"Found existing {CSV_FILE}")

df = pd.read_csv(CSV_FILE, header=None, names=col_names)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ============================================================
# SECTION 2: DATA UNDERSTANDING & CLASS DISTRIBUTION
# ============================================================

print("\n--- Label Distribution (original) ---")
label_counts = df['label'].value_counts()
print(label_counts)

# Convert to binary: 'normal.' = 0, anything else = 1 (attack)
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

print("\n--- Binary Label Distribution ---")
binary_counts = df['binary_label'].value_counts()
print(binary_counts)
print(f"\nClass ratio (attack:normal) = {binary_counts[1]}:{binary_counts[0]}")
imbalance_ratio = binary_counts[1] / binary_counts[0]
print(f"Imbalance ratio: {imbalance_ratio:.4f}")

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original label distribution (top 10)
top10 = label_counts.head(10)
axes[0].bar(range(len(top10)), top10.values, color='steelblue')
axes[0].set_xticks(range(len(top10)))
axes[0].set_xticklabels(top10.index, rotation=45, ha='right', fontsize=9)
axes[0].set_title('Fig. 1: Top 10 Original Label Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Attack Type')

# Binary class distribution
colors = ['#2ecc71', '#e74c3c']
axes[1].bar(['Normal (0)', 'Attack (1)'], binary_counts.values, color=colors, edgecolor='black')
axes[1].set_title('Fig. 2: Binary Class Distribution (Imbalanced)', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Class')
for i, v in enumerate(binary_counts.values):
    axes[1].text(i, v + 500, f'{v:,}\n({v/len(df)*100:.1f}%)',
                 ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig1_class_distribution.png")

# ============================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================

print("\n--- Preprocessing ---")

# Encode categorical features
cat_cols = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(['label', 'binary_label'], axis=1)
y = df['binary_label']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Dataset statistics
print("\n--- Basic Statistics ---")
print(f"Total samples:  {len(df):,}")
print(f"Normal samples: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"Attack samples: {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")
print(f"Features:       {X.shape[1]}")
print(f"Missing values: {X.isnull().sum().sum()}")

# ============================================================
# SECTION 4: DEFINE MODELS AND PIPELINES
# ============================================================

print("\n--- Setting up classifiers and pipelines ---")

# Scoring metrics
scoring = {
    'precision': make_scorer(precision_score, zero_division=0),
    'recall':    make_scorer(recall_score,    zero_division=0),
    'f1':        make_scorer(f1_score,        zero_division=0),
    'accuracy':  make_scorer(accuracy_score)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':             xgb.XGBClassifier(n_estimators=100, random_state=42,
                                              use_label_encoder=False,
                                              eval_metric='logloss', n_jobs=-1)
}

# Sampling strategies
samplers = {
    'No Sampling':     None,
    'Over-Sampling':   RandomOverSampler(random_state=42),
    'SMOTE':           SMOTE(random_state=42, n_jobs=-1)
}

# ============================================================
# SECTION 5: CROSS-VALIDATION EXPERIMENTS
# ============================================================

print("\n--- Running 5-Fold Cross-Validation for all combinations ---")
print("This may take a few minutes...\n")

results = []  # store all results

for clf_name, clf in classifiers.items():
    for samp_name, sampler in samplers.items():
        print(f"  [{clf_name}] + [{samp_name}] ...", end=' ', flush=True)

        if sampler is None:
            # Standard sklearn pipeline (no resampling)
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
        else:
            # Imbalanced-learn pipeline (resampling inside CV)
            pipe = ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampler', sampler),
                ('clf', clf)
            ])

        cv_results = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1  # pipelines already use n_jobs inside
        )

        row = {
            'Classifier':  clf_name,
            'Sampling':    samp_name,
            'Precision':   cv_results['test_precision'].mean(),
            'Recall':      cv_results['test_recall'].mean(),
            'F1-Score':    cv_results['test_f1'].mean(),
            'Accuracy':    cv_results['test_accuracy'].mean(),
            'Precision_std': cv_results['test_precision'].std(),
            'Recall_std':    cv_results['test_recall'].std(),
            'F1_std':        cv_results['test_f1'].std(),
        }
        results.append(row)
        print(f"F1={row['F1-Score']:.4f}  Recall={row['Recall']:.4f}")

results_df = pd.DataFrame(results)

# ============================================================
# SECTION 6: RESULTS TABLES
# ============================================================

print("\n\n========================================================")
print("TABLE 1: Full Cross-Validation Results (Mean ± Std)")
print("========================================================")

display_df = results_df.copy()
display_df['Precision']  = display_df.apply(lambda r: f"{r['Precision']:.4f} ± {r['Precision_std']:.4f}", axis=1)
display_df['Recall']     = display_df.apply(lambda r: f"{r['Recall']:.4f} ± {r['Recall_std']:.4f}", axis=1)
display_df['F1-Score']   = display_df.apply(lambda r: f"{r['F1-Score']:.4f} ± {r['F1_std']:.4f}", axis=1)
display_df['Accuracy']   = display_df['Accuracy'].apply(lambda x: f"{x:.4f}")

table1 = display_df[['Classifier', 'Sampling', 'Precision', 'Recall', 'F1-Score', 'Accuracy']]
print(table1.to_string(index=False))

# Per-classifier summary tables
for clf_name in classifiers.keys():
    print(f"\n--- Table: {clf_name} ---")
    sub = table1[table1['Classifier'] == clf_name][['Sampling', 'Precision', 'Recall', 'F1-Score', 'Accuracy']]
    print(sub.to_string(index=False))

# ============================================================
# SECTION 7: VISUALIZATION - METRIC COMPARISON
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['Precision', 'Recall', 'F1-Score']
colors_map = {'No Sampling': '#3498db', 'Over-Sampling': '#e67e22', 'SMOTE': '#2ecc71'}

for ax, metric in zip(axes, metrics):
    x = np.arange(len(classifiers))
    width = 0.25
    for i, (samp_name, color) in enumerate(colors_map.items()):
        vals = [results_df[(results_df['Classifier']==clf) &
                           (results_df['Sampling']==samp_name)][metric].values[0]
                for clf in classifiers.keys()]
        stds = [results_df[(results_df['Classifier']==clf) &
                           (results_df['Sampling']==samp_name)][f'{metric.split("-")[0]}_std' if metric != 'F1-Score' else 'F1_std'].values[0]
                for clf in classifiers.keys()]
        bars = ax.bar(x + i*width, vals, width, label=samp_name,
                      color=color, alpha=0.85, edgecolor='black')
        ax.errorbar(x + i*width, vals, yerr=stds, fmt='none',
                    color='black', capsize=3, linewidth=1)

    ax.set_title(f'Fig. {metrics.index(metric)+3}: {metric} Comparison', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(classifiers.keys()), rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fig2_metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fig2_metric_comparison.png")

# ============================================================
# SECTION 8: HEATMAP - F1 SCORES
# ============================================================

pivot_f1 = results_df.pivot(index='Classifier', columns='Sampling', values='F1-Score')
# reorder columns
pivot_f1 = pivot_f1[['No Sampling', 'Over-Sampling', 'SMOTE']]

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlGnBu',
            linewidths=0.5, ax=ax, vmin=0.8, vmax=1.0,
            annot_kws={"size": 12, "weight": "bold"})
ax.set_title('Fig. 5: F1-Score Heatmap (Classifier × Sampling Strategy)', fontweight='bold', pad=12)
ax.set_xlabel('Sampling Strategy')
ax.set_ylabel('Classifier')
plt.tight_layout()
plt.savefig('fig3_f1_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_f1_heatmap.png")

# ============================================================
# SECTION 9: IDENTIFY BEST MODEL & OPTIMIZE
# ============================================================

# Best model = highest F1 across all combos
best_row = results_df.loc[results_df['F1-Score'].idxmax()]
print(f"\n========================================================")
print(f"BEST MODEL (pre-optimization):")
print(f"  Classifier : {best_row['Classifier']}")
print(f"  Sampling   : {best_row['Sampling']}")
print(f"  F1-Score   : {best_row['F1-Score']:.4f}")
print(f"  Recall     : {best_row['Recall']:.4f}")
print(f"  Precision  : {best_row['Precision']:.4f}")
print(f"========================================================")

# ============================================================
# SECTION 10: HYPERPARAMETER OPTIMIZATION (RandomizedSearchCV)
# ============================================================

print("\n--- Hyperparameter Optimization with RandomizedSearchCV ---")
print("Using 50,000 sample subset for speed (academically justified)...")

# Subsample for optimization only - stratified to preserve class ratio
from sklearn.model_selection import train_test_split
X_sub, _, y_sub, _ = train_test_split(
    X, y, train_size=50000, stratify=y, random_state=42
)
print(f"Subsample: {X_sub.shape[0]:,} samples | "
      f"Normal: {(y_sub==0).sum():,} | Attack: {(y_sub==1).sum():,}")

# Build the pipeline for optimization
opt_pipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE(random_state=42)),   # removed n_jobs to fix FutureWarning
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Focused parameter grid
param_dist = {
    'clf__n_estimators':      [100, 200, 300],
    'clf__max_depth':         [10, 20, 30, None],
    'clf__min_samples_split': randint(2, 15),
    'clf__min_samples_leaf':  randint(1, 8),
    'clf__max_features':      ['sqrt', 'log2'],
    'clf__class_weight':      [None, 'balanced'],
}

cv_small = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    opt_pipe,
    param_distributions=param_dist,
    n_iter=10,            # reduced to 10 - fast but sufficient
    cv=cv_small,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=2,
    return_train_score=True
)

print("Running RandomizedSearchCV (n_iter=10, 5-fold, 50k subsample)...")
print("Estimated time: 2-4 minutes...")
random_search.fit(X_sub, y_sub)

print(f"\nBest Parameters Found:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nBest CV F1-Score (optimized): {random_search.best_score_:.4f}")
# ============================================================
# SECTION 11: OPTIMIZED MODEL - FULL METRICS
# ============================================================

print("\n--- Evaluating Optimized Model with Full Metrics ---")

opt_cv_results = cross_validate(
    random_search.best_estimator_,
    X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=1
)

opt_metrics = {
    'Classifier':  'Random Forest (Optimized)',
    'Sampling':    'SMOTE',
    'Precision':   opt_cv_results['test_precision'].mean(),
    'Recall':      opt_cv_results['test_recall'].mean(),
    'F1-Score':    opt_cv_results['test_f1'].mean(),
    'Accuracy':    opt_cv_results['test_accuracy'].mean(),
    'Precision_std': opt_cv_results['test_precision'].std(),
    'Recall_std':    opt_cv_results['test_recall'].std(),
    'F1_std':        opt_cv_results['test_f1'].std(),
}

print(f"\n========================================================")
print(f"TABLE 2: Optimized Model Results")
print(f"========================================================")
print(f"  Precision : {opt_metrics['Precision']:.4f} ± {opt_metrics['Precision_std']:.4f}")
print(f"  Recall    : {opt_metrics['Recall']:.4f} ± {opt_metrics['Recall_std']:.4f}")
print(f"  F1-Score  : {opt_metrics['F1-Score']:.4f} ± {opt_metrics['F1_std']:.4f}")
print(f"  Accuracy  : {opt_metrics['Accuracy']:.4f}")
print(f"========================================================")

# ============================================================
# SECTION 12: BEFORE vs AFTER OPTIMIZATION COMPARISON
# ============================================================

# Get RF + SMOTE baseline
rf_smote_base = results_df[
    (results_df['Classifier']=='Random Forest') &
    (results_df['Sampling']=='SMOTE')
].iloc[0]

comparison_data = {
    'Configuration':  ['RF + SMOTE (Baseline)', 'RF + SMOTE (Optimized)'],
    'Precision':      [rf_smote_base['Precision'], opt_metrics['Precision']],
    'Recall':         [rf_smote_base['Recall'],    opt_metrics['Recall']],
    'F1-Score':       [rf_smote_base['F1-Score'],  opt_metrics['F1-Score']],
    'Accuracy':       [rf_smote_base['Accuracy'],  opt_metrics['Accuracy']],
}
comparison_df = pd.DataFrame(comparison_data)

print(f"\n========================================================")
print(f"TABLE 3: Before vs After Optimization")
print(f"========================================================")
print(comparison_df.to_string(index=False))

# ============================================================
# SECTION 13: CONFUSION MATRIX (Optimized Model)
# ============================================================

print("\n--- Generating Confusion Matrix for Optimized Model ---")

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(random_search.best_estimator_, X, y, cv=cv, n_jobs=1)
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'],
            ax=ax, linewidths=0.5,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title('Fig. 6: Confusion Matrix – Optimized Random Forest + SMOTE\n(5-Fold CV Predictions)',
             fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('fig4_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_confusion_matrix.png")

print(f"\nClassification Report (Optimized Model):")
print(classification_report(y, y_pred, target_names=['Normal', 'Attack']))

# ============================================================
# SECTION 14: FEATURE IMPORTANCE
# ============================================================

print("\n--- Feature Importance (Optimized Random Forest) ---")

# Fit on full data to get feature importances
best_rf_pipe = random_search.best_estimator_
best_rf_pipe.fit(X, y)
rf_clf = best_rf_pipe.named_steps['clf']
importances = rf_clf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.invert_yaxis()
ax.set_title('Fig. 7: Top 15 Feature Importances – Optimized Random Forest', fontweight='bold')
ax.set_xlabel('Feature Importance (Gini)')
ax.set_ylabel('Feature')
for i, (val, name) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_feature_importance.png")

print("\nTop 10 Features:")
print(feat_imp.head(10).to_string())

# ============================================================
# SECTION 15: RESAMPLING EFFECT ANALYSIS
# ============================================================

print("\n--- Resampling Effect Analysis ---")

# Show sample sizes after resampling
print("\nSample sizes after resampling (approximate):")
X_arr = X.values
y_arr = y.values

ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_arr, y_arr)
print(f"  Original      : {len(y_arr):,} samples → Normal: {(y_arr==0).sum():,} | Attack: {(y_arr==1).sum():,}")
print(f"  Over-Sampling : {len(y_ros):,} samples → Normal: {(y_ros==0).sum():,} | Attack: {(y_ros==1).sum():,}")

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_arr, y_arr)
print(f"  SMOTE         : {len(y_smote):,} samples → Normal: {(y_smote==0).sum():,} | Attack: {(y_smote==1).sum():,}")

# Visualization: sample count comparison
fig, ax = plt.subplots(figsize=(10, 5))
strategies = ['Original', 'Over-Sampling', 'SMOTE']
normal_counts = [(y_arr==0).sum(), (y_ros==0).sum(), (y_smote==0).sum()]
attack_counts = [(y_arr==1).sum(), (y_ros==1).sum(), (y_smote==1).sum()]
x = np.arange(len(strategies))
w = 0.35
ax.bar(x - w/2, normal_counts, w, label='Normal (0)', color='#2ecc71', edgecolor='black')
ax.bar(x + w/2, attack_counts, w, label='Attack (1)', color='#e74c3c', edgecolor='black')
ax.set_title('Fig. 8: Sample Count per Class After Resampling', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylabel('Sample Count')
ax.legend()
for i in range(len(strategies)):
    ax.text(i - w/2, normal_counts[i] + 500, f'{normal_counts[i]:,}', ha='center', fontsize=8)
    ax.text(i + w/2, attack_counts[i] + 500, f'{attack_counts[i]:,}', ha='center', fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig6_resampling_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig6_resampling_effect.png")

# ============================================================
# SECTION 16: COMPREHENSIVE SUMMARY FOR REPORT
# ============================================================

print("\n\n" + "="*65)
print("COMPLETE RESULTS SUMMARY (COPY FOR REPORT)")
print("="*65)

print("\n--- DATASET STATISTICS ---")
print(f"Total Samples        : {len(df):,}")
print(f"Normal Samples       : {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"Attack Samples       : {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")
print(f"Features             : {X.shape[1]}")
print(f"Imbalance Ratio      : {imbalance_ratio:.2f}:1 (attack:normal)")

print("\n--- TABLE 1: CROSS-VALIDATION RESULTS (5-FOLD) ---")
print(results_df[['Classifier','Sampling','Precision','Recall','F1-Score','Accuracy']].to_string(index=False, float_format='%.4f'))

print("\n--- TABLE 2: OPTIMIZED MODEL RESULTS ---")
print(f"Model: Random Forest + SMOTE (Randomized Search, 20 iterations, 5-fold CV)")
print(f"Precision : {opt_metrics['Precision']:.4f} ± {opt_metrics['Precision_std']:.4f}")
print(f"Recall    : {opt_metrics['Recall']:.4f} ± {opt_metrics['Recall_std']:.4f}")
print(f"F1-Score  : {opt_metrics['F1-Score']:.4f} ± {opt_metrics['F1_std']:.4f}")
print(f"Accuracy  : {opt_metrics['Accuracy']:.4f}")

print("\n--- TABLE 3: BEFORE vs AFTER OPTIMIZATION ---")
print(comparison_df.to_string(index=False, float_format='%.4f'))

print("\n--- BEST HYPERPARAMETERS ---")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")

print("\n--- TOP 10 FEATURES ---")
print(feat_imp.head(10).to_string())

print("\n--- RESAMPLING EFFECT ON CLASS BALANCE ---")
print(f"Original      → Normal: {(y_arr==0).sum():,} | Attack: {(y_arr==1).sum():,}")
print(f"Over-Sampling → Normal: {(y_ros==0).sum():,} | Attack: {(y_ros==1).sum():,}")
print(f"SMOTE         → Normal: {(y_smote==0).sum():,} | Attack: {(y_smote==1).sum():,}")

print("\n--- CONFUSION MATRIX (Optimized Model, 5-Fold CV) ---")
print("          Predicted Normal  Predicted Attack")
print(f"True Normal     {cm[0][0]:>8,}       {cm[0][1]:>10,}")
print(f"True Attack     {cm[1][0]:>8,}       {cm[1][1]:>10,}")

print("\n" + "="*65)
print("ALL FIGURES SAVED:")
print("  fig1_class_distribution.png")
print("  fig2_metric_comparison.png")
print("  fig3_f1_heatmap.png")
print("  fig4_confusion_matrix.png")
print("  fig5_feature_importance.png")
print("  fig6_resampling_effect.png")
print("="*65)
print("\n✅ PROJECT COMPLETE - Ready to write the report!")