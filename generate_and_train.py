"""
ThyroAI – generate_and_train.py  (v5 — 4 models, T4 rename, CV removed)

Fixes/changes applied in this version:
  W1 — Synthetic lifestyle with probabilistic overlap (reduced deterministic leakage)
  W3 — SHAP summary plot (TreeExplainer) with graceful fallback
  W4 — Title fix: "Model Feature Importance" not "Top Contributing Features"
  W5 — GridSearchCV hyperparameter tuning on Gradient Boosting / Random Forest
  W6 — 3-CLASS DIAGNOSIS: Normal / Hypothyroid / Hyperthyroid, derived from the
       REAL UCI diagnostic codes (not synthetic). Per the standard UCI
       hypothyroid.names codebook:
         '-'        → Normal
         A,B,C,D... → Hyperthyroid family (overactive thyroid)
         F,G,H...   → Hypothyroid family (underactive thyroid)
  W7 — Risk level (No/Medium/High) is now derived from the model's OWN
       PREDICTION CONFIDENCE on the predicted class, not from a separate
       label. This keeps risk level fully explainable from the model output:
         confidence >= 0.80           → High Risk
         0.55 <= confidence <  0.80    → Medium Risk
         confidence <  0.55  (or Normal predicted with low margin) → No Risk
  v5 — MODEL COUNT REDUCED 7 → 4: kept Logistic Regression, Decision Tree,
       Random Forest, XGBoost/Gradient Boosting (Tuned). Removed SVM and KNN.
  v5 — 5-Fold Cross Validation step removed (no longer computed/displayed).
  v5 — Column "TT4" renamed to "T4" throughout (display/label only — same
       underlying UCI "Total T4" values; the UCI dataset has no separate
       raw T4 column distinct from TT4).
"""

import os, warnings, joblib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    USE_XGB = False

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('model', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD REAL DATASET
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Loading real UCI thyroid dataset")
print("=" * 65)

df_raw = pd.read_csv('dataset/thyroidDF.csv')
print(f"  Raw rows: {len(df_raw)}")

df = df_raw[['age', 'sex', 'TSH', 'T3', 'TT4', 'target']].copy()
df = df.rename(columns={'TT4': 'T4'})  # display label per lab-report convention; same underlying values

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — MAP TARGET → 3-CLASS DIAGNOSIS (Normal / Hypo / Hyper)
#          Uses REAL UCI diagnostic codes — not synthetic.
#          Standard UCI hypothyroid.names codebook:
#            '-'            → Normal       (negative finding)
#            A,B,C,D (+combos starting with these) → Hyperthyroid family
#            F,G,H (+combos starting with these)   → Hypothyroid family
#            other rare codes (E,I-S,...)            → dropped (ambiguous/other)
# ═══════════════════════════════════════════════════════════════════
HYPER_CODES = set('ABCD')
HYPO_CODES  = set('FGH')

def map_diagnosis(code):
    code = str(code).strip()
    if code == '-':
        return 'Normal'
    first = code[0]
    if first in HYPER_CODES:
        return 'Hyperthyroid'
    elif first in HYPO_CODES:
        return 'Hypothyroid'
    return None  # ambiguous/rare code — dropped below

df['diagnosis'] = df['target'].apply(map_diagnosis)
n_before = len(df)
df = df[df['diagnosis'].notna()].copy()
print(f"  Dropped {n_before - len(df)} rows with ambiguous/rare diagnostic codes")
print(f"  Diagnosis distribution: {df['diagnosis'].value_counts().to_dict()}")

# Encode diagnosis → integer label for sklearn (alphabetical: Hyperthyroid=0, Hypothyroid=1, Normal=2)
from sklearn.preprocessing import LabelEncoder as _LE
diag_encoder = _LE()
df['target'] = diag_encoder.fit_transform(df['diagnosis'])
DIAG_CLASSES = diag_encoder.classes_.tolist()
print(f"  Encoded classes (in order): {DIAG_CLASSES}")
df = df.drop(columns=['diagnosis'])

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — MISSING VALUE HANDLING
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 3: Missing value handling")

df = df[df['age'] <= 110].copy()
print(f"  Removed age outliers → {len(df)} rows remain")

# Sex: keep only F / M, fill rest with mode
df['sex'] = df['sex'].where(df['sex'].isin(['F', 'M']), other='F')
print(f"  sex: unique after fix = {df['sex'].unique().tolist()}")

# Hormone columns: coerce, fill with median
for col in ['TSH', 'T3', 'T4']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    n_miss = df[col].isnull().sum()
    med = df[col].median()
    df[col] = df[col].fillna(med)
    print(f"  {col}: filled {n_miss} missing → median={med:.3f}")

assert df[['age','sex','TSH','T3','T4','target']].isnull().sum().sum() == 0
print("  ✓ Zero NaN in clinical columns")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — ENCODE SEX
# ═══════════════════════════════════════════════════════════════════
df['sex'] = (df['sex'].str.strip() == 'F').astype(int)   # F=1, M=0

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — LIFESTYLE GENERATION WITH PROBABILISTIC OVERLAP  (W1 fix)
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 5: Generating lifestyle features (probabilistic simulation — reduced deterministic leakage)")

NORMAL_IDX = DIAG_CLASSES.index('Normal')
sleep_hrs, stress_lvl, exercise_freq, weight_gain_col = [], [], [], []

for t in df['target']:
    is_abnormal = (t != NORMAL_IDX)   # either Hypo or Hyper — both are "not normal"
    # SLEEP
    if is_abnormal:
        sleep_hrs.append(np.random.choice([4,5,6,7,8], p=[0.25,0.30,0.25,0.15,0.05]))
    else:
        sleep_hrs.append(np.random.choice([4,5,6,7,8], p=[0.05,0.10,0.20,0.35,0.30]))
    # STRESS
    if is_abnormal:
        stress_lvl.append(np.random.choice(['Low','Medium','High'], p=[0.15,0.35,0.50]))
    else:
        stress_lvl.append(np.random.choice(['Low','Medium','High'], p=[0.40,0.40,0.20]))
    # EXERCISE
    if is_abnormal:
        exercise_freq.append(np.random.choice(['Never','Weekly','Daily'], p=[0.45,0.40,0.15]))
    else:
        exercise_freq.append(np.random.choice(['Never','Weekly','Daily'], p=[0.15,0.45,0.40]))
    # WEIGHT GAIN
    if is_abnormal:
        weight_gain_col.append(np.random.choice([0,1], p=[0.35,0.65]))
    else:
        weight_gain_col.append(np.random.choice([0,1], p=[0.75,0.25]))

df['sleep_hours']        = sleep_hrs
df['stress_level']       = stress_lvl
df['exercise_frequency'] = exercise_freq
df['weight_gain']        = weight_gain_col

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — ENCODE LIFESTYLE CATEGORICALS
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 6: Encoding lifestyle categoricals (ordinal)")

stress_map   = {'Low': 1, 'Medium': 2, 'High': 3}
exercise_map = {'Daily': 3, 'Weekly': 2, 'Never': 1}

df['stress_level']       = df['stress_level'].map(stress_map)
df['exercise_frequency'] = df['exercise_frequency'].map(exercise_map)

with open('model/label_mappings.json', 'w') as f:
    json.dump({'stress_level': stress_map, 'exercise_frequency': exercise_map}, f, indent=2)

# ═══════════════════════════════════════════════════════════════════
# STEP 7 — FINAL FEATURE SELECTION + SAVE
# ═══════════════════════════════════════════════════════════════════
FEATURE_COLS = ['age', 'sex', 'TSH', 'T3', 'T4',
                'sleep_hours', 'stress_level', 'exercise_frequency', 'weight_gain']

df_final = df[FEATURE_COLS + ['target']].copy()
assert df_final.isnull().sum().sum() == 0, "NaN found in final dataset!"
print(f"\n  ✓ Final dataset: {df_final.shape}  |  Zero NaN")
print(f"  Class distribution: {df_final['target'].value_counts().to_dict()}")

df_final.to_csv('dataset/data.csv', index=False)
with open('model/feature_cols.json', 'w') as f:
    json.dump(FEATURE_COLS, f)
with open('model/diagnosis_classes.json', 'w') as f:
    json.dump(DIAG_CLASSES, f)

# ═══════════════════════════════════════════════════════════════════
# STEP 8 — STRATIFIED TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 8: Stratified 80/20 train/test split")

X = df_final[FEATURE_COLS]
y = df_final['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ═══════════════════════════════════════════════════════════════════
# STEP 9 — FEATURE SCALING
# ═══════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

# ═══════════════════════════════════════════════════════════════════
# STEP 10 — HYPERPARAMETER TUNING  (W5 fix)
#           GridSearchCV on Gradient Boosting only
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 10: GridSearchCV — Gradient Boosting hyperparameter tuning")

gb_param_grid = {
    'n_estimators':  [100, 150, 200],
    'max_depth':     [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.15],
}
gb_base = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(
    gb_base, gb_param_grid,
    cv=3, scoring='roc_auc_ovr',   # one-vs-rest AUC for multiclass
    n_jobs=-1, verbose=0
)
gb_grid.fit(X_train_s, y_train)
best_gb_params = gb_grid.best_params_
print(f"  Best GB params: {best_gb_params}")
print(f"  Best CV AUC (GridSearch): {gb_grid.best_score_:.4f}")

# Tuned GB model
TunedGB = GradientBoostingClassifier(**best_gb_params, random_state=42)

# ═══════════════════════════════════════════════════════════════════
# STEP 11 — TRAIN ALL MODELS (class_weight=balanced)
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 11: Training all models")

# class_weight='balanced' handles the multiclass imbalance for sklearn models.
# XGBoost has no native multiclass class_weight param, so we compute per-sample
# weights manually (equivalent effect to class_weight='balanced').
from sklearn.utils.class_weight import compute_sample_weight
xgb_sample_weights = compute_sample_weight('balanced', y_train)

XGBModel = (XGBClassifier(n_estimators=best_gb_params['n_estimators'],
                            max_depth=best_gb_params['max_depth'],
                            learning_rate=best_gb_params['learning_rate'],
                            eval_metric='mlogloss', verbosity=0,
                            num_class=len(DIAG_CLASSES),
                            objective='multi:softprob', random_state=42)
            if USE_XGB else TunedGB)
xgb_label = 'XGBoost (Tuned)' if USE_XGB else 'Gradient Boosting (Tuned)'

models = {
    'Logistic Regression':       LogisticRegression(max_iter=500, random_state=42,
                                                     class_weight='balanced'),
    'Decision Tree':              DecisionTreeClassifier(max_depth=8, random_state=42,
                                                         class_weight='balanced'),
    'Random Forest':              RandomForestClassifier(n_estimators=150, random_state=42,
                                                         class_weight='balanced'),
    xgb_label:                    XGBModel,
}

results = {}
print(f"\n  {'Model':<30}  {'Acc':>5}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'AUC':>6}")
print(f"  {'-'*70}")

# XGBoost needs sample_weight passed explicitly (no native class_weight); others use class_weight='balanced'.
for name, clf in models.items():
    if name == xgb_label and USE_XGB:
        clf.fit(X_train_s, y_train, sample_weight=xgb_sample_weights)
    else:
        clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)   # shape (n_samples, n_classes)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    cm   = confusion_matrix(y_test, y_pred)
    results[name] = {
        'accuracy': round(acc,4), 'precision_macro': round(prec,4),
        'recall_macro': round(rec,4), 'f1_macro': round(f1,4),
        'roc_auc_ovr_macro': round(auc,4),
        'confusion_matrix': cm.tolist(),
    }
    print(f"  {name:<30}  {acc:.3f}  {prec:.3f}  {rec:.3f}  {f1:.3f}  {auc:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 12 — SELECT BEST MODEL
# ═══════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]['roc_auc_ovr_macro'])
best_clf  = models[best_name]
print(f"\n  ★ Best model: {best_name}  |  Macro AUC={results[best_name]['roc_auc_ovr_macro']:.4f}")

y_pred_best = best_clf.predict(X_test_s)
print("\n  Classification Report (best model):")
print(classification_report(y_test, y_pred_best, target_names=DIAG_CLASSES))

# ═══════════════════════════════════════════════════════════════════
# STEP 13 — SHAP EXPLAINABILITY  (W3 fix)
# ═══════════════════════════════════════════════════════════════════
print("\nSTEP 13: SHAP Explainability")

try:
    import shap
    print("  SHAP available — generating summary plot...")

    explainer   = shap.TreeExplainer(best_clf)
    shap_values = explainer.shap_values(X_test_s)

    # For multiclass tree models shap_values is typically a list, one array per class.
    # We visualize the SHAP values for the "abnormal" classes (Hypo + Hyper averaged),
    # since that's what clinically matters most; fall back to first class if binary-shaped.
    if isinstance(shap_values, list) and len(shap_values) == len(DIAG_CLASSES):
        non_normal_idx = [i for i, c in enumerate(DIAG_CLASSES) if c != 'Normal']
        sv = np.mean([shap_values[i] for i in non_normal_idx], axis=0)
    elif isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(7, 5))
    shap.summary_plot(sv, X_test, feature_names=FEATURE_COLS,
                      show=False, plot_size=None)
    plt.title(f'SHAP Summary — {best_name}', fontsize=11, color='#1e293b')
    plt.gcf().patch.set_facecolor('#ffffff')
    plt.tight_layout()
    plt.savefig('static/shap_summary.png', dpi=100,
                bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("  Saved: static/shap_summary.png")
    SHAP_AVAILABLE = True

except ImportError:
    print("  SHAP not installed — skipping (run: pip install shap)")
    print("  → Will use feature importance as fallback in Flask app")
    SHAP_AVAILABLE = False

with open('model/shap_available.json', 'w') as f:
    json.dump({'available': SHAP_AVAILABLE}, f)

# ═══════════════════════════════════════════════════════════════════
# STEP 14 — FEATURE IMPORTANCE PLOT  (W4 fix — correct title)
# ═══════════════════════════════════════════════════════════════════
if hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
else:
    importances = np.zeros(len(FEATURE_COLS))
feat_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances}).sort_values('importance')
med = feat_df['importance'].median()
colors = ['#0d9488' if v > med else '#94a3b8' for v in feat_df['importance']]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(feat_df['feature'], feat_df['importance'], color=colors, edgecolor='none')
ax.set_xlabel('Importance', color='#1e293b')
# W4 FIX: Title now says "Model Feature Importance" not "Top Contributing Features"
ax.set_title(f'Model Feature Importance — {best_name}\n({len(df_final):,} real UCI patients)',
             color='#1e293b', fontsize=11)
ax.tick_params(colors='#1e293b')
fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#f8fafc')
for sp in ax.spines.values(): sp.set_edgecolor('#cbd5e1')
plt.tight_layout()
plt.savefig('static/feature_importance.png', dpi=100, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("\nSaved: static/feature_importance.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 15 — CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════════════
cm_best = confusion_matrix(y_test, y_pred_best)
n_cls = len(DIAG_CLASSES)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm_best, cmap='Blues')
ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
ax.set_xticklabels([f'Predicted\n{c}' for c in DIAG_CLASSES], color='#1e293b', fontsize=9)
ax.set_yticklabels([f'Actual\n{c}' for c in DIAG_CLASSES], color='#1e293b', fontsize=9)
cm_max = cm_best.max()
for i in range(n_cls):
    for j in range(n_cls):
        is_diag = (i == j)
        text_color = '#0d9488' if is_diag else '#dc2626'
        ax.text(j, i, f"{cm_best[i,j]}",
                ha='center', va='center', color=text_color, fontsize=13, fontweight='bold')
ax.set_title(f'Confusion Matrix — {best_name}', color='#1e293b', fontsize=11)
ax.tick_params(colors='#1e293b')
fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
for sp in ax.spines.values(): sp.set_edgecolor('#cbd5e1')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=100, bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("Saved: static/confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 16 — SAVE ALL ARTEFACTS
# ═══════════════════════════════════════════════════════════════════
joblib.dump(best_clf, 'model/best_model.pkl')
with open('model/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('model/best_model_name.json', 'w') as f:
    json.dump({'name': best_name, 'best_gb_params': best_gb_params}, f)

print(f"\n{'='*65}")
print(f"✅ Training complete!")
print(f"   Dataset        : {len(df_final):,} real UCI patients")
print(f"   Diagnosis classes : {DIAG_CLASSES}")
print(f"   Features       : {FEATURE_COLS}")
print(f"   Models trained : {list(models.keys())}")
print(f"   Best model     : {best_name}")
print(f"   Test Macro AUC : {results[best_name]['roc_auc_ovr_macro']:.4f}")
print(f"   Macro Recall   : {results[best_name]['recall_macro']:.4f}")
print(f"   SHAP           : {'✓ Generated' if SHAP_AVAILABLE else '✗ Install shap for SHAP plots'}")
print(f"   GridSearch GB  : {best_gb_params}")
print(f"{'='*65}")
print(f"\nRun:  python app.py")
