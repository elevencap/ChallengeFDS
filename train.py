import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from features import build

# =========================================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# =========================================================
print("Estrazione feature da 'train.jsonl' e 'test.jsonl'...")
train_df = build("train.jsonl")
test_df = build("test.jsonl")

target_col = "target" if "target" in train_df.columns else "player_won"
X_train = train_df.drop(columns=["battle_id", target_col], errors="ignore")
y_train = train_df[target_col].astype(int)

X_test = test_df.reindex(columns=X_train.columns, fill_value=0)

# Categoriali mappate in frequenza per CatBoost
cat_cols = [c for c in X_train.columns if c.endswith("_name")]
for col in cat_cols:
    freq = X_train[col].value_counts().to_dict()
    X_train[col] = X_train[col].map(freq).fillna(0).astype(float)
    X_test[col] = X_test[col].map(freq).fillna(0).astype(float)

# Creazione sotto-insieme pulito per XGBoost (senza feature categoriali rumorose)
X_train_xgb = X_train.drop(columns=cat_cols)
X_test_xgb = X_test.drop(columns=cat_cols)

print(f"Dataset pronto: {X_train.shape[0]} battaglie | CatBoost: {X_train.shape[1]} feature | XGBoost: {X_train_xgb.shape[1]} feature.")

# =========================================================
# 2. FEATURE WEIGHTS PRIORITARI PER CATBOOST
# =========================================================
feature_weights = [1.0] * X_train.shape[1]
priority_features = [
    "phys_bulk_diff",
    "spec_bulk_diff",
    "leftovers_bulk",
    "leftovers_survival_factor",
    "passive_status_ratio_diff",
    "team_hp_diff",
    "spikes_switch_pressure",
    "snorlax_hp_diff",
    "p2_snorlax_hp_seen",
    "status_pressure",
    "fainted_diff",
    "spikes_hp_interaction",
    "snorlax_relative_weight"
]

for feat in priority_features:
    if feat in X_train.columns:
        idx = X_train.columns.get_loc(feat)
        feature_weights[idx] = 1.8

# =========================================================
# 3. ENSEMBLE IBRIDO CON XGBOOST SEVERAMENTE REGOLARIZZATO
# =========================================================
SEEDS = [42, 101, 202]

print(f"\n--- ADDESTRAMENTO ENSEMBLE IBRIDO REGOLARIZZATO (MULTI-SEED {len(SEEDS)} SEEDS) ---")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

train_preds_cat = np.zeros(len(X_train))
train_preds_xgb = np.zeros(len(X_train))

test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    X_tr_xgb, X_va_xgb = X_train_xgb.iloc[train_idx], X_train_xgb.iloc[val_idx]

    fold_cat_tr = np.zeros(len(train_idx))
    fold_xgb_tr = np.zeros(len(train_idx))

    fold_cat_va = np.zeros(len(val_idx))
    fold_xgb_va = np.zeros(len(val_idx))

    for seed in SEEDS:
        # 1. CatBoost Symmetric (Depth 4, L2=8.0)
        c4 = CatBoostClassifier(
            iterations=700,
            learning_rate=0.02,
            depth=4,
            grow_policy='SymmetricTree',
            l2_leaf_reg=8.0,
            feature_weights=feature_weights,
            random_seed=seed,
            verbose=0,
            thread_count=-1
        )
        c4.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)

        fold_cat_tr += c4.predict_proba(X_tr)[:, 1] / len(SEEDS)
        fold_cat_va += c4.predict_proba(X_va)[:, 1] / len(SEEDS)
        test_preds_cat += c4.predict_proba(X_test)[:, 1] / (n_splits * len(SEEDS))

        # 2. XGBoost Severamente Regolarizzato (max_depth=3, reg_alpha=4.0, reg_lambda=8.0)
        xgb = XGBClassifier(
            n_estimators=600,
            learning_rate=0.015,
            max_depth=3,           # Profondità ridotta per spezzare l'overfitting
            reg_alpha=4.0,         # Penalità L1 elevata
            reg_lambda=8.0,        # Penalità L2 elevata
            subsample=0.75,        # Campionamento righe al 75%
            colsample_bytree=0.75, # Campionamento colonne al 75%
            random_state=seed,
            eval_metric="logloss",
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

        fold_xgb_tr += xgb.predict_proba(X_tr_xgb)[:, 1] / len(SEEDS)
        fold_xgb_va += xgb.predict_proba(X_va_xgb)[:, 1] / len(SEEDS)
        test_preds_xgb += xgb.predict_proba(X_test_xgb)[:, 1] / (n_splits * len(SEEDS))

    train_preds_cat[train_idx] += fold_cat_tr / (n_splits - 1)
    train_preds_xgb[train_idx] += fold_xgb_tr / (n_splits - 1)

    oof_cat[val_idx] = fold_cat_va
    oof_xgb[val_idx] = fold_xgb_va

    fold_tr = (fold_cat_tr + fold_xgb_tr) / 2.0
    fold_va = (fold_cat_va + fold_xgb_va) / 2.0

    tr_acc_fold = accuracy_score(y_tr, (fold_tr >= 0.5).astype(int))
    va_acc_fold = accuracy_score(y_va, (fold_va >= 0.5).astype(int))
    gap_fold = (tr_acc_fold - va_acc_fold) * 100

    print(f"Fold {fold}: Train Acc = {tr_acc_fold * 100:.2f}% | Val Acc = {va_acc_fold * 100:.2f}% | Gap = {gap_fold:.2f}%")

# =========================================================
# 4. BLENDING META-LEARNER ELASTICNET & CALIBRAZIONE OOF
# =========================================================
oof_matrix = np.column_stack([oof_cat, oof_xgb])
train_matrix = np.column_stack([train_preds_cat, train_preds_xgb])
test_matrix = np.column_stack([test_preds_cat, test_preds_xgb])

meta_learner = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.3,
    C=0.5,
    max_iter=1000,
    random_state=42
)
meta_learner.fit(oof_matrix, y_train)

train_raw_blend = meta_learner.predict_proba(train_matrix)[:, 1]
oof_raw_blend = meta_learner.predict_proba(oof_matrix)[:, 1]
test_raw_blend = meta_learner.predict_proba(test_matrix)[:, 1]

iso_calibrator = IsotonicRegression(out_of_bounds='clip')
oof_calibrated = iso_calibrator.fit_transform(oof_raw_blend, y_train)
test_calibrated = iso_calibrator.transform(test_raw_blend)

def opt_threshold_loss(thresh):
    return -accuracy_score(y_train, (oof_calibrated >= thresh).astype(int))

res_t = minimize_scalar(opt_threshold_loss, bounds=(0.35, 0.65), method='bounded')
best_threshold = res_t.x
best_val_acc = -res_t.fun
final_auc = roc_auc_score(y_train, oof_calibrated)

train_acc_final = accuracy_score(y_train, (train_raw_blend >= 0.5).astype(int))
overfitting_gap = (train_acc_final - best_val_acc) * 100

print("\n" + "=" * 65)
print(" RISULTATI ENSEMBLE CATBOOST + XGBOOST REGOLARIZZATO (SEVERO)  ")
print("=" * 65)
print(f"Pesi Meta-Learner (Coeff): CatBoost = {meta_learner.coef_[0][0]:.4f} | XGBoost = {meta_learner.coef_[0][1]:.4f}")
print(f"Intercetta Meta-Learner:   {meta_learner.intercept_[0]:.4f}")
print(f"Soglia Ottimale Calibrata: {best_threshold:.4f}")
print("-" * 65)
print(f"Accuratezza TRAIN Globale:       {train_acc_final * 100:.2f}%")
print(f"Accuratezza VAL OOF Calibrata:   {best_val_acc * 100:.2f}%")
print(f"GAP OVERFITTING (Train - Val):   {overfitting_gap:.2f}%")
print(f"ROC-AUC Finale OOF:              {final_auc:.4f}")
print("=" * 65)

# =========================================================
# 5. GENERAZIONE SUBMISSION
# =========================================================
submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": (test_calibrated >= best_threshold).astype(int)
})
submission.to_csv("submission.csv", index=False)
print(f"\n[OK] 'submission.csv' generato con successo con l'ensemble regolarizzato!")