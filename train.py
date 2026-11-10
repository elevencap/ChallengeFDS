import os
import warnings
warnings.filterwarnings("ignore")
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

from features import build

# =========================================================
# 1. CARICAMENTO DATI DA FEATURES.PY (69 FEATURE PULITE)
# =========================================================
print("Estrazione feature pulite da 'train.jsonl' e 'test.jsonl'...")
train_df = build("train.jsonl")
test_df = build("test.jsonl")

target_col = "target" if "target" in train_df.columns else "player_won"
X_train = train_df.drop(columns=["battle_id", target_col], errors="ignore")
y_train = train_df[target_col].astype(int)

X_test = test_df.reindex(columns=X_train.columns, fill_value=0)

# Frequency Encoding per le categoriche (_name)
cat_cols = [c for c in X_train.columns if c.endswith("_name")]
for col in cat_cols:
    freq = X_train[col].value_counts().to_dict()
    X_train[col] = X_train[col].map(freq).fillna(0).astype(float)
    X_test[col] = X_test[col].map(freq).fillna(0).astype(float)

print(f"Dataset pronto: {X_train.shape[0]} battaglie, {X_train.shape[1]} feature pulite.")

# =========================================================
# 2. CONFIGURAZIONE IPERPARAMETRI PER I 3 MODELLI AD ALBERO
# =========================================================

# Modello 1: LightGBM (Leaf-wise)
LGB_PARAMS = dict(
    n_estimators=550,
    learning_rate=0.02,
    num_leaves=14,
    max_depth=4,
    min_child_samples=45,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_alpha=2.5,
    reg_lambda=4.0,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

# Modello 2: CatBoost (Symmetric / Oblivious Trees)
CAT_SYM_PARAMS = dict(
    iterations=650,
    learning_rate=0.02,
    depth=5,
    grow_policy='SymmetricTree',
    l2_leaf_reg=4.0,
    random_seed=42,
    verbose=0,
    thread_count=-1
)

# Modello 3: CatBoost (Lossguide / Leaf-wise Trees)
CAT_LOSS_PARAMS = dict(
    iterations=600,
    learning_rate=0.02,
    depth=5,
    grow_policy='Lossguide',
    max_leaves=16,
    l2_leaf_reg=5.0,
    random_seed=101,
    verbose=0,
    thread_count=-1
)

# =========================================================
# 3. 5-FOLD CROSS-VALIDATION
# =========================================================
print("\n--- ADDESTRAMENTO ENSEMBLE TREE-BASED IN 5-FOLD CV ---")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_cat_sym = np.zeros(len(X_train))
oof_cat_loss = np.zeros(len(X_train))

train_preds_lgb = np.zeros(len(X_train))
train_preds_cat_sym = np.zeros(len(X_train))
train_preds_cat_loss = np.zeros(len(X_train))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat_sym = np.zeros(len(X_test))
test_preds_cat_loss = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    # 1. LightGBM
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    p_lgb_tr = lgb_model.predict_proba(X_tr)[:, 1]
    p_lgb_va = lgb_model.predict_proba(X_va)[:, 1]
    train_preds_lgb[train_idx] += p_lgb_tr / (n_splits - 1)
    oof_lgb[val_idx] = p_lgb_va
    test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

    # 2. CatBoost Symmetric
    cat_sym = CatBoostClassifier(**CAT_SYM_PARAMS)
    cat_sym.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    p_cs_tr = cat_sym.predict_proba(X_tr)[:, 1]
    p_cs_va = cat_sym.predict_proba(X_va)[:, 1]
    train_preds_cat_sym[train_idx] += p_cs_tr / (n_splits - 1)
    oof_cat_sym[val_idx] = p_cs_va
    test_preds_cat_sym += cat_sym.predict_proba(X_test)[:, 1] / n_splits

    # 3. CatBoost Lossguide
    cat_loss = CatBoostClassifier(**CAT_LOSS_PARAMS)
    cat_loss.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    p_cl_tr = cat_loss.predict_proba(X_tr)[:, 1]
    p_cl_va = cat_loss.predict_proba(X_va)[:, 1]
    train_preds_cat_loss[train_idx] += p_cl_tr / (n_splits - 1)
    oof_cat_loss[val_idx] = p_cl_va
    test_preds_cat_loss += cat_loss.predict_proba(X_test)[:, 1] / n_splits

    fold_tr = (p_lgb_tr + p_cs_tr + p_cl_tr) / 3.0
    fold_va = (p_lgb_va + p_cs_va + p_cl_va) / 3.0

    tr_acc = accuracy_score(y_tr, (fold_tr >= 0.5).astype(int))
    va_acc = accuracy_score(y_va, (fold_va >= 0.5).astype(int))
    
    print(f"Fold {fold}: Train Acc = {tr_acc * 100:.2f}% | Val Acc = {va_acc * 100:.2f}% | Gap = {(tr_acc - va_acc) * 100:.2f}%")

# Accuratezze OOF singole
acc_lgb = accuracy_score(y_train, (oof_lgb >= 0.5).astype(int))
acc_cs = accuracy_score(y_train, (oof_cat_sym >= 0.5).astype(int))
acc_cl = accuracy_score(y_train, (oof_cat_loss >= 0.5).astype(int))

print("\n" + "-" * 65)
print(f"Accuratezza OOF LightGBM (Leaf-wise):       {acc_lgb * 100:.2f}%")
print(f"Accuratezza OOF CatBoost (Symmetric):        {acc_cs * 100:.2f}%")
print(f"Accuratezza OOF CatBoost (Lossguide):        {acc_cl * 100:.2f}%")
print("-" * 65)

# =========================================================
# 4. OTTIMIZZAZIONE PESI BLENDING VIA SLSQP
# =========================================================
print("\nCalcolo dei pesi ottimali via SLSQP...")

def blend_loss(weights):
    w1, w2, w3 = weights
    blend = (w1 * oof_lgb) + (w2 * oof_cat_sym) + (w3 * oof_cat_loss)
    return log_loss(y_train, blend)

bounds = [(0, 1), (0, 1), (0, 1)]
constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
res_w = minimize(blend_loss, [0.34, 0.33, 0.33], method='SLSQP', bounds=bounds, constraints=constraints)

w1, w2, w3 = res_w.x
w_sum = w1 + w2 + w3
w_lgb, w_cs, w_cl = w1 / w_sum, w2 / w_sum, w3 / w_sum

oof_blend = (w_lgb * oof_lgb) + (w_cs * oof_cat_sym) + (w_cl * oof_cat_loss)
train_blend = (w_lgb * train_preds_lgb) + (w_cs * train_preds_cat_sym) + (w_cl * train_preds_cat_loss)

# =========================================================
# 5. OTTIMIZZAZIONE SOGLIA DI DECISIONE E REPORT
# =========================================================
def opt_threshold_loss(thresh):
    return -accuracy_score(y_train, (oof_blend >= thresh).astype(int))

res_t = minimize_scalar(opt_threshold_loss, bounds=(0.35, 0.65), method='bounded')
best_threshold = res_t.x
best_val_acc = -res_t.fun
final_train_acc = accuracy_score(y_train, (train_blend >= best_threshold).astype(int))
final_auc = roc_auc_score(y_train, oof_blend)

print("=" * 65)
print("             RISULTATI FINALI TRIPLE-TREE ENSEMBLE            ")
print("=" * 65)
print(f"Pesi: LGBM = {w_lgb:.4f} | CatBoost Sym = {w_cs:.4f} | CatBoost Loss = {w_cl:.4f}")
print(f"Soglia Ottimale: {best_threshold:.4f}")
print("-" * 65)
print(f"Train Acc:                {final_train_acc * 100:.2f}%")
print(f"Val Acc (OOF):            {best_val_acc * 100:.2f}%")
print(f"Gap di Generalizzazione: {(final_train_acc - best_val_acc) * 100:.2f}%")
print(f"ROC-AUC Finale OOF:       {final_auc:.4f}")
print("=" * 65)

# =========================================================
# 6. SUBMISSION FINALE
# =========================================================
final_test_prob = (w_lgb * test_preds_lgb) + (w_cs * test_preds_cat_sym) + (w_cl * test_preds_cat_loss)
submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": (final_test_prob >= best_threshold).astype(int)
})
submission.to_csv("submission.csv", index=False)
print(f"\n[OK] 'submission.csv' generato con successo!")