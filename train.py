import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from features import build

# =========================================================
# 1. COSTRUZIONE E PREPARAZIONE FEATURE (train & test)
# =========================================================
print("Estrazione feature da 'train.jsonl' e 'test.jsonl'...")
train_df = build("train.jsonl")
test_df = build("test.jsonl")

target_col = "target" if "target" in train_df.columns else "player_won"
X_train = train_df.drop(columns=["battle_id", target_col], errors="ignore")
y_train = train_df[target_col].astype(int)

# Allineamento colonne sul test set
X_test = test_df.reindex(columns=X_train.columns, fill_value=0)

cat_cols = [c for c in X_train.columns
            if c.endswith("_name") or c.endswith("_item") or c.endswith("status_last")]

# Preprocessing categoriche (sia Extra Trees che XGBoost richiedono codifica numerica)
X_tr_enc, X_te_enc = X_train.copy(), X_test.copy()
for col in cat_cols:
    X_tr_enc[col] = X_tr_enc[col].astype("category").cat.codes
    X_te_enc[col] = X_te_enc[col].astype("category").cat.codes

print(f"Dataset pronto: {X_train.shape[0]} battaglie di train, {X_test.shape[0]} di test.")

# =========================================================
# 2. DEFINIZIONE DEI PARAMETRI DEI MODELLI
# =========================================================

ET_PARAMS = dict(
    n_estimators=500, 
    max_depth=8, 
    min_samples_leaf=10,
    max_features=0.4, 
    random_state=42, 
    n_jobs=-1
)

XGB_PARAMS = dict(
    n_estimators=500, 
    learning_rate=0.03, 
    max_depth=6,
    random_state=42, 
    eval_metric="logloss", 
    n_jobs=-1
)

# =========================================================
# 3. VALUTAZIONE IN CROSS-VALIDATION (5-FOLD) PER L'ENSEMBLE
# =========================================================
print("\n--- VALUTAZIONE ENSEMBLE (Extra Trees + XGBoost) IN 5-FOLD CV ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 1. Extra Trees
    et = ExtraTreesClassifier(**ET_PARAMS)
    et.fit(X_tr_enc.iloc[train_idx], y_tr)
    p_et_val = et.predict_proba(X_tr_enc.iloc[val_idx])[:, 1]

    # 2. XGBoost
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_tr_enc.iloc[train_idx], y_tr)
    p_xgb_val = xgb.predict_proba(X_tr_enc.iloc[val_idx])[:, 1]
    
    # Media delle probabilità (Soft Voting)
    p_ens_val = (p_et_val + p_xgb_val) / 2.0
    preds_val = (p_ens_val >= 0.5).astype(int)
    
    score = accuracy_score(y_va, preds_val)
    cv_scores.append(score)
    print(f"Fold {fold}: Accuratezza Ensemble = {score * 100:.2f}%")

print(f"Accuratezza Media CV Ensemble: {np.mean(cv_scores)*100:.2f}% (± {np.std(cv_scores)*100:.2f}%)")

# =========================================================
# 4. ADDESTRAMENTO FINALE E PREDIZIONE SUBMISSION
# =========================================================
print("\nAddestramento finale sui dati completi...")

et_final = ExtraTreesClassifier(**ET_PARAMS)
et_final.fit(X_tr_enc, y_train)
p_et_test = et_final.predict_proba(X_te_enc)[:, 1]

xgb_final = XGBClassifier(**XGB_PARAMS)
xgb_final.fit(X_tr_enc, y_train)
p_xgb_test = xgb_final.predict_proba(X_te_enc)[:, 1]

# Soft Voting finale
p_test_avg = (p_et_test + p_xgb_test) / 2.0
final_predictions = (p_test_avg >= 0.5).astype(int)

submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": final_predictions
})

submission.to_csv("submission.csv", index=False)
print("\n[OK] 'submission.csv' generato con successo unendo Extra Trees e XGBoost!")