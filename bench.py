import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
# Se vuoi includere anche XGBoost, basta decommentare le righe corrispondenti
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

# Preprocessing categoriche specifico per modello:
# - CatBoost vuole stringhe (o la lista delle feature categoriche)
# - Random Forest (e XGBoost) richiedono codifica numerica (cat.codes)

X_tr_cb, X_te_cb = X_train.copy(), X_test.copy()
for col in cat_cols:
    X_tr_cb[col] = X_tr_cb[col].astype(str)
    X_te_cb[col] = X_te_cb[col].astype(str)

X_tr_enc, X_te_enc = X_train.copy(), X_test.copy()
for col in cat_cols:
    X_tr_enc[col] = X_tr_enc[col].astype("category").cat.codes
    X_te_enc[col] = X_te_enc[col].astype("category").cat.codes

print(f"Dataset pronto: {X_train.shape[0]} battaglie di train, {X_test.shape[0]} di test.")

# =========================================================
# 2. DEFINIZIONE DEI MODELLI
# =========================================================
def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=15, random_state=42, n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            cat_features=cat_cols, random_state=42, verbose=0
        ),
        # XGBoost facoltativo se vuoi fare un ensemble a 3:
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=42, eval_metric="logloss", n_jobs=-1
        )
    }

# =========================================================
# 3. VALUTAZIONE IN CROSS-VALIDATION (5-FOLD) PER L'ENSEMBLE
# =========================================================
print("\n--- VALUTAZIONE ENSEMBLE IN CROSS-VALIDATION (5-FOLD) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 1. Random Forest (dati codificati)
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr_enc.iloc[train_idx], y_tr)
    p_rf_val = rf.predict_proba(X_tr_enc.iloc[val_idx])[:, 1]

    # 2. CatBoost (dati in stringa)
    cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, cat_features=cat_cols, random_state=42, verbose=0)
    cb.fit(X_tr_cb.iloc[train_idx], y_tr)
    p_cb_val = cb.predict_proba(X_tr_cb.iloc[val_idx])[:, 1]
    
    # Media delle probabilità (Soft Voting)
    p_ens_val = (p_rf_val + p_cb_val) / 2.0
    preds_val = (p_ens_val >= 0.5).astype(int)
    
    score = accuracy_score(y_va, preds_val)
    cv_scores.append(score)
    print(f"Fold {fold}: Accuratezza Ensemble = {score * 100:.2f}%")

print(f"Accuratezza Media CV Ensemble: {np.mean(cv_scores)*100:.2f}% (± {np.std(cv_scores)*100:.2f}%)")

# =========================================================
# 4. ADDESTRAMENTO FINALE E PREDIZIONE SUBMISSION
# =========================================================
print("\nAddestramento finale sui dati completi...")

rf_final = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf_final.fit(X_tr_enc, y_train)
p_rf_test = rf_final.predict_proba(X_te_enc)[:, 1]

cb_final = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, cat_features=cat_cols, random_state=42, verbose=0)
cb_final.fit(X_tr_cb, y_train)
p_cb_test = cb_final.predict_proba(X_te_cb)[:, 1]

# Soft Voting finale
p_test_avg = (p_rf_test + p_cb_test) / 2.0
final_predictions = (p_test_avg >= 0.5).astype(int)

submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": final_predictions
})

submission.to_csv("submission.csv", index=False)
print("\n[OK] 'submission.csv' generato con successo unendo Random Forest e CatBoost!")