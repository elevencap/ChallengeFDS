import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from features import build

# =========================================================
# 1. COSTRUZIONE FEATURE DI TRAIN (train.jsonl) CON features.build
# =========================================================
print("Estrazione feature da 'train.jsonl' in corso...")
train_df = build("train.jsonl")
train_df.to_csv("features_train.csv", index=False)  # solo per ispezione, non viene riletto

target_col = "target" if "target" in train_df.columns else "player_won"
X_train = train_df.drop(columns=["battle_id", target_col], errors="ignore")
y_train = train_df[target_col].astype(int)

# Con features.py tutte le colonne sono numeriche: cat_cols resta vuoto,
# ma lo mantengo così il resto del codice non cambia.
cat_cols = [c for c in X_train.columns
            if c.endswith("_name") or c.endswith("_item") or c.endswith("status_last")]

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")

print(f"Dataset di addestramento pronto: {X_train.shape[0]} battaglie, {X_train.shape[1]} feature.")

# =========================================================
# 2. COSTRUZIONE FEATURE DI TEST (test.jsonl) CON LA STESSA FUNZIONE
# =========================================================
print("Estrazione feature da 'test.jsonl' in corso...")
test_df = build("test.jsonl")
test_df.to_csv("features_test.csv", index=False)  # solo per ispezione, non viene riletto

# Allineamento colonne e categoriche tra Train e Test
X_test = test_df.reindex(columns=X_train.columns, fill_value=0)

for col in cat_cols:
    X_test[col] = X_test[col].astype("category")

print(f"Dataset di test pronto: {X_test.shape[0]} battaglie.")

# =========================================================
# 3. VALUTAZIONE ACCURATEZZA IN CROSS-VALIDATION
# =========================================================
print("\n--- VALUTAZIONE IN CROSS-VALIDATION (5-FOLD) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    fold_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    fold_model.fit(X_tr, y_tr)
    
    score = fold_model.score(X_va, y_va)
    cv_scores.append(score)
    print(f"Fold {fold}: Accuratezza = {score * 100:.2f}%")

print(f"Accuratezza Media CV: {np.mean(cv_scores)*100:.2f}% (± {np.std(cv_scores)*100:.2f}%)")

# =========================================================
# 4. ADDESTRAMENTO FINALE E AGGIORNAMENTO DI SUBMISSION.CSV
# =========================================================
print("\nAddestramento finale di LightGBM sulle 10.000 battaglie...")
model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train, y_train)

print("Generazione delle nuove predizioni per il test set...")
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": predictions
})

submission.to_csv("submission.csv", index=False)
print("\n[OK] 'submission.csv' aggiornato con successo con 5.000 predizioni!")