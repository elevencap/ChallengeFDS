import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# =========================================================
# 1. CARICAMENTO DATI DI TRAIN DA battles_features.csv
# =========================================================
print("Caricamento dataset di addestramento da 'battles_features.csv'...")

if not os.path.exists("battles_features.csv"):
    raise FileNotFoundError("Il file 'battles_features.csv' non è stato trovato nella cartella del progetto.")

train_df = pd.read_csv("battles_features.csv")

target_col = "target" if "target" in train_df.columns else "player_won"
X_train = train_df.drop(columns=["battle_id", target_col], errors="ignore")
y_train = train_df[target_col].astype(int)

cat_cols = [c for c in X_train.columns 
            if c.endswith("_name") or c.endswith("_item") or c.endswith("status_last")]

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")

print(f"Dataset di addestramento pronto: {X_train.shape[0]} battaglie, {X_train.shape[1]} feature.")

# =========================================================
# 2. CARICAMENTO O ESTRAZIONE FEATURE DI TEST (test.jsonl)
# =========================================================
test_file = "test_features.csv"
test_df = None

# Tenta la lettura rapida da test_features.csv
if os.path.exists(test_file):
    try:
        temp_df = pd.read_csv(test_file)
        if not temp_df.empty:
            test_df = temp_df
            print("Caricamento rapido feature di test da 'test_features.csv'.")
        else:
            os.remove(test_file)
    except Exception:
        os.remove(test_file)

# Estrazione da test.jsonl se test_features.csv non è disponibile
if test_df is None:
    print("Estrazione feature da 'test.jsonl' in corso...")
    STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
    MAX_TURNS = 30

    def static_features(b):
        f = {}
        team = b.get("p1_team_details", [])
        for i, mon in enumerate(team):
            f[f"p1_{i}_name"] = mon.get("name", "")
            f[f"p1_{i}_item"] = mon.get("item", "")
            for s in STATS:
                f[f"p1_{i}_{s}"] = mon.get(s, 0)

        for s in STATS:
            vals = [m[s] for m in team if s in m]
            f[f"p1_mean_{s}"] = np.mean(vals) if vals else 0
            f[f"p1_max_{s}"] = np.max(vals) if vals else 0
            f[f"p1_min_{s}"] = np.min(vals) if vals else 0

        for t, c in Counter(t for m in team for t in m.get("types", []) if t != "notype").items():
            f[f"cnt_p1_type_{t}"] = c
        for it, c in Counter(m.get("item", "") for m in team if m.get("item")).items():
            f[f"cnt_p1_item_{it}"] = c

        lead = b.get("p2_lead_details", {})
        f["p2_lead_name"] = lead.get("name", "")
        for s in STATS:
            f[f"p2_lead_{s}"] = lead.get(s, 0)
        for t in lead.get("types", []):
            if t != "notype":
                f[f"cnt_p2_lead_type_{t}"] = 1
        return f

    def timeline_features(b):
        tl = b.get("battle_timeline", [])
        f = {"n_turns": len(tl)}
        for side in ("p1", "p2"):
            states = [t[f"{side}_pokemon_state"] for t in tl if f"{side}_pokemon_state" in t]
            moves = [t[f"{side}_move_details"] for t in tl if f"{side}_move_details" in t]
            hp = [s.get("hp_pct", 0) for s in states]
            used = [m for m in moves if m is not None]

            f[f"{side}_hp_mean"] = np.mean(hp) if hp else 0
            f[f"{side}_hp_min"] = np.min(hp) if hp else 0
            f[f"{side}_hp_last"] = hp[-1] if hp else 0
            f[f"{side}_n_fainted"] = len({s["name"] for s in states if s.get("status") == "fnt"})
            f[f"{side}_n_unique_seen"] = len({s["name"] for s in states if "name" in s})
            f[f"{side}_n_switches"] = sum(1 for m in moves if m is None)
            f[f"{side}_turns_with_status"] = sum(1 for s in states if s.get("status") not in ("nostatus", None))
            f[f"{side}_status_last"] = states[-1]["status"] if states and "status" in states[-1] else "nostatus"

            cats = Counter(m.get("category") for m in used if m and "category" in m)
            for c in ("physical", "special", "status"):
                f[f"{side}_n_{c}_moves"] = cats.get(c, 0)
            dmg = [m.get("base_power", 0) for m in used if m and m.get("category") != "status"]
            f[f"{side}_bp_mean"] = np.mean(dmg) if dmg else 0
            f[f"{side}_n_priority"] = sum(1 for m in used if m and m.get("priority", 0) != 0)

            for k in ("atk", "def", "spa", "spd", "spe"):
                vals = [s.get("boosts", {}).get(k, 0) for s in states if "boosts" in s]
                f[f"{side}_boost_{k}_max"] = max(vals) if vals else 0
                f[f"{side}_boost_{k}_last"] = vals[-1] if vals else 0

            for t in range(MAX_TURNS):
                f[f"{side}_hp_t{t+1}"] = hp[t] if t < len(hp) else np.nan

        f["hp_diff_last"] = f["p1_hp_last"] - f["p2_hp_last"]
        f["fainted_diff"] = f["p2_n_fainted"] - f["p1_n_fainted"]
        return f

    def flatten(b):
        return {"battle_id": b.get("battle_id"), **static_features(b), **timeline_features(b)}

    records = []
    with open("test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(flatten(json.loads(line)))
    test_df = pd.DataFrame(records)
    test_df.to_csv(test_file, index=False)

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