import json
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
'''
def retrieve_items():
    items = set()

    with open("train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            battle = json.loads(line)
            for mon in battle.get("p1_team_details", []):
                item = mon.get("item")
                if item:
                    items.add(item)

    items_list = sorted(items)
    print(len(items_list))
    print(items_list)

import json

with open("train.jsonl", "r", encoding="utf-8") as f:
    first = json.loads(f.readline())

print(first)


counts = Counter()

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        battle = json.loads(line)
        for mon in battle.get("p1_team_details", []):
            item = mon.get("item")
            if item:  # salta None / stringa vuota
                counts[item] += 1

total = sum(counts.values())
print(f"Item distinti: {len(counts)}  |  Occorrenze totali: {total}\n")

for item, n in counts.most_common():
    print(f"{item:<20} {n:>6}  {n / total:6.1%}")

items_list = sorted(counts)  # la lista degli item unici, se ti serve ancora

import json
from collections import Counter


def count_turns(path):
    """Distribuzione della lunghezza della timeline per battaglia."""
    counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            battle = json.loads(line)
            counts[len(battle.get("battle_timeline", []))] += 1

    total = sum(counts.values())
    print(f"{path}: {total} battaglie, {len(counts)} durate distinte")
    for n_turns, n in sorted(counts.items()):
        print(f"  {n_turns:>3} turni  {n:>6}  {n / total:6.1%}")
    return counts




def analizza_status_da_dataset(file_path):
    # Mosse che infliggono o sono associate a sonno/congelamento
    SLEEP_MOVES = {"hypnosis", "sleep powder", "spore", "sing", "lovely kiss", "yawn", "relic song"}
    FREEZE_MOVES = {"ice beam", "blizzard", "ice punch", "powder snow", "ice fang", "freeze-dry"}

    total_turns = 0
    
    sleep_attempts = 0
    sleep_inflicted = 0
    
    freeze_attempts = 0
    freeze_inflicted = 0

    # Legge il file riga per riga (formato JSONL)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            battle = json.loads(line)
            timeline = battle.get("battle_timeline", [])
            
            for turn in timeline:
                total_turns += 1
                
                # Analizziamo la mossa di P1 se presente
                p1_move = turn.get("p1_move_details")
                if p1_move and isinstance(p1_move, dict):
                    move_name = p1_move.get("name", "").lower()
                    if move_name in SLEEP_MOVES:
                        sleep_attempts += 1
                    elif move_name in FREEZE_MOVES:
                        freeze_attempts += 1

                # Controlliamo lo stato finale dei Pokémon nei turni per vedere se lo status è attivo/inflitto
                # (Guardiamo sia P1 che P2 attraverso gli stati nei turni)
                p1_state = turn.get("p1_pokemon_state", {})
                p2_state = turn.get("p2_pokemon_state", {})
                
                if p1_state.get("status") == "slp":
                    sleep_inflicted += 1
                elif p1_state.get("status") == "frz":
                    freeze_inflicted += 1
                    
                if p2_state.get("status") == "slp":
                    sleep_inflicted += 1
                elif p2_state.get("status") == "frz":
                    freeze_inflicted += 1

    # Calcolo delle percentuali
    pct_sleep_per_attempt = (sleep_inflicted / sleep_attempts * 100) if sleep_attempts > 0 else 0
    pct_freeze_per_attempt = (freeze_inflicted / freeze_attempts * 100) if freeze_attempts > 0 else 0
    
    pct_sleep_total = (sleep_inflicted / total_turns * 100) if total_turns > 0 else 0
    pct_freeze_total = (freeze_inflicted / total_turns * 100) if total_turns > 0 else 0

    print("=== RISULTATI ANALISI STATUS (DATASET TRAIN.JSONL) ===")
    print(f"Totale turni/eventi analizzati: {total_turns}\n")
    
    print(f"SLEEP:")
    print(f"  - Tentativi (mosse dedicate usate): {sleep_attempts}")
    print(f"  - Volte rilevato come status: {sleep_inflicted}")
    print(f"  - Tasso di successo sulle mosse usate: {pct_sleep_per_attempt:.2f}%")
    print(f"  - Percentuale sul totale dei turni: {pct_sleep_total:.2f}%\n")
    
    print(f"FREEZE:")
    print(f"  - Tentativi (mosse Gelide usate): {freeze_attempts}")
    print(f"  - Volte rilevato come status: {freeze_inflicted}")
    print(f"  - Tasso di successo sulle mosse usate: {pct_freeze_per_attempt:.2f}%")
    print(f"  - Percentuale sul totale dei turni: {pct_freeze_total:.2f}%")
'''
 


def evaluate_f_scores(input_csv="features_train.csv", output_csv="f_scores.csv"):
    # 1. Carica il dataset delle feature
    df = pd.read_csv(input_csv)
    
    if "target" not in df.columns:
        raise ValueError("Il colonna 'target' non è presente nel dataset.")
    
    # 2. Separa feature (X) e target (y)
    # Escludiamo eventuali ID o colonne non numeriche/non-feature
    ignore_cols = ["battle_id", "target"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Gestione eventuali valori mancanti (NaN) con l'imputazione della media
    if X.isnull().values.any():
        X = X.fillna(X.mean())
    
    # 3. Calcolo ANOVA F-score e p-value con f_classif
    f_values, p_values = f_classif(X, y)
    
    # 4. Creazione DataFrame dei risultati
    # 4. Creazione DataFrame dei risultati (MODIFICATO: ascending=True)
    results_df = pd.DataFrame({
        "feature": feature_cols,
        "f_score": f_values,
        "p_value": p_values
    })
    
    # Ordina per F-score crescente (le feature meno discriminanti in alto)
    results_df = results_df.sort_values(by="f_score", ascending=True).reset_index(drop=True)
    
    # 5. Salva i risultati su file CSV
    results_df.to_csv(output_csv, index=False)
    
    print(f"---------------F-SCORE------------")
    print("\nTop 15 Feature PEGGIORI con F-score più basso:")
    print(results_df.head(15).to_string(index=False))


def train_and_evaluate_tree_model(input_csv="features_train.csv", output_csv="tree_feature_importances.csv"):
    # 1. Carica il dataset delle feature
    df = pd.read_csv(input_csv)
    
    if "target" not in df.columns:
        raise ValueError("La colonna 'target' non è presente nel dataset.")
    
    # 2. Separa feature (X) e target (y)
    ignore_cols = ["battle_id", "target"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols]
    y = df["target"]
    
    # Imputazione dei valori mancanti con la media (se presenti)
    if X.isnull().values.any():
        X = X.fillna(X.mean())
    
    # 3. Inizializza e addestra un modello a blocchi / basato su alberi (Random Forest)
    # Usiamo 200 alberi con una profondità massima per evitare overfitting
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Valutazione in Cross-Validation (5-fold) per verificare la qualità del modello
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print("--- Performance del Modello in Cross-Validation (5-Fold) ---")
    print(f"Accuracy media: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"ROC-AUC media:  {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})\n")
    
    # 5. Fit finale sull'intero dataset di train
    model.fit(X, y)
    
# 6. Estrazione delle feature importances (MODIFICATO: ascending=True)
    importances = model.feature_importances_
    
    results_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=True).reset_index(drop=True)
    
    # Salva le importanze su file CSV
    results_df.to_csv(output_csv, index=False)
    
    print(f"--------ALBERO------------")
    print("\nTop 15 Feature PEGGIORI con importanza minore:")
    print(results_df.head(15).to_string(index=False))

if __name__ == "__main__":
    try:
        train_and_evaluate_tree_model()
    except FileNotFoundError:
        print("File 'features_train.csv' non trovato. Generalo prima tramite features.py!")
    try:
            evaluate_f_scores()
    except FileNotFoundError:
        print("File 'features_train.csv' non trovato. Assicurati di aver generato prima il CSV con features.py!")
