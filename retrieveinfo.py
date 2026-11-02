import json
from collections import Counter
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
'''



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

# Esegui la funzione puntando al tuo file:
# analizza_status_da_dataset("train.jsonl")

# Esempio di utilizzo:
# analizza_status_pokemon("battles.json")
if __name__ == "__main__":
    analizza_status_da_dataset("train.jsonl")
    ###count_turns("train.jsonl")
    ###count_turns("test.jsonl")
    pass