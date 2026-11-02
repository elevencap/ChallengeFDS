import json
from collections import defaultdict, Counter
import numpy as np

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]


def build_db(paths=("train.jsonl", "test.jsonl"),
             tipi_path="pokemon_tipi.json", out_path="pokemon_db.json"):
    with open(tipi_path, encoding="utf-8") as fh:
        tipi = json.load(fh)

    vals = defaultdict(lambda: {s: [] for s in STATS})   # nome -> stat osservate
    types = {}
    items = defaultdict(Counter)                          # solo p1: di p2 gli item non si vedono
    n_p1, n_p2_lead, n_p2_seen = Counter(), Counter(), Counter()

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                b = json.loads(line)
                for m in b["p1_team_details"]:
                    for s in STATS:
                        vals[m["name"]][s].append(m[s])
                    types[m["name"]] = m["types"]
                    items[m["name"]][m["item"]] += 1
                    n_p1[m["name"]] += 1
                lead = b["p2_lead_details"]
                for s in STATS:
                    vals[lead["name"]][s].append(lead[s])
                types[lead["name"]] = lead["types"]
                n_p2_lead[lead["name"]] += 1
                # specie p2 viste in timeline (anche quelle senza dettagli)
                for name in {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}:
                    n_p2_seen[name] += 1

    db, unstable = {}, []
    for name, d in vals.items():
        entry = {"types": types[name]}
        for s in STATS:
            entry[s] = float(np.mean(d[s]))
            if np.std(d[s]) > 0:
                unstable.append((name, s))
        entry["n_p1_team"] = n_p1[name]
        entry["n_p2_lead"] = n_p2_lead[name]
        entry["n_p2_seen"] = n_p2_seen[name]
        if items[name]:
            entry["top_item"] = items[name].most_common(1)[0][0]
        db[name] = entry

    # specie viste solo in timeline: tipi dal vecchio json, stat da riempire con la media globale
    missing = [n for n in n_p2_seen if n not in db]
    glob = {s: float(np.mean([e[s] for e in db.values()])) for s in STATS}
    for name in missing:
        db[name] = {"types": tipi.get(name, ["notype", "notype"]), **glob,
                    "n_p1_team": 0, "n_p2_lead": 0, "n_p2_seen": n_p2_seen[name],
                    "stats_imputed": True}

    db["_global"] = glob   # media su tutte le specie viste, per gli sconosciuti

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"{len(db) - 1} specie | {len(missing)} con stat imputate: {missing[:10]}")
    print(f"stat non costanti tra occorrenze: {unstable[:10] or 'nessuna'}")
    return db


if __name__ == "__main__":
    build_db()