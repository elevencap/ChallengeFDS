import json
import numpy as np
import pandas as pd
from collections import Counter

from type_chart import TYPES, effectiveness, clean_types

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
MAX_TURNS = 30  # aumenta se nel dataset ci sono battaglie più lunghe

# nome -> tipi (senza 'notype'); serve per conoscere i tipi dei Pokémon di p2
# che nel dataset compaiono solo per nome nella timeline
with open("pokemon_tipi.json", "r", encoding="utf-8") as fh:
    POKEMON_TYPES = {name: clean_types(ts) for name, ts in json.load(fh).items()}


def types_of(name):
    return POKEMON_TYPES.get(name, [])


# ---------------------------------------------------------------- type helpers
def move_eff(m, def_types):
    """Efficacia di una mossa offensiva contro i tipi del difensore (None se non applicabile)."""
    if m is None or m["category"] == "status" or not def_types:
        return None
    try:
        return effectiveness(m["type"], *def_types)
    except ValueError:  # tipo mossa non presente nella chart
        return None


def static_features(b):
    f = {}
    team = b["p1_team_details"]

    # per-slot (utile se l'ordine ha significato, es. il primo è il lead)
    for i, mon in enumerate(team):
        f[f"p1_{i}_name"] = mon["name"]
        f[f"p1_{i}_item"] = mon["item"]
        for s in STATS:
            f[f"p1_{i}_{s}"] = mon[s]

    # aggregate invarianti all'ordine
    for s in STATS:
        vals = [m[s] for m in team]
        f[f"p1_mean_{s}"] = np.mean(vals)
        f[f"p1_max_{s}"] = np.max(vals)
        f[f"p1_min_{s}"] = np.min(vals)

    # multi-hot (conteggi) di tipi e item
    for t, c in Counter(t for m in team for t in m["types"] if t != "notype").items():
        f[f"cnt_p1_type_{t}"] = c
    for it, c in Counter(m["item"] for m in team).items():
        f[f"cnt_p1_item_{it}"] = c

    # lead p2
    lead = b["p2_lead_details"]
    f["p2_lead_name"] = lead["name"]
    for s in STATS:
        f[f"p2_lead_{s}"] = lead[s]
    for t in lead["types"]:
        if t != "notype":
            f[f"cnt_p2_lead_type_{t}"] = 1
    return f


def matchup_features(b):
    """Feature basate sulla type chart, senza ipotesi sui moveset (non sono nel dataset).

    - profilo difensivo dei tipi (debolezze/resistenze) di p1 e dei Pokémon p2 visti
    - efficacia delle mosse EFFETTIVAMENTE usate nella timeline
    """
    f = {}
    tl = b["battle_timeline"]
    p1_team = [clean_types(m["types"]) for m in b["p1_team_details"]]
    lead_name = b["p2_lead_details"]["name"]
    lead_types = clean_types(b["p2_lead_details"]["types"])

    # Pokémon p2 visti (lead + timeline); tipi da pokemon_tipi.json
    seen = list(dict.fromkeys([lead_name] + [t["p2_pokemon_state"]["name"] for t in tl]))
    p2_seen = [lead_types if n == lead_name else types_of(n) for n in seen]
    p2_seen = [t for t in p2_seen if t]

    # --- 1) profilo difensivo per tipo attaccante
    for at in TYPES:
        f[f"p2_lead_mult_{at.value}"] = effectiveness(at, *lead_types)
        f[f"p1_n_weak_{at.value}"] = sum(1 for t in p1_team if effectiveness(at, *t) >= 2)
        f[f"p2_seen_n_weak_{at.value}"] = sum(1 for t in p2_seen if effectiveness(at, *t) >= 2)

    # --- 2) aggregati: n. di tipi attaccanti a cui il Pokémon è debole/resistente
    def n_weak(t):
        return sum(1 for at in TYPES if effectiveness(at, *t) >= 2)

    def n_resist(t):
        return sum(1 for at in TYPES if effectiveness(at, *t) < 1)

    f["p1_mean_n_weak"] = np.mean([n_weak(t) for t in p1_team])
    f["p1_mean_n_resist"] = np.mean([n_resist(t) for t in p1_team])
    f["p2_seen_mean_n_weak"] = np.mean([n_weak(t) for t in p2_seen]) if p2_seen else np.nan
    f["p2_seen_mean_n_resist"] = np.mean([n_resist(t) for t in p2_seen]) if p2_seen else np.nan
    f["weak_diff"] = f["p2_seen_mean_n_weak"] - f["p1_mean_n_weak"]
    f["resist_diff"] = f["p1_mean_n_resist"] - f["p2_seen_mean_n_resist"]

    # --- 3) efficacia delle mosse usate contro il difensore attivo (dato osservato)
    for side, opp in (("p1", "p2"), ("p2", "p1")):
        effs, mtypes = [], set()
        for t in tl:
            m = t[f"{side}_move_details"]
            if m is not None and m["category"] != "status":
                mtypes.add(m["type"])
            e = move_eff(m, types_of(t[f"{opp}_pokemon_state"]["name"]))
            if e is not None:
                effs.append(e)
        f[f"{side}_move_eff_mean"] = np.mean(effs) if effs else np.nan
        f[f"{side}_n_super_eff"] = sum(1 for e in effs if e >= 2)
        f[f"{side}_n_resisted"] = sum(1 for e in effs if 0 < e < 1)
        f[f"{side}_n_immune_hit"] = sum(1 for e in effs if e == 0)
        f[f"{side}_n_move_types"] = len(mtypes)
    f["move_eff_diff"] = f["p1_move_eff_mean"] - f["p2_move_eff_mean"]
    f["super_eff_diff"] = f["p1_n_super_eff"] - f["p2_n_super_eff"]
    return f


def timeline_features(b):
    tl = b["battle_timeline"]
    f = {"n_turns": len(tl)}

    for side in ("p1", "p2"):
        states = [t[f"{side}_pokemon_state"] for t in tl]
        moves = [t[f"{side}_move_details"] for t in tl]
        hp = [s["hp_pct"] for s in states]
        used = [m for m in moves if m is not None]

        f[f"{side}_hp_mean"] = np.mean(hp)
        f[f"{side}_hp_min"] = np.min(hp)
        f[f"{side}_hp_last"] = hp[-1]
        f[f"{side}_n_fainted"] = len({s["name"] for s in states if s["status"] == "fnt"})
        f[f"{side}_n_unique_seen"] = len({s["name"] for s in states})
        f[f"{side}_n_switches"] = sum(1 for m in moves if m is None)  # turni senza mossa
        f[f"{side}_turns_with_status"] = sum(1 for s in states if s["status"] not in ("nostatus",))
        f[f"{side}_status_last"] = states[-1]["status"]

        # mosse
        cats = Counter(m["category"] for m in used)
        for c in ("physical", "special", "status"):
            f[f"{side}_n_{c}_moves"] = cats.get(c, 0)
        dmg = [m["base_power"] for m in used if m["category"] != "status"]
        f[f"{side}_bp_mean"] = np.mean(dmg) if dmg else 0
        f[f"{side}_n_priority"] = sum(1 for m in used if m["priority"] != 0)

        # boost
        for k in ("atk", "def", "spa", "spd", "spe"):
            vals = [s["boosts"][k] for s in states]
            f[f"{side}_boost_{k}_max"] = max(vals)
            f[f"{side}_boost_{k}_last"] = vals[-1]

        # andamento HP turno per turno (padding NaN)
        for t in range(MAX_TURNS):
            f[f"{side}_hp_t{t+1}"] = hp[t] if t < len(hp) else np.nan

    f["hp_diff_last"] = f["p1_hp_last"] - f["p2_hp_last"]
    f["fainted_diff"] = f["p2_n_fainted"] - f["p1_n_fainted"]
    return f


def flatten(b):
    return {"battle_id": b["battle_id"], **static_features(b), **matchup_features(b),
            **timeline_features(b), "target": b["player_won"]}


rows = []
with open("train.jsonl", "r", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            rows.append(flatten(json.loads(line)))

df = pd.DataFrame(rows)

# i conteggi assenti sono 0, non NaN
cnt_cols = [c for c in df.columns if c.startswith("cnt_")]
df[cnt_cols] = df[cnt_cols].fillna(0).astype(int)

# colonne categoriche (nomi, item, status)
cat_cols = [c for c in df.columns
            if c.endswith("_name") or c.endswith("_item") or c.endswith("status_last")]
df[cat_cols] = df[cat_cols].astype("category")

df.to_csv("battles_features.csv", index=False)
print(df.shape)