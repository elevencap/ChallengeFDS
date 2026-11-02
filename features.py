import json
import numpy as np
import pandas as pd
from type_chart import Type, TYPES, effectiveness

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ("atk", "def", "spa", "spd", "spe")
TEAM_SIZE = 6  # p2: i Pokémon non ancora visti si contano a HP pieni

# ---------- database Pokémon (generato da build_pokemon_db.py) ----------
with open("pokemon_db.json", encoding="utf-8") as fh:
    DB = json.load(fh)

GLOBAL = DB.pop("_global")                      # stat medie su tutte le specie viste
TIPI = {k: v["types"] for k, v in DB.items()}   # nome -> tipi (compatibile con move_eff)


def stat_name(s):
    return s.replace("base_", "")


def real_types(types):
    return [Type(t) for t in types if t != "notype"]


def phys_bulk(m):
    return m["base_hp"] * m["base_def"]


def spec_bulk(m):
    return m["base_hp"] * m["base_spd"]


def move_eff(move, defender_name):
    """Efficacia di una mossa sul Pokémon avversario attivo."""
    try:
        return effectiveness(Type(move["type"]), *real_types(TIPI[defender_name]))
    except (KeyError, ValueError):
        return None


def has_quad_weakness(types, attacker_types=TYPES):
    """True se almeno un tipo in attacker_types colpisce con efficacia >= 4
    un Pokémon con i tipi `types` (lista di Type, già senza 'notype')."""
    return any(effectiveness(a, *types) >= 4 for a in attacker_types)


def best_stab(att_types, def_types):
    """Miglior efficacia STAB di un attaccante contro un difensore (1.0 se senza tipi)."""
    return max((effectiveness(t, *def_types) for t in att_types), default=1.0)


# ---------- item: solo leftovers ----------
def item_features(team):
    holders = [m for m in team if m["item"] == "leftovers"]
    return {
        "n_leftovers": len(holders),
        # recupero 1/16 HP a turno: rende di più su chi regge molto
        "leftovers_bulk": sum(m["base_hp"] * (m["base_def"] + m["base_spd"])
                              for m in holders) / 1e4,
    }


# ---------- statiche: p1 (team completo) vs lead p2 ----------
def static_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    lead_types = real_types(lead["types"])
    p1_lead_name = b["battle_timeline"][0]["p1_pokemon_state"]["name"]
    p1_lead = next((m for m in team if m["name"] == p1_lead_name), team[0])

    off = [max(effectiveness(t, *lead_types) for t in real_types(m["types"])) for m in team]
    dfn = [max(effectiveness(t, *real_types(m["types"])) for t in lead_types) for m in team]

    f = {
        "n_faster_than_lead": sum(m["base_spe"] > lead["base_spe"] for m in team),
        "n_slower_than_lead": sum(m["base_spe"] < lead["base_spe"] for m in team),
        "phys_bulk_diff": np.mean([phys_bulk(m) for m in team]) - phys_bulk(lead),
        "spec_bulk_diff": np.mean([spec_bulk(m) for m in team]) - spec_bulk(lead),
        "n_quad_weak": sum(has_quad_weakness(real_types(m["types"])) for m in team),
        "n_quad_weak_to_lead": sum(has_quad_weakness(real_types(m["types"]), lead_types)
                                   for m in team),
        "team_off_vs_lead": np.mean(off),    # quanto il mio team colpisce il lead avversario
        "lead_off_vs_team": np.mean(dfn),    # quanto il lead avversario colpisce il mio team
        "matchup_diff": np.mean(off) - np.mean(dfn),
        "n_super_effective_answers": sum(o >= 2 for o in off),
    }

    # stat spacchettate: una feature per ogni stat base, nessuna somma
    for s in STATS:
        n = stat_name(s)
        vals = [m[s] for m in team]
        f[f"lead_{n}_diff"] = p1_lead[s] - lead[s]          # mio lead vs lead avversario
        f[f"team_{n}_mean_diff"] = np.mean(vals) - lead[s]  # media team vs lead avversario
        f[f"team_{n}_max_diff"] = max(vals) - lead[s]       # miglior Pokémon del team vs lead

    f.update(item_features(team))
    return f


# ---------- team p2 rivelato in timeline (stat dal DB) ----------
def revealed_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]

    # specie p2 viste, nell'ordine di comparsa (lead incluso)
    seen = list(dict.fromkeys(
        [lead["name"]] + [t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]]
    ))
    n_seen = len(seen)
    n_unknown = max(TEAM_SIZE - n_seen, 0)
    total = n_seen + n_unknown

    p2 = [DB.get(n, GLOBAL) for n in seen]
    p2_types = [real_types(TIPI.get(n, [])) for n in seen]
    p1_types = [real_types(m["types"]) for m in team]

    f = {"p2_n_seen": n_seen}

    # stat del team p2 stimato: rivelati + sconosciuti alla media globale
    for s in STATS:
        n = stat_name(s)
        p2_est = (sum(m[s] for m in p2) + n_unknown * GLOBAL[s]) / total
        f[f"p2team_{n}_diff"] = np.mean([m[s] for m in team]) - p2_est   # p1 - p2

    p2_phys = (sum(phys_bulk(m) for m in p2) + n_unknown * phys_bulk(GLOBAL)) / total
    p2_spec = (sum(spec_bulk(m) for m in p2) + n_unknown * spec_bulk(GLOBAL)) / total
    f["p2team_phys_bulk_diff"] = np.mean([phys_bulk(m) for m in team]) - p2_phys
    f["p2team_spec_bulk_diff"] = np.mean([spec_bulk(m) for m in team]) - p2_spec

    # velocità: quanti dei miei sono più veloci dei p2 visti (media sui visti)
    p2_spe = np.array([m["base_spe"] for m in p2])
    f["p1_faster_than_seen_frac"] = np.mean(
        [[m["base_spe"] > sp for sp in p2_spe] for m in team]
    )

    # matchup di tipo: team p1 vs team p2 rivelato (media su tutte le coppie)
    off = np.mean([[best_stab(at, dt) for dt in p2_types] for at in p1_types])
    dfn = np.mean([[best_stab(at, dt) for dt in p1_types] for at in p2_types])
    f["seen_off"] = off                 # quanto colpisco i p2 visti
    f["seen_dfn"] = dfn                 # quanto mi colpiscono i p2 visti
    f["seen_matchup_diff"] = off - dfn

    # debolezze x4 del team p2 rivelato (frazione sui visti, per non dipendere da n_seen)
    f["p2_quad_weak_frac"] = np.mean([has_quad_weakness(t) for t in p2_types])
    f["p1_quad_weak_frac"] = np.mean([has_quad_weakness(t) for t in p1_types])
    f["quad_weak_frac_diff"] = f["p1_quad_weak_frac"] - f["p2_quad_weak_frac"]

    return f


# ---------- timeline: statistiche per lato, poi differenza ----------
def side_stats(tl, side, other):
    states = [t[f"{side}_pokemon_state"] for t in tl]
    opp = [t[f"{other}_pokemon_state"] for t in tl]
    moves = [t[f"{side}_move_details"] for t in tl]
    hp = np.array([0.0 if s["status"] == "fnt" else s["hp_pct"] for s in states])

    last_hp, first_faint = {}, None
    for i, s in enumerate(states):
        last_hp[s["name"]] = 0.0 if s["status"] == "fnt" else s["hp_pct"]
        if s["status"] == "fnt" and first_faint is None:
            first_faint = i

    switches = sum(
        1 for i in range(1, len(states))
        if states[i]["name"] != states[i - 1]["name"] and states[i - 1]["status"] != "fnt"
    )
    used = [m for m in moves if m is not None]
    dmg = [(m, o) for m, o in zip(moves, opp) if m is not None and m["category"] != "status"]
    effs = [e for m, o in dmg if (e := move_eff(m, o["name"])) is not None]

    return {
        "hp": hp,
        "team_hp": (sum(last_hp.values()) + (TEAM_SIZE - len(last_hp))) / TEAM_SIZE,
        "fainted": sum(v == 0 for v in last_hp.values()),
        "first_faint": first_faint,
        "switches": switches,
        "status_frac": np.mean([s["status"] not in ("nostatus", "fnt") for s in states]),
        "sleep_frac": np.mean([s["status"] == "slp" for s in states]), # <-- NUOVA FEATURE DI STATO SLEEP
        "boost_net": np.mean([sum(s["boosts"][k] for k in BOOSTS) for s in states]),
        "bp_mean": np.mean([m["base_power"] for m, _ in dmg]) if dmg else 0.0,
        "status_move_frac": (len(used) - len(dmg)) / max(len(used), 1),
        "move_eff": np.mean(effs) if effs else 1.0,
    }


def timeline_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    a, o = side_stats(tl, "p1", "p2"), side_stats(tl, "p2", "p1")
    d = a["hp"] - o["hp"]

    return {
        "team_hp_diff": a["team_hp"] - o["team_hp"],
        "fainted_diff": o["fainted"] - a["fainted"],            # KO fatti - KO subiti
        "first_ko": (a["first_faint"] if a["first_faint"] is not None else n)
                    - (o["first_faint"] if o["first_faint"] is not None else n),
        "active_hp_mean_diff": d.mean(),
        "active_hp_last5_diff": d[-5:].mean(),
        "active_hp_last_diff": d[-1],
        "status_pressure": o["status_frac"] - a["status_frac"],  # status subiti dal rivale - miei
        "sleep_pressure": o["sleep_frac"] - a["sleep_frac"],    # <-- NUOVA DIFFERENZA DI SLEEP (avversario - miei)
        "switches_diff": a["switches"] - o["switches"],
        "boost_diff": a["boost_net"] - o["boost_net"],
        "bp_diff": a["bp_mean"] - o["bp_mean"],
        "status_move_diff": a["status_move_frac"] - o["status_move_frac"],
        "move_eff_diff": a["move_eff"] - o["move_eff"],
    }



def flatten(b):
    row = {
        "battle_id": b["battle_id"],
        **static_features(b),
        **revealed_features(b),
        **timeline_features(b),
    }
    if "player_won" in b:
        row["target"] = int(b["player_won"])
    return row


def build(path):
    with open(path, "r", encoding="utf-8") as fh:
        return pd.DataFrame([flatten(json.loads(l)) for l in fh if l.strip()])


if __name__ == "__main__":
    train = build("train.jsonl")
    train.to_csv("features_train.csv", index=False)
    print(train.shape)
    try:
        test = build("test.jsonl")
        test.to_csv("features_test.csv", index=False)
        print(test.shape)
    except FileNotFoundError:
        pass