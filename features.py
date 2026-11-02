import json
import numpy as np
import pandas as pd
from type_chart import Type, effectiveness

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ("atk", "def", "spa", "spd", "spe")
TEAM_SIZE = 6  # p2: i Pokémon non ancora visti si contano a HP pieni

# ---------- PESI (da tarare con la CV) ----------
WEIGHTS = {"base_hp": 1.0, "base_atk": 1.0, "base_def": 1.25,
           "base_spa": 1.0, "base_spd": 1.0, "base_spe": 1.0}

with open("pokemon_tipi.json", encoding="utf-8") as fh:
    TIPI = json.load(fh)  # nome -> tipi (per i Pokémon p2 visti in timeline)


def real_types(types):
    return [Type(t) for t in types if t != "notype"]


def bst(m):
    """Somma pesata delle stat base (difesa leggermente più pesante)."""
    return sum(WEIGHTS[s] * m[s] for s in STATS)


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


# ---------- item: solo leftovers ----------
def item_features(team):
    holders = [m for m in team if m["item"] == "leftovers"]
    return {
        "n_leftovers": len(holders),
        # recupero 1/16 HP a turno: rende di più su chi regge molto
        "leftovers_bulk": sum(m["base_hp"] * (m["base_def"] + m["base_spd"])
                              for m in holders) / 1e4,
    }


# ---------- statiche: già confrontate (p1 - p2) ----------
def static_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    lead_types = real_types(lead["types"])
    p1_lead_name = b["battle_timeline"][0]["p1_pokemon_state"]["name"]
    p1_lead = next((m for m in team if m["name"] == p1_lead_name), team[0])

    off = [max(effectiveness(t, *lead_types) for t in real_types(m["types"])) for m in team]
    dfn = [max(effectiveness(t, *real_types(m["types"])) for t in lead_types) for m in team]

    f = {
        "team_bst_vs_lead": np.mean([bst(m) for m in team]) - bst(lead),
        "lead_bst_diff": bst(p1_lead) - bst(lead),
        "lead_spe_diff": p1_lead["base_spe"] - lead["base_spe"],
        "n_faster_than_lead": sum(m["base_spe"] > lead["base_spe"] for m in team),
        "phys_bulk_diff": np.mean([phys_bulk(m) for m in team]) - phys_bulk(lead),
        "spec_bulk_diff": np.mean([spec_bulk(m) for m in team]) - spec_bulk(lead),
        "team_off_vs_lead": np.mean(off),    # quanto il mio team colpisce il lead avversario
        "lead_off_vs_team": np.mean(dfn),    # quanto il lead avversario colpisce il mio team
        "matchup_diff": np.mean(off) - np.mean(dfn),
        "n_super_effective_answers": sum(o >= 2 for o in off),
    }
    f.update(item_features(team))
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
        "n_turns": n,
        "team_hp_diff": a["team_hp"] - o["team_hp"],
        "fainted_diff": o["fainted"] - a["fainted"],            # KO fatti - KO subiti
        "first_ko": (a["first_faint"] if a["first_faint"] is not None else n)
                    - (o["first_faint"] if o["first_faint"] is not None else n),
        "active_hp_mean_diff": d.mean(),
        "active_hp_last5_diff": d[-5:].mean(),
        "active_hp_last_diff": d[-1],
        "status_pressure": o["status_frac"] - a["status_frac"],  # status subiti dal rivale - miei
        "switches_diff": a["switches"] - o["switches"],
        "boost_diff": a["boost_net"] - o["boost_net"],
        "bp_diff": a["bp_mean"] - o["bp_mean"],
        "status_move_diff": a["status_move_frac"] - o["status_move_frac"],
        "move_eff_diff": a["move_eff"] - o["move_eff"],
    }


def flatten(b):
    row = {"battle_id": b["battle_id"], **static_features(b), **timeline_features(b)}
    # il recupero passivo pesa di più nelle battaglie lunghe
    row["leftovers_x_turns"] = row["n_leftovers"] * row["n_turns"]
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