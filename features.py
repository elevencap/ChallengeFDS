import json
import numpy as np
import pandas as pd
from type_chart import Type, TYPES, effectiveness

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ("atk", "def", "spa", "spd", "spe")
TEAM_SIZE = 6

with open("pokemon_db.json", encoding="utf-8") as fh:
    DB = json.load(fh)

GLOBAL = DB.pop("_global")
TIPI = {k: v["types"] for k, v in DB.items()}

TYPES_BY_NAME = {}

def populate_types_map(records):
    """Raccoglie i tipi di tutti i Pokémon visibili nei dettagli di team e lead."""
    for b in records:
        candidates = list(b.get("p1_team_details", []))
        lead = b.get("p2_lead_details")
        if lead:
            candidates.append(lead)
        for pokemon in candidates:
            name = (pokemon.get("name") or "").strip().lower()
            types = [
                t.lower()
                for t in pokemon.get("types", [])
                if t and t.lower() != "notype"
            ]
            if name and types:
                TYPES_BY_NAME[name] = types

def stat_name(s):
    return s.replace("base_", "")

def real_types(types):
    return [Type(t) for t in types if t != "notype"]

def phys_bulk(m):
    return m["base_hp"] * m["base_def"]

def spec_bulk(m):
    return m["base_hp"] * m["base_spd"]

def mixed_bulk(m):
    return m["base_hp"] * (m["base_def"] + m["base_spd"])

def move_eff(move, defender_name):
    if not move:
        return None
    
    move_name = (move.get("name") or "").strip().lower()
    move_type = (move.get("type") or "").strip().lower()
    base_power = move.get("base_power", 1)
    
    if move_name == "hiddenpower" or (move_type == "normal" and base_power == 0):
        return None

    def_name_clean = (defender_name or "").strip().lower()
    def_types = TYPES_BY_NAME.get(def_name_clean)

    try:
        if def_types:
            return effectiveness(Type(move["type"]), *real_types(def_types))
        else:
            return effectiveness(Type(move["type"]), *real_types(TIPI.get(defender_name, [])))
    except (KeyError, ValueError):
        return None

def has_quad_weakness(types, attacker_types=TYPES):
    return any(effectiveness(a, *types) >= 4 for a in attacker_types)

def norm_press(atk, dfn):
    return (atk - dfn) / (atk + dfn + 1e-5)

# ---------- Feature Statiche (Essenziali) ----------
def static_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    p1_lead_name = b["battle_timeline"][0]["p1_pokemon_state"]["name"]
    p1_lead = next((m for m in team if m["name"] == p1_lead_name), team[0])

    holders = [m for m in team if m.get("item") == "leftovers"]
    leftovers_bulk = sum(mixed_bulk(m) for m in holders) / 1e4

    p1_atk_mean = np.mean([m["base_atk"] for m in team])
    p1_spa_mean = np.mean([m["base_spa"] for m in team])
    p1_def_mean = np.mean([m["base_def"] for m in team])
    p1_spd_mean = np.mean([m["base_spd"] for m in team])

    p1_phys_press_vs_lead = norm_press(p1_atk_mean, lead["base_def"])
    p1_spec_press_vs_lead = norm_press(p1_spa_mean, lead["base_spd"])
    p1_static_off_press = (p1_phys_press_vs_lead + p1_spec_press_vs_lead) / 2

    p2_lead_phys_press_vs_team = norm_press(lead["base_atk"], p1_def_mean)
    p2_lead_spec_press_vs_team = norm_press(lead["base_spa"], p1_spd_mean)
    p2_static_off_press = (p2_lead_phys_press_vs_team + p2_lead_spec_press_vs_team) / 2

    static_net_pressure_diff = (p1_static_off_press - p2_static_off_press) / 2

    f = {
        "phys_bulk_diff": np.mean([phys_bulk(m) for m in team]) - phys_bulk(lead),
        "spec_bulk_diff": np.mean([spec_bulk(m) for m in team]) - spec_bulk(lead),
        "n_quad_weak": sum(has_quad_weakness(real_types(m["types"])) for m in team),
        "leftovers_bulk": leftovers_bulk,
        "static_net_pressure_diff": static_net_pressure_diff
    }

    for s in STATS:
        n = stat_name(s)
        vals = [m[s] for m in team]
        f[f"lead_{n}_diff"] = p1_lead[s] - lead[s]
        f[f"team_{n}_mean_diff"] = np.mean(vals) - lead[s]

    return f

# ---------- Feature Team P2 Rivelato ----------
def revealed_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    seen_names = list(dict.fromkeys(
        [lead["name"]] + [t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]]
    ))
    n_seen = len(seen_names)
    n_unknown = max(TEAM_SIZE - n_seen, 0)
    total = n_seen + n_unknown

    p2 = [DB.get(n, GLOBAL) for n in seen_names]
    p2_types = [real_types(TYPES_BY_NAME.get(n.strip().lower(), TIPI.get(n, []))) for n in seen_names]
    p1_types = [real_types(m["types"]) for m in team]

    f = {
        "p2_n_seen": n_seen
    }

    p2_est = {}
    for s in STATS:
        n = stat_name(s)
        est_val = (sum(m[s] for m in p2) + n_unknown * GLOBAL[s]) / total
        p2_est[s] = est_val
        f[f"p2team_{n}_diff"] = np.mean([m[s] for m in team]) - est_val

    p2_spe = np.array([m["base_spe"] for m in p2])
    f["p1_faster_than_seen_frac"] = np.mean(
        [[m["base_spe"] > sp for sp in p2_spe] for m in team]
    )
    f["quad_weak_frac_diff"] = np.mean([has_quad_weakness(t) for t in p1_types]) - np.mean([has_quad_weakness(t) for t in p2_types])

    return f

# ---------- Feature Timeline (I pilastri del controllo) ----------
def side_stats(tl, side, other):
    states = [t[f"{side}_pokemon_state"] for t in tl]
    opp = [t[f"{other}_pokemon_state"] for t in tl]
    moves = [t[f"{side}_move_details"] for t in tl]
    opp_moves = [t[f"{other}_move_details"] for t in tl]
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

    return {
        "hp_min": hp.min() if len(hp) > 0 else 0.0,
        "hp_last": hp[-1] if len(hp) > 0 else 0.0,
        "team_hp": (sum(last_hp.values()) + (TEAM_SIZE - len(last_hp))) / TEAM_SIZE,
        "fainted": sum(v == 0 for v in last_hp.values()),
        "first_faint": first_faint,
        "switches": switches,
        "status_frac": np.mean([s["status"] not in ("nostatus", "fnt") for s in states]),
        "sleep_frac": np.mean([s["status"] == "slp" for s in states]),
        "boost_net": np.mean([sum(s["boosts"][k] for k in BOOSTS) for s in states]),
        "last_hp": last_hp
    }

def timeline_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    a, o = side_stats(tl, "p1", "p2"), side_stats(tl, "p2", "p1")

    p1_team = b["p1_team_details"]
    p1_main_tank = max(p1_team, key=mixed_bulk).get("name", "unknown")
    p1_tank_hp = a["last_hp"].get(p1_main_tank, 1.0)

    p2_lead = b["p2_lead_details"]
    seen_names = list(dict.fromkeys([p2_lead["name"]] + [t["p2_pokemon_state"]["name"] for t in tl]))
    if seen_names:
        p2_main_tank = max(seen_names, key=lambda nm: mixed_bulk(DB.get(nm, GLOBAL)))
        p2_tank_hp = o["last_hp"].get(p2_main_tank, 1.0)
    else:
        p2_tank_hp = 1.0

    # Calcolo copertura superefficace (type_coverage_score_diff)
    p1_super_effective_given = 0
    p1_moves_thrown = 0
    p1_super_effective_taken = 0
    p2_moves_thrown = 0

    for t in tl:
        m1, m2 = t.get("p1_move_details"), t.get("p2_move_details")
        p1_state = t.get("p1_pokemon_state")
        p2_state = t.get("p2_pokemon_state")

        if p2_state and m1 and m1.get("category") != "status":
            p1_moves_thrown += 1
            eff = move_eff(m1, p2_state.get("name"))
            if eff is not None and eff > 1.0:
                p1_super_effective_given += 1
                
        if p1_state and m2 and m2.get("category") != "status":
            p2_moves_thrown += 1
            eff_taken = move_eff(m2, p1_state.get("name"))
            if eff_taken is not None and eff_taken > 1.0:
                p1_super_effective_taken += 1

    p1_cov_ratio = (p1_super_effective_given / p1_moves_thrown) if p1_moves_thrown > 0 else 0.0
    p2_cov_ratio = (p1_super_effective_taken / p2_moves_thrown) if p2_moves_thrown > 0 else 0.0

    board_control_diff = (
        (o["status_frac"] - a["status_frac"]) + 
        (o["sleep_frac"] - a["sleep_frac"]) + 
        (a["switches"] - o["switches"]) * 0.02
    )

    return {
        "hp_min_diff": a["hp_min"] - o["hp_min"],
        "hp_last_diff": a["hp_last"] - o["hp_last"],
        "team_hp_diff": a["team_hp"] - o["team_hp"],
        "fainted_diff": o["fainted"] - a["fainted"],
        "tank_hp_diff": p1_tank_hp - p2_tank_hp,
        "type_coverage_score_diff": p1_cov_ratio - p2_cov_ratio,
        "board_control_diff": board_control_diff,
        "boost_diff": a["boost_net"] - o["boost_net"],
    }

def team_hp_series(tl, side):
    last, out = {}, []
    for t in tl:
        s = t[f"{side}_pokemon_state"]
        last[s["name"]] = 0.0 if s["status"] == "fnt" else s["hp_pct"]
        out.append((sum(last.values()) + (TEAM_SIZE - len(last))) / TEAM_SIZE)
    return np.array(out, dtype=float)

def status_inflicted(tl, side):
    """Conteggio status pulito dal bug del Rest."""
    total = slp = par = 0
    for i in range(1, len(tl)):
        prev, cur = tl[i-1], tl[i]
        prev_move = prev.get(f"{side}_move_details")
        if prev_move and prev_move.get("name") == "rest":
            continue
        ps, cs = prev[f"{side}_pokemon_state"], cur[f"{side}_pokemon_state"]
        if ps["name"] == cs["name"] and ps["status"] == "nostatus" and cs["status"] not in ("nostatus", "fnt"):
            total += 1
            slp += cs["status"] == "slp"
            par += cs["status"] == "par"
    return total, slp, par

def move_usage(tl, side):
    moves = [t.get(f"{side}_move_details") for t in tl]
    used = [m for m in moves if m]
    n = max(len(tl), 1)
    status_n = sum(m.get("category") == "status" for m in used)
    hazards = sum(m.get("name") in ("spikes",) for m in used)
    
    setup_moves = {"swordsdance", "curse", "amnesia"}
    phazing_moves = {"roar", "whirlwind", "haze"}
    screen_moves = {"reflect", "lightscreen"}
    inflict_status_moves = {"toxic", "thunderwave", "hypnosis", "spore"}
    
    return {
        "status_frac": status_n / n,
        "hazards": hazards,
        "setup": sum(m.get("name") in setup_moves for m in used),
        "phaze": sum(m.get("name") in phazing_moves for m in used),
        "screen": sum(m.get("name") in screen_moves for m in used),
        "inflict_status": sum(m.get("name") in inflict_status_moves for m in used)
    }

def extra_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    f = {"n_turns": n}

    h1, h2 = team_hp_series(tl, "p1"), team_hp_series(tl, "p2")
    f["team_hp_ratio"] = (h1[-1] + 0.05) / (h2[-1] + 0.05)

    s1, s2 = status_inflicted(tl, "p1"), status_inflicted(tl, "p2")
    f["status_inflicted_diff"] = s2[0] - s1[0]
    f["sleep_inflicted_diff"] = s2[1] - s1[1]

    u1, u2 = move_usage(tl, "p1"), move_usage(tl, "p2")
    f["status_move_frac_diff"] = u1["status_frac"] - u2["status_frac"]
    f["hazard_moves_diff"] = u1["hazards"] - u2["hazards"]
    f["setup_moves_diff"] = u1["setup"] - u2["setup"]
    f["phaze_moves_diff"] = u1["phaze"] - u2["phaze"]
    f["screen_moves_diff"] = u1["screen"] - u2["screen"]
    f["inflict_status_moves_diff"] = u1["inflict_status"] - u2["inflict_status"]

    return {k: (0.0 if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in f.items()}

def dmg_stats(tl, side):
    taken = recent = healed = 0.0
    n = len(tl)
    for i in range(1, n):
        ps, cs = tl[i - 1][f"{side}_pokemon_state"], tl[i][f"{side}_pokemon_state"]
        if ps["name"] != cs["name"]:
            continue
        ph = 0.0 if ps["status"] == "fnt" else ps["hp_pct"]
        ch = 0.0 if cs["status"] == "fnt" else cs["hp_pct"]
        delta = ph - ch
        if delta > 0:
            taken += delta
            if i >= n - 10:
                recent += delta
        else:
            healed += -delta
    return taken, recent, healed

def new_features(b):
    tl = b["battle_timeline"]
    f = {}

    t1, r1, h1 = dmg_stats(tl, "p1")
    t2, r2, h2 = dmg_stats(tl, "p2")
    f["heal_diff"] = h1 - h2

    # Stato dell'ultimo turno (fondamentale)
    s1, s2 = tl[-1]["p1_pokemon_state"], tl[-1]["p2_pokemon_state"]
    bad = lambda s: int(s["status"] not in ("nostatus", "fnt"))
    f["last_status_diff"] = bad(s2) - bad(s1)
    f["last_active_fnt_diff"] = int(s2["status"] == "fnt") - int(s1["status"] == "fnt")
    f["last_boost_diff"] = sum(s1["boosts"][k] for k in BOOSTS) - sum(s2["boosts"][k] for k in BOOSTS)

    return {k: (0.0 if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in f.items()}

def last_hp_map(tl, side):
    last = {}
    for t in tl:
        st = t[f"{side}_pokemon_state"]
        last[st["name"]] = 0.0 if st["status"] == "fnt" else st["hp_pct"]
    return last

# Le migliori feature non lineari premiate dai modelli (Top importance)
def v4_features(b):
    tl = b["battle_timeline"]
    team = b["p1_team_details"]
    f = {}

    h1, h2 = team_hp_series(tl, "p1"), team_hp_series(tl, "p2")
    lr = np.log((h1 + 0.05) / (h2 + 0.05))
    f["log_hp_ratio"] = lr[-1]

    l1, l2 = last_hp_map(tl, "p1"), last_hp_map(tl, "p2")
    a1 = sum(v > 0 for v in l1.values()) + max(TEAM_SIZE - len(l1), 0)
    a2 = sum(v > 0 for v in l2.values()) + max(TEAM_SIZE - len(l2), 0)
    f["alive_log_ratio"] = float(np.log((a1 + 0.5) / (a2 + 0.5)))
    f["hp_adv_x_alive"] = (h1[-1] - h2[-1]) * (a1 + a2) / (2 * TEAM_SIZE)
    f["hp_logratio_x_alive_logratio"] = f["log_hp_ratio"] * f["alive_log_ratio"]

    return {k: (0.0 if (isinstance(v, float) and not np.isfinite(v)) else float(v)) for k, v in f.items()}

def flatten(b):
    row = {
        "battle_id": b["battle_id"],
        **static_features(b),
        **revealed_features(b),
        **timeline_features(b),
        **extra_features(b),
        **new_features(b),
        **v4_features(b),
    }
    if "player_won" in b:
        row["target"] = int(b["player_won"])
    return row

def build(path):
    with open(path, "r", encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    populate_types_map(records)
    return pd.DataFrame([flatten(b) for b in records])

if __name__ == "__main__":
    train = build("train.jsonl")
    train.to_csv("battles_features.csv", index=False)
    print(f"Dataset pulito e snellito estratto con successo: {train.shape}")