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

def stat_name(s):
    return s.replace("base_", "")

def real_types(types):
    return [Type(t) for t in types if t != "notype"]

def phys_bulk(m):
    return m["base_hp"] * m["base_def"]

def spec_bulk(m):
    return m["base_hp"] * m["base_spd"]

def move_eff(move, defender_name):
    try:
        return effectiveness(Type(move["type"]), *real_types(TIPI[defender_name]))
    except (KeyError, ValueError):
        return None

def has_quad_weakness(types, attacker_types=TYPES):
    return any(effectiveness(a, *types) >= 4 for a in attacker_types)

def best_stab(att_types, def_types):
    return max((effectiveness(t, *def_types) for t in att_types), default=1.0)

# ---------- Feature Statiche ----------
def static_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    lead_types = real_types(lead["types"])
    p1_lead_name = b["battle_timeline"][0]["p1_pokemon_state"]["name"]
    p1_lead = next((m for m in team if m["name"] == p1_lead_name), team[0])

    off = [max(effectiveness(t, *lead_types) for t in real_types(m["types"])) for m in team]
    dfn = [max(effectiveness(t, *real_types(m["types"])) for t in lead_types) for m in team]

    holders = [m for m in team if m.get("item") == "leftovers"]
    leftovers_bulk = sum(m["base_hp"] * (m["base_def"] + m["base_spd"]) for m in holders) / 1e4

    f = {
        "p2_lead_name": lead.get("name", ""),
        "phys_bulk_diff": np.mean([phys_bulk(m) for m in team]) - phys_bulk(lead),
        "spec_bulk_diff": np.mean([spec_bulk(m) for m in team]) - spec_bulk(lead),
        "n_quad_weak": sum(has_quad_weakness(real_types(m["types"])) for m in team),
        "team_off_vs_lead": np.mean(off),
        "lead_off_vs_team": np.mean(dfn),
        "matchup_diff": np.mean(off) - np.mean(dfn),
        "leftovers_bulk": leftovers_bulk
    }

    for i, m in enumerate(team):
        f[f"p1_{i}_name"] = m.get("name", "")

    for s in STATS:
        n = stat_name(s)
        vals = [m[s] for m in team]
        f[f"lead_{n}_diff"] = p1_lead[s] - lead[s]
        f[f"team_{n}_mean_diff"] = np.mean(vals) - lead[s]
        f[f"team_{n}_max_diff"] = max(vals) - lead[s]

    return f

# ---------- Feature Team P2 Rivelato ----------
def revealed_features(b):
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    seen = list(dict.fromkeys(
        [lead["name"]] + [t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]]
    ))
    n_seen = len(seen)
    n_unknown = max(TEAM_SIZE - n_seen, 0)
    total = n_seen + n_unknown

    p2 = [DB.get(n, GLOBAL) for n in seen]
    p2_types = [real_types(TIPI.get(n, [])) for n in seen]
    p1_types = [real_types(m["types"]) for m in team]

    f = {
        "p2_n_seen": n_seen
    }

    for s in STATS:
        n = stat_name(s)
        p2_est = (sum(m[s] for m in p2) + n_unknown * GLOBAL[s]) / total
        f[f"p2team_{n}_diff"] = np.mean([m[s] for m in team]) - p2_est

    p2_phys = (sum(phys_bulk(m) for m in p2) + n_unknown * phys_bulk(GLOBAL)) / total
    p2_spec = (sum(spec_bulk(m) for m in p2) + n_unknown * spec_bulk(GLOBAL)) / total
    f["p2team_phys_bulk_diff"] = np.mean([phys_bulk(m) for m in team]) - p2_phys
    f["p2team_spec_bulk_diff"] = np.mean([spec_bulk(m) for m in team]) - p2_spec

    p2_spe = np.array([m["base_spe"] for m in p2])
    f["p1_faster_than_seen_frac"] = np.mean(
        [[m["base_spe"] > sp for sp in p2_spe] for m in team]
    )

    off = np.mean([[best_stab(at, dt) for dt in p2_types] for at in p1_types])
    dfn = np.mean([[best_stab(at, dt) for dt in p1_types] for at in p2_types])
    f["seen_off"] = off
    f["seen_dfn"] = dfn
    f["seen_matchup_diff"] = off - dfn

    p2_quad_weak_frac = np.mean([has_quad_weakness(t) for t in p2_types])
    p1_quad_weak_frac = np.mean([has_quad_weakness(t) for t in p1_types])
    f["quad_weak_frac_diff"] = p1_quad_weak_frac - p2_quad_weak_frac

    return f

# ---------- Feature Timeline ----------
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

    spikes_active = False
    switches_under_spikes = 0
    
    for i in range(len(tl)):
        o_m = opp_moves[i]
        if o_m and o_m.get("name") == "spikes":
            spikes_active = True
        
        m_m = moves[i]
        if m_m and m_m.get("name") == "rapidspin":
            spikes_active = False
            
        if i > 0:
            prev_n = states[i-1]["name"]
            curr_n = states[i]["name"]
            prev_fnt = (states[i-1]["status"] == "fnt")
            if curr_n != prev_n and not prev_fnt:
                if spikes_active:
                    switches_under_spikes += 1

    switches = sum(
        1 for i in range(1, len(states))
        if states[i]["name"] != states[i - 1]["name"] and states[i - 1]["status"] != "fnt"
    )
    used = [m for m in moves if m is not None]
    dmg = [(m, o) for m, o in zip(moves, opp) if m is not None and m["category"] != "status"]
    effs = [e for m, o in dmg if (e := move_eff(m, o["name"])) is not None]

    return {
        "hp": hp,
        "hp_mean": hp.mean() if len(hp) > 0 else 0.0,
        "hp_min": hp.min() if len(hp) > 0 else 0.0,
        "hp_last": hp[-1] if len(hp) > 0 else 0.0,
        "team_hp": (sum(last_hp.values()) + (TEAM_SIZE - len(last_hp))) / TEAM_SIZE,
        "fainted": sum(v == 0 for v in last_hp.values()),
        "first_faint": first_faint,
        "switches": switches,
        "spikes_damage_est": switches_under_spikes * 0.125,
        "status_frac": np.mean([s["status"] not in ("nostatus", "fnt") for s in states]),
        "sleep_frac": np.mean([s["status"] == "slp" for s in states]),
        "boost_net": np.mean([sum(s["boosts"][k] for k in BOOSTS) for s in states]),
        "bp_mean": np.mean([m["base_power"] for m, _ in dmg]) if dmg else 0.0,
        "status_move_frac": (len(used) - len(dmg)) / max(len(used), 1),
        "move_eff": np.mean(effs) if effs else 1.0,
        "last_hp": last_hp
    }

def timeline_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    a, o = side_stats(tl, "p1", "p2"), side_stats(tl, "p2", "p1")
    d = a["hp"] - o["hp"]

    # 1. Tracciamento Explosion / Self-Destruct
    p1_booms = sum(1 for t in tl if t.get("p1_move_details") and t["p1_move_details"].get("name") in ("explosion", "selfdestruct"))
    p2_booms = sum(1 for t in tl if t.get("p2_move_details") and t["p2_move_details"].get("name") in ("explosion", "selfdestruct"))

    # 2. Tracciamento uso di Rest
    p1_rests = sum(1 for t in tl if t.get("p1_move_details") and t["p1_move_details"].get("name") == "rest")
    p2_rests = sum(1 for t in tl if t.get("p2_move_details") and t["p2_move_details"].get("name") == "rest")

    # 3. Verifica Spikes attive a fine timeline per P1 e P2
    p1_spikes_end = False
    p2_spikes_end = False
    for t in tl:
        m1, m2 = t.get("p1_move_details"), t.get("p2_move_details")
        if m2 and m2.get("name") == "spikes": p1_spikes_end = True
        if m1 and m1.get("name") == "rapidspin": p1_spikes_end = False
        if m1 and m1.get("name") == "spikes": p2_spikes_end = True
        if m2 and m2.get("name") == "rapidspin": p2_spikes_end = False

    # 4. Inerzia del Momentum HP (Ultimi 10 turni vs Primi 10 turni)
    trend_first10 = d[:10].mean() if len(d) >= 10 else d.mean()
    trend_last10 = d[-10:].mean() if len(d) >= 10 else d.mean()
    hp_momentum_trend = trend_last10 - trend_first10

    # Tracciamento Snorlax
    p1_team = b["p1_team_details"]
    p1_has_lax = any(m.get("name") == "snorlax" for m in p1_team)
    p1_lax_hp = a["last_hp"].get("snorlax", 1.0) if p1_has_lax else 0.0

    p2_lead = b["p2_lead_details"]
    seen = list(dict.fromkeys([p2_lead["name"]] + [t["p2_pokemon_state"]["name"] for t in tl]))
    p2_has_lax = "snorlax" in seen
    p2_lax_hp = o["last_hp"].get("snorlax", 1.0) if p2_has_lax else 0.0

    return {
        "p1_hp_mean": a["hp_mean"],
        "p2_hp_mean": o["hp_mean"],
        "p1_hp_min": a["hp_min"],
        "p2_hp_min": o["hp_min"],
        "p1_hp_last": a["hp_last"],
        "p2_hp_last": o["hp_last"],
        "hp_diff_last": a["hp_last"] - o["hp_last"],
        "team_hp_diff": a["team_hp"] - o["team_hp"],
        "fainted_diff": o["fainted"] - a["fainted"],
        "first_ko": (a["first_faint"] if a["first_faint"] is not None else n)
                    - (o["first_faint"] if o["first_faint"] is not None else n),
        "active_hp_mean_diff": d.mean(),
        "active_hp_last5_diff": d[-5:].mean(),
        "status_pressure": o["status_frac"] - a["status_frac"],
        "sleep_pressure": o["sleep_frac"] - a["sleep_frac"],
        "switches_diff": a["switches"] - o["switches"],
        "boost_diff": a["boost_net"] - o["boost_net"],
        "bp_diff": a["bp_mean"] - o["bp_mean"],
        "status_move_diff": a["status_move_frac"] - o["status_move_frac"],
        "move_eff_diff": a["move_eff"] - o["move_eff"],
        "p1_snorlax_hp_last": p1_lax_hp,
        "p2_snorlax_hp_seen": p2_lax_hp,
        "snorlax_hp_diff": p1_lax_hp - p2_lax_hp,
        "spikes_switch_pressure": o["spikes_damage_est"] - a["spikes_damage_est"],

        # NUOVE 4 MICRO-FEATURE TATTICHE 
        "boom_trade_diff": p1_booms - p2_booms,
        "rest_recovery_diff": p1_rests - p2_rests,
        "spikes_active_diff": int(p2_spikes_end) - int(p1_spikes_end),
        "hp_momentum_trend": hp_momentum_trend,
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
    train.to_csv("battles_features.csv", index=False)
    print(f"Train dataset estratto con successo: {train.shape}")