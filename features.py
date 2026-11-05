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

# ---------- Feature Statiche ----------
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

    p1_atk_mean = np.mean([m["base_atk"] for m in team])
    p1_def_mean = np.mean([m["base_def"] for m in team])
    p1_spa_mean = np.mean([m["base_spa"] for m in team])
    p1_spd_mean = np.mean([m["base_spd"] for m in team])

    p1_phys_press = norm_press(p1_atk_mean, p2_est["base_def"])
    p1_spec_press = norm_press(p1_spa_mean, p2_est["base_spd"])
    p1_rev_off_press = (p1_phys_press + p1_spec_press) / 2

    p2_phys_press = norm_press(p2_est["base_atk"], p1_def_mean)
    p2_spec_press = norm_press(p2_est["base_spa"], p1_spd_mean)
    p2_rev_off_press = (p2_phys_press + p2_spec_press) / 2

    f["revealed_net_pressure_diff"] = (p1_rev_off_press - p2_rev_off_press) / 2

    p2_phys = (sum(phys_bulk(m) for m in p2) + n_unknown * phys_bulk(GLOBAL)) / total
    p2_spec = (sum(spec_bulk(m) for m in p2) + n_unknown * spec_bulk(GLOBAL)) / total
    f["p2team_phys_bulk_diff"] = np.mean([phys_bulk(m) for m in team]) - p2_phys
    f["p2team_spec_bulk_diff"] = np.mean([spec_bulk(m) for m in team]) - p2_spec

    p2_spe = np.array([m["base_spe"] for m in p2])
    f["p1_faster_than_seen_frac"] = np.mean(
        [[m["base_spe"] > sp for sp in p2_spe] for m in team]
    )

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
        "move_eff": np.mean(effs) if effs else 1.0,
        "last_hp": last_hp
    }

def timeline_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    a, o = side_stats(tl, "p1", "p2"), side_stats(tl, "p2", "p1")

    p1_booms = sum(1 for t in tl if t.get("p1_move_details") and t["p1_move_details"].get("name") in ("explosion", "selfdestruct"))
    p2_booms = sum(1 for t in tl if t.get("p2_move_details") and t["p2_move_details"].get("name") in ("explosion", "selfdestruct"))

    p1_rests = sum(1 for t in tl if t.get("p1_move_details") and t["p1_move_details"].get("name") == "rest")
    p2_rests = sum(1 for t in tl if t.get("p2_move_details") and t["p2_move_details"].get("name") == "rest")

    p1_spikes_end = False
    p2_spikes_end = False
    
    p1_speed_advantages = 0
    p1_super_effective_given = 0
    p1_moves_thrown = 0
    
    p1_super_effective_taken = 0
    p2_moves_thrown = 0

    for t in tl:
        m1, m2 = t.get("p1_move_details"), t.get("p2_move_details")
        
        if m2 and m2.get("name") == "spikes": p1_spikes_end = True
        if m1 and m1.get("name") == "rapidspin": p1_spikes_end = False
        if m1 and m1.get("name") == "spikes": p2_spikes_end = True
        if m2 and m2.get("name") == "rapidspin": p2_spikes_end = False

        p1_state = t.get("p1_pokemon_state")
        p2_state = t.get("p2_pokemon_state")
        if p1_state and p2_state:
            p1_name = p1_state.get("name", "").strip().lower()
            p2_name = p2_state.get("name", "").strip().lower()
            
            p1_mon_data = DB.get(p1_name, GLOBAL)
            p2_mon_data = DB.get(p2_name, GLOBAL)
            
            base_spe_p1 = p1_mon_data.get("base_spe", 100)
            base_spe_p2 = p2_mon_data.get("base_spe", 100)
            
            boost_stages = {-6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3, 
                             0: 2/2, 1: 3/2, 2: 4/2, 3: 5/2, 4: 6/2, 5: 7/2, 6: 8/2}
            
            p1_boost_spe = p1_state.get("boosts", {}).get("spe", 0)
            p2_boost_spe = p2_state.get("boosts", {}).get("spe", 0)
            
            eff_spe_p1 = base_spe_p1 * boost_stages.get(p1_boost_spe, 1.0)
            eff_spe_p2 = base_spe_p2 * boost_stages.get(p2_boost_spe, 1.0)
            
            if eff_spe_p1 > eff_spe_p2:
                p1_speed_advantages += 1

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

    board_control_diff = (
        (o["status_frac"] - a["status_frac"]) + 
        (o["sleep_frac"] - a["sleep_frac"]) + 
        (int(p2_spikes_end) - int(p1_spikes_end)) * 0.1 +
        (a["switches"] - o["switches"]) * 0.02
    )

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

    p1_cov_ratio = (p1_super_effective_given / p1_moves_thrown) if p1_moves_thrown > 0 else 0.0
    p2_cov_ratio = (p1_super_effective_taken / p2_moves_thrown) if p2_moves_thrown > 0 else 0.0

    return {
        "hp_min_diff": a["hp_min"] - o["hp_min"],
        "hp_last_diff": a["hp_last"] - o["hp_last"],
        "team_hp_diff": a["team_hp"] - o["team_hp"],
        "fainted_diff": o["fainted"] - a["fainted"],
        "first_ko": (a["first_faint"] if a["first_faint"] is not None else n)
                    - (o["first_faint"] if o["first_faint"] is not None else n),
        "tank_hp_diff": p1_tank_hp - p2_tank_hp,
        "board_control_diff": board_control_diff,
        "boost_diff": a["boost_net"] - o["boost_net"],
        "bp_diff": a["bp_mean"] - o["bp_mean"],
        "move_eff_diff": a["move_eff"] - o["move_eff"],
        "boom_trade_diff": p1_booms - p2_booms,
        "rest_recovery_diff": p1_rests - p2_rests,
        "speed_advantage_ratio": p1_speed_advantages / n if n > 0 else 0.0,
        "type_coverage_score_diff": p1_cov_ratio - p2_cov_ratio,
    }

def team_hp_series(tl, side):
    last, out = {}, []
    for t in tl:
        s = t[f"{side}_pokemon_state"]
        last[s["name"]] = 0.0 if s["status"] == "fnt" else s["hp_pct"]
        out.append((sum(last.values()) + (TEAM_SIZE - len(last))) / TEAM_SIZE)
    return np.array(out, dtype=float)

def status_inflicted(tl, side):
    """Quante volte uno status nuovo è comparso sul Pokémon attivo di `side`, escludendo il turno successivo a un Rest."""
    other = "p2" if side == "p1" else "p1"
    total = slp = par = 0
    for i in range(1, len(tl)):
        prev, cur = tl[i-1], tl[i]
        
        # FIX: Se nel turno precedente il lato ha usato Rest, saltiamo la transizione
        prev_move = prev.get(f"{side}_move_details")
        if prev_move and prev_move.get("name") == "rest":
            continue
            
        ps, cs = prev[f"{side}_pokemon_state"], cur[f"{side}_pokemon_state"]
        if ps["name"] == cs["name"] and ps["status"] == "nostatus" and cs["status"] not in ("nostatus", "fnt"):
            total += 1
            slp += cs["status"] == "slp"
            par += cs["status"] == "par"
    return total, slp, par

def boost_stats(tl, side):
    nets = np.array([sum(t[f"{side}_pokemon_state"]["boosts"][k] for k in BOOSTS) for t in tl], dtype=float)
    return nets.max() if len(nets) else 0.0, float(np.mean(nets > 0)) if len(nets) else 0.0

def move_usage(tl, side):
    moves = [t.get(f"{side}_move_details") for t in tl]
    used = [m for m in moves if m]
    n = max(len(tl), 1)
    status_n = sum(m.get("category") == "status" for m in used)
    hazards = sum(m.get("name") in ("spikes",) for m in used)
    
    # Nuove categorie di mosse richieste
    recovery_moves = {"recover", "softboiled", "moonlight", "rest"}
    setup_moves = {"swordsdance", "curse", "amnesia", "bellydrum", "growl", "screech"}
    phazing_moves = {"roar", "whirlwind", "haze"}
    screen_moves = {"reflect", "lightscreen"}
    inflict_status_moves = {"toxic", "thunderwave", "hypnosis", "sing", "spore", "stunspore", "willowisp"}
    
    rec_count = sum(m.get("name") in recovery_moves for m in used)
    setup_count = sum(m.get("name") in setup_moves for m in used)
    phaze_count = sum(m.get("name") in phazing_moves for m in used)
    screen_count = sum(m.get("name") in screen_moves for m in used)
    inflict_count = sum(m.get("name") in inflict_status_moves for m in used)
    
    return {
        "status_frac": status_n / n,
        "hazards": hazards,
        "move_rate": len(used) / n,
        "recovery": rec_count,
        "setup": setup_count,
        "phaze": phaze_count,
        "screen": screen_count,
        "inflict_status": inflict_count
    }

def extra_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    team, lead = b["p1_team_details"], b["p2_lead_details"]
    f = {"n_turns": n}

    h1, h2 = team_hp_series(tl, "p1"), team_hp_series(tl, "p2")
    d = h1 - h2
    f["team_hp_ratio"] = (h1[-1] + 0.05) / (h2[-1] + 0.05)

    s1, s2 = status_inflicted(tl, "p1"), status_inflicted(tl, "p2")
    f["status_inflicted_diff"] = s2[0] - s1[0]
    f["sleep_inflicted_diff"] = s2[1] - s1[1]
    f["para_inflicted_diff"] = s2[2] - s1[2]
    
    b1, b2 = boost_stats(tl, "p1"), boost_stats(tl, "p2")
    f["boost_max_diff"] = b1[0] - b2[0]
    f["boost_turns_diff"] = b1[1] - b2[1]
    
    u1, u2 = move_usage(tl, "p1"), move_usage(tl, "p2")
    f["status_move_frac_diff"] = u1["status_frac"] - u2["status_frac"]
    f["hazard_moves_diff"] = u1["hazards"] - u2["hazards"]
    f["move_rate_diff"] = u1["move_rate"] - u2["move_rate"]
    f["recovery_moves_diff"] = u1["recovery"] - u2["recovery"]
    f["setup_moves_diff"] = u1["setup"] - u2["setup"]
    f["phaze_moves_diff"] = u1["phaze"] - u2["phaze"]
    f["screen_moves_diff"] = u1["screen"] - u2["screen"]
    f["inflict_status_moves_diff"] = u1["inflict_status"] - u2["inflict_status"]

    seen = list(dict.fromkeys([lead["name"]] + [t["p2_pokemon_state"]["name"] for t in tl]))
    n_seen = len(seen)
    n_unk = max(TEAM_SIZE - n_seen, 0)
    p2 = [DB.get(nm, GLOBAL) for nm in seen]
    bst = lambda m: sum(m[s] for s in STATS)
    p2_bst = (sum(bst(m) for m in p2) + n_unk * bst(GLOBAL)) / (n_seen + n_unk)
    f["bst_mean_diff"] = np.mean([bst(m) for m in team]) - p2_bst
    f["fast_frac_diff"] = np.mean([m["base_spe"] >= 100 for m in team]) - np.mean([m["base_spe"] >= 100 for m in p2])
    f["max_spe_diff"] = max(m["base_spe"] for m in team) - max(m["base_spe"] for m in p2)
    f["max_atk_diff"] = max(max(m["base_atk"], m["base_spa"]) for m in team) - max(max(m["base_atk"], m["base_spa"]) for m in p2)

    try:
        p1_t = [real_types(m["types"]) for m in team]
        p2_t = [real_types(TYPES_BY_NAME.get(nm.strip().lower(), TIPI.get(nm, []))) for nm in seen]
        def matchup(att_side, def_side):
            vals = []
            for ts in att_side:
                for dt in def_side:
                    if ts and dt:
                        vals.append(max(effectiveness(t, *dt) for t in ts))
            return float(np.mean(vals)) if vals else 1.0
        m1 = matchup(p1_t, p2_t)
        m2 = matchup(p2_t, p1_t)
        f["type_matchup_diff"] = m1 - m2
    except Exception:
        f["type_matchup_diff"] = 0.0

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

def _types_of(name):
    return real_types(TYPES_BY_NAME.get((name or "").strip().lower(), TIPI.get(name, [])))

def pair_matchup(n1, n2):
    r1, r2 = _types_of(n1), _types_of(n2)
    if not r1 or not r2:
        return None
    return max(effectiveness(t, *r2) for t in r1) - max(effectiveness(t, *r1) for t in r2)

def new_features(b):
    tl = b["battle_timeline"]
    n = len(tl)
    f = {}

    t1, r1, h1 = dmg_stats(tl, "p1")
    t2, r2, h2 = dmg_stats(tl, "p2")
    f["dmg_dealt_diff"] = t2 - t1
    f["dmg_dealt_recent_diff"] = r2 - r1
    f["dmg_ratio"] = (t2 + 0.1) / (t1 + 0.1)
    f["heal_diff"] = h1 - h2

    hp1, hp2 = team_hp_series(tl, "p1"), team_hp_series(tl, "p2")
    w = 0.9 ** np.arange(n - 1, -1, -1)
    f["hp_adv_ewm"] = float(np.dot(w, hp1 - hp2) / w.sum())

    try:
        vals = [v for v in (pair_matchup(t["p1_pokemon_state"]["name"], t["p2_pokemon_state"]["name"]) for t in tl) if v is not None]
        f["active_matchup_mean"] = float(np.mean(vals)) if vals else 0.0
        f["active_matchup_last5"] = float(np.mean(vals[-5:])) if vals else 0.0
        lead_v = pair_matchup(tl[0]["p1_pokemon_state"]["name"], b["p2_lead_details"]["name"])
        f["lead_matchup"] = lead_v if lead_v is not None else 0.0
    except Exception:
        f["active_matchup_mean"] = f["active_matchup_last5"] = f["lead_matchup"] = 0.0

    s1, s2 = tl[-1]["p1_pokemon_state"], tl[-1]["p2_pokemon_state"]
    bad = lambda s: int(s["status"] not in ("nostatus", "fnt"))
    f["last_status_diff"] = bad(s2) - bad(s1)
    f["last_active_fnt_diff"] = int(s2["status"] == "fnt") - int(s1["status"] == "fnt")
    f["last_boost_diff"] = sum(s1["boosts"][k] for k in BOOSTS) - sum(s2["boosts"][k] for k in BOOSTS)
    d1, d2 = DB.get(s1["name"], GLOBAL), DB.get(s2["name"], GLOBAL)
    f["last_active_bst_diff"] = sum(d1[k] for k in STATS) - sum(d2[k] for k in STATS)
    f["last_active_spe_diff"] = d1["base_spe"] - d2["base_spe"]

    def alive_hp(side):
        last = {}
        for t in tl:
            s = t[f"{side}_pokemon_state"]
            last[s["name"]] = 0.0 if s["status"] == "fnt" else s["hp_pct"]
        alive = [v for v in last.values() if v > 0]
        n_unseen = TEAM_SIZE - len(last)
        return (sum(alive) + n_unseen) / max(len(alive) + n_unseen, 1)
    f["alive_hp_mean_diff"] = alive_hp("p1") - alive_hp("p2")

    def recent_switches(side):
        c = 0
        for i in range(max(1, n - 10), n):
            a, c2 = tl[i - 1][f"{side}_pokemon_state"], tl[i][f"{side}_pokemon_state"]
            if a["name"] != c2["name"] and a["status"] != "fnt":
                c += 1
        return c
    f["recent_switch_diff"] = recent_switches("p1") - recent_switches("p2")

    return {k: (0.0 if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in f.items()}

def last_hp_map(tl, side):
    last = {}
    for t in tl:
        st = t[f"{side}_pokemon_state"]
        last[st["name"]] = 0.0 if st["status"] == "fnt" else st["hp_pct"]
    return last

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

    seen2 = list(l2.keys())
    n_unk = max(TEAM_SIZE - len(seen2), 0)
    power = lambda m: max(m["base_atk"], m["base_spa"])
    p1_bulk = sum(l1.get(m["name"], 1.0) * mixed_bulk(m) for m in team)
    p2_bulk = sum(l2[nm] * mixed_bulk(DB.get(nm, GLOBAL)) for nm in seen2) + n_unk * mixed_bulk(GLOBAL)
    p1_pow = sum(l1.get(m["name"], 1.0) * power(m) for m in team)
    p2_pow = sum(l2[nm] * power(DB.get(nm, GLOBAL)) for nm in seen2) + n_unk * power(GLOBAL)
    f["bulk_hp_log_ratio"] = float(np.log((p1_bulk + 1) / (p2_bulk + 1)))
    f["power_hp_log_ratio"] = float(np.log((p1_pow + 1) / (p2_pow + 1)))

    t1, r1, _ = dmg_stats(tl, "p1")
    t2, r2, _ = dmg_stats(tl, "p2")
    f["dmg_log_ratio"] = float(np.log((t2 + 0.1) / (t1 + 0.1)))
    f["dmg_recent_log_ratio"] = float(np.log((r2 + 0.05) / (r1 + 0.05)))

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
    print(f"Train dataset estratto con successo: {train.shape}")