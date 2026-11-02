"""
ablation.py - Selezione feature: permutation importance + ablation a gruppi in CV.

Uso:  python ablation.py
Richiede: features.py, type_chart.py, pokemon_db.json, train.jsonl nella stessa cartella.

Output:
  - permutation_importance.csv   (importanza media su fold di validazione)
  - ablation_results.csv         (confronto baseline vs. rimozione di ogni gruppo)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance

from features import build

SEED = 42
N_SPLITS, N_REPEATS = 5, 3          # 15 fit per configurazione, stessi split per tutti
PERM_REPEATS = 5                    # shuffle per feature in permutation importance
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, random_state=SEED,
                  verbose=-1, n_jobs=-1)

# ---------------------------------------------------------
# Gruppi di rimozione (ridondanti o deboli secondo i tuoi report).
# Le colonne inesistenti vengono ignorate in automatico.
# ---------------------------------------------------------
GROUPS = {
    "quad_weak (tieni quad_weak_frac_diff)": [
        "n_quad_weak", "n_quad_weak_to_lead", "p1_quad_weak_frac", "p2_quad_weak_frac",
    ],
    "speed_count (tieni n_faster_than_lead)": [
        "n_slower_than_lead",
    ],
    "leftovers": [
        "n_leftovers", "leftovers_bulk",
    ],
    "F-score bottom": [
        "n_quad_weak_to_lead", "team_def_mean_diff", "leftovers_bulk", "phys_bulk_diff",
        "lead_hp_diff", "p2team_spe_diff", "p2team_spa_diff", "team_spd_mean_diff",
        "bp_diff", "lead_spe_diff",
    ],
    "tree bottom": [
        "n_quad_weak_to_lead", "n_leftovers", "n_super_effective_answers",
        "n_slower_than_lead", "n_quad_weak", "n_faster_than_lead",
        "p1_quad_weak_frac", "p2_quad_weak_frac",
    ],
    "matchup_ridondanti": [
        "team_off_vs_lead", "lead_off_vs_team",   # riassunte da matchup_diff
    ],
}
# gruppo "tutti i candidati" = unione di tutti
GROUPS["TUTTI i candidati"] = sorted({c for cols in GROUPS.values() for c in cols})


def load_data():
    print("Estrazione feature da 'train.jsonl'...")
    df = build("train.jsonl")
    target = "target" if "target" in df.columns else "player_won"
    X = df.drop(columns=["battle_id", target], errors="ignore")
    y = df[target].astype(int)
    print(f"{X.shape[0]} battaglie, {X.shape[1]} feature.")
    return X, y


def make_model():
    return lgb.LGBMClassifier(**LGB_PARAMS)


# ---------------------------------------------------------
# 1. Permutation importance su fold di validazione
# ---------------------------------------------------------
def run_permutation(X, y):
    print("\n=== PERMUTATION IMPORTANCE (accuratezza, fold di validazione) ===")
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)
    imps = []
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(model, X.iloc[va], y.iloc[va],
                                   scoring="accuracy", n_repeats=PERM_REPEATS,
                                   random_state=SEED, n_jobs=1)
        imps.append(r.importances_mean)
        print(f"  fold {fold}/{N_SPLITS} fatto")
    imps = np.array(imps)
    out = pd.DataFrame({
        "feature": X.columns,
        "perm_mean": imps.mean(axis=0),
        "perm_std_folds": imps.std(axis=0),
    }).sort_values("perm_mean")
    out.to_csv("permutation_importance.csv", index=False)
    print("\nFeature con importanza più bassa (<= 0 = inutile o dannosa):")
    print(out.head(20).to_string(index=False))
    return out


# ---------------------------------------------------------
# 2. Ablation a gruppi (stessi split per tutti -> confronto appaiato)
# ---------------------------------------------------------
def cv_scores(X, y, splits):
    acc, ll = [], []
    for tr, va in splits:
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        acc.append(accuracy_score(y.iloc[va], proba >= 0.5))
        ll.append(log_loss(y.iloc[va], proba))
    return np.array(acc), np.array(ll)


def run_ablation(X, y):
    print("\n=== ABLATION A GRUPPI ===")
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(cv.split(X, y))

    base_acc, base_ll = cv_scores(X, y, splits)
    print(f"BASELINE ({X.shape[1]} feature): acc={base_acc.mean()*100:.2f}% "
          f"(± {base_acc.std()*100:.2f}) | logloss={base_ll.mean():.4f}")

    rows = []
    for name, cols in GROUPS.items():
        present = [c for c in cols if c in X.columns]
        if not present:
            continue
        acc, ll = cv_scores(X.drop(columns=present), y, splits)
        diff = acc - base_acc                       # >0: rimuovere migliora
        se = diff.std(ddof=1) / np.sqrt(len(diff))  # errore standard della differenza appaiata
        verdict = ("RIMUOVI" if diff.mean() > -se and ll.mean() <= base_ll.mean() + 0.001
                   else "TIENI")
        rows.append({
            "gruppo": name, "n_rimosse": len(present),
            "acc": acc.mean(), "diff_acc": diff.mean(), "se_diff": se,
            "logloss": ll.mean(), "diff_logloss": ll.mean() - base_ll.mean(),
            "verdetto": verdict,
        })
        print(f"- {name:<42} -{len(present):>2} feat | acc {acc.mean()*100:.2f}% "
              f"(Δ {diff.mean()*100:+.2f} ± {se*100:.2f}) | "
              f"logloss {ll.mean():.4f} (Δ {ll.mean()-base_ll.mean():+.4f}) -> {verdict}")

    res = pd.DataFrame(rows)
    res.to_csv("ablation_results.csv", index=False)
    print("\nRegola: RIMUOVI se Δacc non è peggiore di -1 errore standard "
          "e la logloss non peggiora oltre 0.001.")
    return res


if __name__ == "__main__":
    X, y = load_data()
    run_permutation(X, y)
    run_ablation(X, y)
