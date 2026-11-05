import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = os.environ.get("USE_EBM", "0") == "1"   # nel run precedente EBM non era attivo: off di default
except Exception:
    HAS_EBM = False

from features import build

# =========================================================
# CONFIG
# =========================================================
N_SPLITS = int(os.environ.get("N_SPLITS", 10))
N_REPEATS = int(os.environ.get("N_REPEATS", 3))     # CV ripetuta: stima piu' stabile
SEEDS = [42, 101]                                    # seed per i modelli GBM
CV_SEED = 42
MAX_ITERS = int(os.environ.get("MAX_ITERS", 2000))
CORR_THRESHOLD = 0.90                                # soglia correlazione per il pruning
MIN_KEEP = 20                                        # minimo di feature tenute per fold
IMP_Z = float(os.environ.get("IMP_Z", 1.0))          # tieni una feature se importanza media > IMP_Z * std
INNER_VAL = 0.15                                     # quota di train del fold usata per l'early stopping
GBM_NAMES = ["cb_sym", "cb_loss", "lgb"]
# Pesi FISSI per famiglia di modelli (rinormalizzati sui modelli disponibili)
# Nel run di riferimento logreg ~ GBM (acc 67.39 vs 67.05-67.31, AUC 0.7414 vs 0.7386-0.7418): peso uguale.
FAMILY_W = {"gbm": 0.5, "logreg": 0.5, "ebm": 0.2}
# Colonne da escludere a priori (es. gruppi risultati inutili da ablation.py)
# Mai tenute in nessun fold (30 fold, run di riferimento): rimuoverle accelera il run senza cambiare i fold.
# Nota: elenco ricavato dagli stessi dati di train, ma toglie solo feature che la selezione scartava comunque.
DROP_COLS = [
    "log_hp_ratio", "n_turns", "hp_adv_max", "fainted_diff", "hp_adv_trend", "p2team_spe_diff",
    "team_hp_mean_diff", "team_hp_diff", "p2team_spec_bulk_diff", "hp_adv_ewm",
    "log_hp_ratio_last10", "hp_adv_last5",
]
IMP_LOG = []                                         # importanza per fold, per il report finale

# =========================================================
# 1. DATI E FEATURE
# =========================================================
print("Estrazione feature da 'train.jsonl' e 'test.jsonl'...")
train_df = build("train.jsonl")
test_df = build("test.jsonl")

target_col = "target" if "target" in train_df.columns else "player_won"
X_train = train_df.drop(columns=["battle_id", target_col], errors="ignore").reset_index(drop=True)
y_train = train_df[target_col].astype(int).reset_index(drop=True)
X_test = test_df.reindex(columns=X_train.columns, fill_value=0)


def add_interactions(X):
    if {"board_control_diff", "hp_last_diff"} <= set(X.columns):
        X["momentum_multiplier"] = X["board_control_diff"] * (X["hp_last_diff"] + 0.1)
    if {"team_hp_diff", "board_control_diff"} <= set(X.columns):
        X["hp_x_board"] = X["team_hp_diff"] * X["board_control_diff"]
    if {"phys_bulk_diff", "static_net_pressure_diff"} <= set(X.columns):
        X["bulk_x_pressure"] = X["phys_bulk_diff"] * X["static_net_pressure_diff"]
    if {"hp_min_diff", "hp_last_diff"} <= set(X.columns):
        X["hp_volatility_spread"] = X["hp_min_diff"] - X["hp_last_diff"]
    return X


X_train = add_interactions(X_train)
X_test = add_interactions(X_test)

X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

for col in [c for c in X_train.columns if c.endswith("_name")]:
    freq = X_train[col].value_counts().to_dict()
    X_train[col] = X_train[col].map(freq).fillna(0).astype(float)
    X_test[col] = X_test[col].map(freq).fillna(0).astype(float)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
print(f"Dataset pronto: {X_train.shape[0]} battaglie | {X_train.shape[1]} feature iniziali.")


# =========================================================
# 2. SELEZIONE FEATURE (dentro ogni fold, solo sul train del fold -> niente leak)
# =========================================================
def corr_prune(X, thr=CORR_THRESHOLD):
    """Rimuove una feature per ogni coppia con |corr| > thr (tiene quella meno ridondante)."""
    c = X.corr().abs()
    mean_c = c.mean()
    cols, drop = list(X.columns), set()
    for i in range(len(cols)):
        if cols[i] in drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in drop:
                continue
            if c.iloc[i, j] > thr:
                drop.add(cols[i] if mean_c[cols[i]] >= mean_c[cols[j]] else cols[j])
                if cols[i] in drop:
                    break
    return [c for c in cols if c not in drop]


def select_features(X_tr, y_tr):
    cols = [c for c in X_tr.columns if X_tr[c].nunique(dropna=True) > 1]
    cols = corr_prune(X_tr[cols])
    Xa, Xb, ya, yb = train_test_split(X_tr[cols], y_tr, test_size=0.25, stratify=y_tr, random_state=0)
    m = LGBMClassifier(
        n_estimators=250, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
        random_state=0, n_jobs=-1, verbose=-1,
    ).fit(Xa, ya)
    pi = permutation_importance(m, Xb, yb, scoring="neg_log_loss", n_repeats=10, random_state=0, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=cols)
    std = pd.Series(pi.importances_std, index=cols)
    # Soglia piu' severa di "> 0": la media deve superare IMP_Z deviazioni standard degli shuffle
    keep = imp[(imp > 0) & (imp > IMP_Z * std)].index.tolist()
    if len(keep) < MIN_KEEP:
        keep = imp.sort_values(ascending=False).index[:MIN_KEEP].tolist()
    IMP_LOG.append((imp, set(keep)))
    return keep


# =========================================================
# 3. MODELLI (alberi poco profondi e molto regolarizzati)
# =========================================================
def make_model(name, seed, iters):
    if name == "cb_sym":
        return CatBoostClassifier(
            iterations=iters, learning_rate=0.03, depth=4, grow_policy="SymmetricTree",
            l2_leaf_reg=10.0, random_strength=1.5, bagging_temperature=0.8,
            random_seed=seed, verbose=0, thread_count=-1)
    if name == "cb_loss":
        return CatBoostClassifier(
            iterations=iters, learning_rate=0.03, depth=4, grow_policy="Lossguide",
            max_leaves=15, l2_leaf_reg=10.0, random_strength=1.5, bagging_temperature=0.8,
            random_seed=seed, verbose=0, thread_count=-1)
    if name == "lgb":
        return LGBMClassifier(
            n_estimators=iters, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=50, reg_alpha=1.0, reg_lambda=10.0,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(name)


def fit_logreg(X, y):
    """Logistic regression L2 (lbfgs, veloce) con C scelto da una CV interna al train del fold."""
    lr = LogisticRegressionCV(Cs=np.logspace(-3, 0, 8), cv=5, scoring="neg_log_loss",
                              max_iter=2000, n_jobs=-1)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr).fit(X, y)


def make_ebm():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExplainableBoostingClassifier(interactions=5, learning_rate=0.02, outer_bags=4,
                                      max_bins=128, random_state=0, n_jobs=-1))


EXTRA_NAMES = ["logreg"] + (["ebm"] if HAS_EBM else [])
if not HAS_EBM:
    print("[info] EBM disattivato (USE_EBM=1 per attivarlo, richiede il pacchetto 'interpret').")


def blend(p, w=None):
    """Blend a pesi fissi tra famiglie: media dei GBM, logreg, (EBM)."""
    w = w or FAMILY_W
    fam = {"gbm": np.mean([p[k] for k in GBM_NAMES], axis=0)}
    fam.update({k: p[k] for k in EXTRA_NAMES})
    tot = sum(w[k] for k in fam)
    return sum(w[k] * v for k, v in fam.items()) / tot


rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_SEED)
folds = list(rskf.split(X_train, y_train))

# ---------- Fase A: stima n_iter (primo giro di CV, un seed) ----------
# L'early stopping usa uno split INTERNO al train del fold: il fold di validazione esterno
# non viene mai toccato, quindi la Fase B resta una stima onesta.
print("\n--- FASE A: stima iterazioni (early stopping su split interno al train del fold) ---")
selected = {}
best_iters = {k: [] for k in GBM_NAMES}
for fi, (tr, va) in enumerate(folds[:N_SPLITS]):
    X_tr, y_tr = X_train.iloc[tr], y_train.iloc[tr]
    cols = select_features(X_tr, y_tr)
    selected[fi] = cols
    Xi, Xv, yi, yv = train_test_split(X_tr[cols], y_tr, test_size=INNER_VAL,
                                      stratify=y_tr, random_state=fi)
    for name in GBM_NAMES:
        m = make_model(name, CV_SEED, MAX_ITERS)
        if name == "lgb":
            m.fit(Xi, yi, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                  callbacks=[early_stopping(100, verbose=False)])
            best_iters[name].append(m.best_iteration_ or MAX_ITERS)
        else:
            m.fit(Xi, yi, eval_set=(Xv, yv), early_stopping_rounds=100, verbose=False)
            best_iters[name].append(m.get_best_iteration() + 1)
    print(f"  fold {fi + 1}: {len(cols)} feature tenute")

# +~1/(1-INNER_VAL): il modello finale vede piu' dati dello split interno
N_ITERS = {k: max(150, int(np.median(v) / (1 - INNER_VAL))) for k, v in best_iters.items()}
print("Iterazioni scelte (mediana sui fold):", N_ITERS)

# ---------- Fase B: CV ripetuta a iterazioni fisse ----------
print(f"\n--- FASE B: {N_SPLITS}-fold CV x {N_REPEATS} ripetizioni, iterazioni fisse ---")
MODEL_NAMES = GBM_NAMES + EXTRA_NAMES
print("Modelli nel blend:", MODEL_NAMES)
oof = {k: np.zeros((N_REPEATS, len(X_train))) for k in MODEL_NAMES}
test_pred = {k: np.zeros(len(X_test)) for k in MODEL_NAMES}
gaps, n_kept = [], []

for fi, (tr, va) in enumerate(folds):
    rep = fi // N_SPLITS
    X_tr, y_tr = X_train.iloc[tr], y_train.iloc[tr]
    X_va, y_va = X_train.iloc[va], y_train.iloc[va]
    cols = selected.get(fi) or select_features(X_tr, y_tr)
    n_kept.append(len(cols))
    Xtr_c, Xva_c, Xte_c = X_tr[cols], X_va[cols], X_test[cols]

    tr_p = {k: np.zeros(len(tr)) for k in MODEL_NAMES}
    va_p = {k: np.zeros(len(va)) for k in MODEL_NAMES}
    for name in GBM_NAMES:
        for seed in SEEDS:
            m = make_model(name, seed, N_ITERS[name]).fit(Xtr_c, y_tr)
            tr_p[name] += m.predict_proba(Xtr_c)[:, 1] / len(SEEDS)
            va_p[name] += m.predict_proba(Xva_c)[:, 1] / len(SEEDS)
            test_pred[name] += m.predict_proba(Xte_c)[:, 1] / (len(folds) * len(SEEDS))
    lr = fit_logreg(Xtr_c, y_tr)
    tr_p["logreg"] = lr.predict_proba(Xtr_c)[:, 1]
    va_p["logreg"] = lr.predict_proba(Xva_c)[:, 1]
    test_pred["logreg"] += lr.predict_proba(Xte_c)[:, 1] / len(folds)

    if HAS_EBM:
        ebm = make_ebm().fit(Xtr_c, y_tr)
        tr_p["ebm"] = ebm.predict_proba(Xtr_c)[:, 1]
        va_p["ebm"] = ebm.predict_proba(Xva_c)[:, 1]
        test_pred["ebm"] += ebm.predict_proba(Xte_c)[:, 1] / len(folds)

    for k in MODEL_NAMES:
        oof[k][rep, va] = va_p[k]

    tr_acc = accuracy_score(y_tr, blend(tr_p) >= 0.5) * 100
    va_acc = accuracy_score(y_va, blend(va_p) >= 0.5) * 100
    gaps.append(tr_acc - va_acc)
    print(f"Rep {rep + 1} Fold {fi % N_SPLITS + 1:2d}: feat={len(cols):2d} | Train(in-sample) = {tr_acc:.2f}% | "
          f"Val = {va_acc:.2f}% | Gap = {tr_acc - va_acc:.2f}%")


# =========================================================
# 4. RISULTATI (media +/- std sulle ripetizioni)
# =========================================================
def summarize(oof_mat):
    accs = [accuracy_score(y_train, oof_mat[r] >= 0.5) * 100 for r in range(N_REPEATS)]
    aucs = [roc_auc_score(y_train, oof_mat[r]) for r in range(N_REPEATS)]
    lls = [log_loss(y_train, np.clip(oof_mat[r], 1e-6, 1 - 1e-6)) for r in range(N_REPEATS)]
    return accs, aucs, lls


print("\n--- OOF per modello (media +/- std sulle ripetizioni) ---")
for k in MODEL_NAMES:
    a, u, l = summarize(oof[k])
    print(f"{k:8s} acc = {np.mean(a):.2f}% +/- {np.std(a):.2f} | AUC = {np.mean(u):.4f} | logloss = {np.mean(l):.4f}")

oof_blend = blend(oof)
a, u, l = summarize(oof_blend)
se = np.sqrt(np.mean(a) / 100 * (1 - np.mean(a) / 100) / len(y_train)) * 100

print("\n" + "=" * 65)
print(" RISULTATI FINALI (blend a pesi fissi per famiglia di modelli)")
print("=" * 65)
print(f"Accuratezza OOF:      {np.mean(a):.2f}% (std tra ripetizioni {np.std(a):.2f}%, errore standard ~{se:.2f}%)")
print(f"ROC-AUC OOF:          {np.mean(u):.4f} (std {np.std(u):.4f})")
print(f"Log-loss OOF:         {np.mean(l):.4f} (std {np.std(l):.4f})")
print(f"Gap medio Train-Val:  {np.mean(gaps):.2f}%  (train in-sample, da leggere in relativo)")
print(f"Feature medie tenute: {np.mean(n_kept):.1f} su {X_train.shape[1]}")
print("=" * 65)


# =========================================================
# 4a. DIAGNOSTICA: il blend aggiunge qualcosa? (differenze appaiate sulle stesse ripetizioni)
# =========================================================
print("\n--- VARIANTI DI BLEND (stesse predizioni OOF) ---")
variants = {
    "blend (pesi config)": None,
    "solo GBM": {"gbm": 1.0, "logreg": 0.0, "ebm": 0.0},
    "solo logreg": {"gbm": 0.0, "logreg": 1.0, "ebm": 0.0},
}
for vname, w in variants.items():
    va_, vu_, vl_ = summarize(blend(oof, w))
    print(f"{vname:22s} acc = {np.mean(va_):.2f}% | AUC = {np.mean(vu_):.4f} | logloss = {np.mean(vl_):.4f}")

best = min(MODEL_NAMES, key=lambda k: np.mean(summarize(oof[k])[2]))
ba, bu, bl = summarize(oof[best])
d_acc, d_auc, d_ll = np.array(a) - np.array(ba), np.array(u) - np.array(bu), np.array(l) - np.array(bl)
print(f"\nBlend vs miglior singolo per logloss ({best}), per ripetizione:")
print(f"  d_acc = {np.round(d_acc, 2)} | d_AUC = {np.round(d_auc, 4)} | d_logloss = {np.round(d_ll, 4)}")
print(f"  (d_logloss < 0 in tutte le ripetizioni = il blend aiuta davvero; errore standard acc ~{se:.2f}%)")

print("\nCorrelazione tra le predizioni OOF dei modelli (bassa = piu' diversita' da sfruttare):")
corr = pd.DataFrame(np.corrcoef([oof[k].mean(axis=0) for k in MODEL_NAMES]), index=MODEL_NAMES, columns=MODEL_NAMES)
print(corr.round(3).to_string())

# =========================================================
# 4b. REPORT IMPORTANZA FEATURE (permutation importance media sui fold)
# =========================================================
all_cols = list(X_train.columns)
imp_df = pd.DataFrame({
    "importance_mean": pd.concat([i for i, _ in IMP_LOG], axis=1).reindex(all_cols).fillna(0).mean(axis=1),
    "kept_frac": [np.mean([c in kept for _, kept in IMP_LOG]) for c in all_cols],
}, index=all_cols).sort_values("importance_mean", ascending=False)
imp_df.to_csv("feature_importance.csv")
print("\n--- TOP 25 FEATURE (importanza = calo medio di -logloss se permutata) ---")
print(imp_df.head(25).round(5).to_string())
never = imp_df[imp_df["kept_frac"] == 0].index.tolist()
print(f"\nFeature mai tenute in nessun fold ({len(never)}): {never}")
print("Report completo salvato in 'feature_importance.csv'")

# =========================================================
# 5. SUBMISSION
# =========================================================
test_blend = blend(test_pred)
submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": (test_blend >= 0.5).astype(int),
})
submission.to_csv("submission.csv", index=False)
print(f"\n[OK] 'submission.csv' generato. Predetti positivi: {submission['player_won'].mean() * 100:.1f}%")