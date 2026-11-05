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
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = os.environ.get("USE_EBM", "0") == "1"
except Exception:
    HAS_EBM = False

from features import build

# =========================================================
# CONFIG
# =========================================================
N_SPLITS = int(os.environ.get("N_SPLITS", 10))
N_REPEATS = int(os.environ.get("N_REPEATS", 3))
SEEDS = [42, 101]
CV_SEED = 42
MAX_ITERS = int(os.environ.get("MAX_ITERS", 2000))
CORR_THRESHOLD = 0.90
MIN_KEEP = 20
IMP_Z = float(os.environ.get("IMP_Z", 1.0))
INNER_VAL = 0.15

# Modelli inclusi nel training
GBM_NAMES = ["cb_sym", "cb_loss", "lgb", "et"]

# --- PESI DEL BLEND OTTIMIZZATI ---
# Diamo il 70% complessivo a CatBoost, 25% a Logreg e azzeriamo o minimizziamo et e lgb
MODEL_WEIGHTS = {
    "cb_sym": 0.35,
    "cb_loss": 0.35,
    "lgb": 0.02,
    "et": 0.03,
    "logreg": 0.25
}

DROP_COLS = [
    "log_hp_ratio", "n_turns", "hp_adv_max", "fainted_diff", "hp_adv_trend", "p2team_spe_diff",
    "team_hp_mean_diff", "team_hp_diff", "p2team_spec_bulk_diff", "hp_adv_ewm",
    "log_hp_ratio_last10", "hp_adv_last5",
]
IMP_LOG = []

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
# 2. SELEZIONE FEATURE
# =========================================================
def corr_prune(X, thr=CORR_THRESHOLD):
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
    keep = imp[(imp > 0) & (imp > IMP_Z * std)].index.tolist()
    if len(keep) < MIN_KEEP:
        keep = imp.sort_values(ascending=False).index[:MIN_KEEP].tolist()
    IMP_LOG.append((imp, set(keep)))
    return keep


# =========================================================
# 3. MODELLI
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
    if name == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=min(iters, 500), max_depth=12, min_samples_leaf=15,
                max_features=0.8, random_state=seed, n_jobs=-1)
        )
    raise ValueError(name)


def fit_logreg(X, y):
    lr = LogisticRegressionCV(Cs=np.logspace(-3, 0, 8), cv=5, scoring="neg_log_loss",
                              max_iter=2000, n_jobs=-1)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr).fit(X, y)


def make_ebm():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExplainableBoostingClassifier(interactions=5, learning_rate=0.02, outer_bags=4,
                                      max_bins=128, random_state=0, n_jobs=-1))


EXTRA_NAMES = ["logreg"] + (["ebm"] if HAS_EBM else [])
MODEL_NAMES = GBM_NAMES + EXTRA_NAMES


def blend(p, w=None):
    weights = w or MODEL_WEIGHTS
    tot = sum(weights.get(k, 0) for k in MODEL_NAMES if k in p)
    return sum(weights.get(k, 0) * p[k] for k in MODEL_NAMES if k in p) / (tot + 1e-8)


rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_SEED)
folds = list(rskf.split(X_train, y_train))

# ---------- Fase A: stima n_iter ----------
print("\n--- FASE A: stima iterazioni (early stopping su split interno al train del fold) ---")
selected = {}
best_iters = {k: [] for k in GBM_NAMES if k != "et"}
for fi, (tr, va) in enumerate(folds[:N_SPLITS]):
    X_tr, y_tr = X_train.iloc[tr], y_train.iloc[tr]
    cols = select_features(X_tr, y_tr)
    selected[fi] = cols
    Xi, Xv, yi, yv = train_test_split(X_tr[cols], y_tr, test_size=INNER_VAL,
                                      stratify=y_tr, random_state=fi)
    for name in GBM_NAMES:
        if name == "et":
            continue
        m = make_model(name, CV_SEED, MAX_ITERS)
        if name == "lgb":
            m.fit(Xi, yi, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                  callbacks=[early_stopping(100, verbose=False)])
            best_iters[name].append(m.best_iteration_ or MAX_ITERS)
        else:
            m.fit(Xi, yi, eval_set=(Xv, yv), early_stopping_rounds=100, verbose=False)
            best_iters[name].append(m.get_best_iteration() + 1)
    print(f"  fold {fi + 1}: {len(cols)} feature tenute")

N_ITERS = {k: max(150, int(np.median(v) / (1 - INNER_VAL))) for k, v in best_iters.items()}
N_ITERS["et"] = 400
print("Iterazioni scelte (mediana sui fold):", N_ITERS)

# ---------- Fase B: CV ripetuta ----------
print(f"\n--- FASE B: {N_SPLITS}-fold CV x {N_REPEATS} ripetizioni, iterazioni fisse ---")
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
# 4. RISULTATI
# =========================================================
def summarize(oof_mat):
    accs = [accuracy_score(y_train, oof_mat[r] >= 0.5) * 100 for r in range(N_REPEATS)]
    aucs = [roc_auc_score(y_train, oof_mat[r]) for r in range(N_REPEATS)]
    lls = [log_loss(y_train, np.clip(oof_mat[r], 1e-6, 1 - 1e-6)) for r in range(N_REPEATS)]
    return accs, aucs, lls

all_cols = list(X_train.columns)
imp_df = pd.DataFrame({
    "importance_mean": pd.concat([i for i, _ in IMP_LOG], axis=1).reindex(all_cols).fillna(0).mean(axis=1),
    "kept_frac": [np.mean([c in kept for _, kept in IMP_LOG]) for c in all_cols],
}, index=all_cols).sort_values("importance_mean", ascending=False)
imp_df.to_csv("feature_importance.csv")
print("Report completo salvato in 'feature_importance.csv'")

print("\n--- OOF per modello (media +/- std sulle ripetizioni) ---")
for k in MODEL_NAMES:
    a, u, l = summarize(oof[k])
    print(f"{k:8s} acc = {np.mean(a):.2f}% +/- {np.std(a):.2f} | AUC = {np.mean(u):.4f} | logloss = {np.mean(l):.4f}")

oof_blend = blend(oof)
a, u, l = summarize(oof_blend)
se = np.sqrt(np.mean(a) / 100 * (1 - np.mean(a) / 100) / len(y_train)) * 100

print("\n" + "=" * 65)
print(" RISULTATI FINALI (blend con pesi individuali ottimizzati)")
print("=" * 65)
print(f"Accuratezza OOF:      {np.mean(a):.2f}% (std tra ripetizioni {np.std(a):.2f}%, errore standard ~{se:.2f}%)")
print(f"ROC-AUC OOF:          {np.mean(u):.4f} (std {np.std(u):.4f})")
print(f"Log-loss OOF:         {np.mean(l):.4f} (std {np.std(l):.4f})")
print(f"Gap medio Train-Val:  {np.mean(gaps):.2f}%")
print(f"Feature medie tenute: {np.mean(n_kept):.1f} su {X_train.shape[1]}")
print("=" * 65)

test_blend = blend(test_pred)
submission = pd.DataFrame({
    "battle_id": test_df["battle_id"],
    "player_won": (test_blend >= 0.5).astype(int),
})
submission.to_csv("submission.csv", index=False)
print(f"\n[OK] 'submission.csv' generato. Predetti positivi: {submission['player_won'].mean() * 100:.1f}%")