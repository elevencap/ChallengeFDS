# ChallengeFDS – Pokémon Battle Prediction (FDS 2026-2027)

Predict whether Player 1 wins a Gen-2 OU Pokémon battle from its first 30 turns.
Only classical ML models are used. All features come from the competition data and the Gen-2 type chart.

## Repository structure

| File | Content |
|---|---|
| `features.py` | Feature extraction: `build(path)` turns a `.jsonl` file into one row of features per battle |
| `train.py` | Cross-validation, model training, blending and creation of `submission.csv` |
| `type_chart.py` | Gen-2 type-effectiveness chart |
| `pokemonDB.py` | Builds `pokemon_db.json`, the base stats and types of every species observed in `train.jsonl` and `test.jsonl` |
| `train2.py` | Extra Trees baseline used during model comparison |
| `ablation.py` | Feature selection: permutation importance and grouped ablation with repeated stratified K-fold |
| `retrieveinfo.py` | Exploratory data analysis |

## How to run

```bash
pip install numpy pandas scikit-learn lightgbm catboost xgboost scipy
# put train.jsonl and test.jsonl in the repository folder
python train.py          # prints the CV results and writes submission.csv
```

## Final submissions

Each submission is produced by `train.py` at the git reference below.
The Kaggle notebook clones this repository and checks out each reference in turn.

| # | Git reference | Model | Local CV acc. | Public LB | Private LB |
|---|---|---|---|---|---|
| 1 (best) | branch `submission-best` | CatBoost + XGBoost, logistic-regression meta-learner, isotonic calibration | 68.01 % | 0.69066 | 0.66742 |
| 2 | commit `39b802d` (tag `submission-2`) | LightGBM + 2 × CatBoost, SLSQP-weighted blend | 67.96 % | 0.68933 | 0.66657 |
| 3 | commit `be656da` (tag `submission-3`) | Random Forest + CatBoost, soft voting | 66.58 % | 0.68866 | 0.66257 |

### Submission 1 – CatBoost + XGBoost stacking (best)

**Features (79).** Three groups:

- **Static team features:** stat differences between P1's lead/team and P2's lead; physical and special bulk; bulk of Leftovers holders; 4× type weaknesses; cross "pressure" features (P1 offence minus P2 defence, and the reverse).
- **Estimated P2 team:** built from the species revealed in the 30 turns, with unseen slots filled with the average species.
- **Timeline features:** HP trajectory and momentum, KOs, status and sleep pressure, switches, boosts, move power and effectiveness, Spikes/Rapid Spin, Explosion/Rest, and the HP of each side's bulkiest Pokémon ("main tank").

**Model.** Stratified 5-fold CV run with 3 seeds. In each fold:

- **CatBoost:** symmetric trees, depth 4, L2 = 8, with higher feature weights on key features.
- **XGBoost:** depth 3, strong L1/L2, 75 % row and column subsampling, without the species-name columns.

Both use early stopping. An elastic-net logistic regression is fitted on the out-of-fold (OOF) predictions as meta-learner. Its output is calibrated with isotonic regression, and the decision threshold is chosen on the OOF predictions.

### Submission 2 – LightGBM + 2 × CatBoost blend

**Features.** Same feature pipeline at an earlier stage. It used type-matchup (STAB) features instead of the cross-pressure ones, and tracked Snorlax HP instead of the main tank.

**Model.** Same 5-fold CV with LightGBM (leaf-wise) and two CatBoost models (symmetric trees and Lossguide). The blend weights minimise the OOF log-loss (SLSQP), and the threshold is tuned on the OOF blend.

### Submission 3 – Random Forest + CatBoost

**Features.** First version of the feature set (58 features).

**Model.** Random Forest (500 trees, depth 15) and CatBoost (500 iterations, depth 6, native categorical features) are evaluated with 5-fold CV. The final models are trained on all training data and their probabilities are averaged (soft voting).
