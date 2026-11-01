import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("battles_features.csv")
X = df.drop(columns=["battle_id", "target"])
y = df["target"]

cat_cols = [c for c in X.columns
            if c.endswith("_name") or c.endswith("_item") or c.endswith("status_last")]
X[cat_cols] = X[cat_cols].astype("category")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
print(model.score(X_te, y_te))