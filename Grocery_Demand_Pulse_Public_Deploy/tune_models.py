# Paste after Section 4 of the notebook (needs train, test, FEATURES)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, balanced_accuracy_score)

# validation = last 2 TRAIN days (used to choose settings); test days are touched only once at the end
vcut = train.day_no.max() - 1
fit, val = train[train.day_no < vcut], train[train.day_no >= vcut]
rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))

# ---------------- REGRESSION ----------------
grid = {
 "Linear": [LinearRegression()],
 "Ridge": [make_pipeline(StandardScaler(), Ridge(alpha=a)) for a in (1, 10, 100, 300)],
 "Lasso": [make_pipeline(StandardScaler(), Lasso(alpha=a)) for a in (0.01, 0.05, 0.1, 0.3)],
 "ElasticNet": [make_pipeline(StandardScaler(), ElasticNet(alpha=a, l1_ratio=.5)) for a in (0.01, 0.05, 0.1)],
 "Random Forest": [RandomForestRegressor(n_estimators=150, max_depth=d, min_samples_leaf=20, random_state=1) for d in (3, 5)],
 "Gradient Boosting": [GradientBoostingRegressor(n_estimators=n, learning_rate=.03, max_depth=2, subsample=.8, random_state=1) for n in (50, 150)],
}
reg_rows, best_reg = [], {}
for fam, cands in grid.items():
    scores = [rmse(val.qty, np.clip(m.fit(fit[FEATURES], fit.qty).predict(val[FEATURES]), 0, None)) for m in cands]
    m = cands[int(np.argmin(scores))]                         # best setting for this family (by validation RMSE)
    m.fit(train[FEATURES], train.qty)
    p = np.clip(m.predict(test[FEATURES]), 0, None)
    best_reg[fam] = m
    reg_rows.append({"Model": fam, "Chosen setting": str(m[-1] if hasattr(m, "steps") else m)[:60],
                     "MAE": mean_absolute_error(test.qty, p), "RMSE": rmse(test.qty, p), "R2": r2_score(test.qty, p)})
reg_table = pd.DataFrame(reg_rows).set_index("Model").sort_values("RMSE")

# ---------------- CLASSIFICATION (will it sell tomorrow?) ----------------
cgrid = {
 "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
 "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=1),
 "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=15, class_weight="balanced", random_state=1),
 "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=.05, max_depth=2, random_state=1),
}
clf_rows = []
for name, m in cgrid.items():
    m.fit(fit[FEATURES], fit.sold)
    pv = m.predict_proba(val[FEATURES])[:, 1]
    # threshold chosen on VALIDATION days by balanced accuracy (avoids the trivial 'always yes/no' answer)
    thr = max(np.arange(.2, .8, .025), key=lambda t: balanced_accuracy_score(val.sold, pv >= t))
    m.fit(train[FEATURES], train.sold)
    pt = m.predict_proba(test[FEATURES])[:, 1]
    pc = (pt >= thr).astype(int)
    clf_rows.append({"Model": name, "Threshold": round(thr, 3), "AUC": roc_auc_score(test.sold, pt),
                     "Accuracy": accuracy_score(test.sold, pc), "Precision": precision_score(test.sold, pc, zero_division=0),
                     "Recall": recall_score(test.sold, pc, zero_division=0), "F1": f1_score(test.sold, pc, zero_division=0)})
base = np.zeros(len(test))
clf_rows.append({"Model": "Baseline: always 'no sale'", "Threshold": None, "AUC": 0.5, "Accuracy": accuracy_score(test.sold, base),
                 "Precision": 0.0, "Recall": 0.0, "F1": 0.0})
clf_table = pd.DataFrame(clf_rows).set_index("Model").sort_values("F1", ascending=False)
print(reg_table.round(3).to_string()); print(); print(clf_table.round(3).to_string())
