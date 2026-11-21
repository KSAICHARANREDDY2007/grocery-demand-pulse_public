# Model comparison cell - paste after Section 4 of the notebook (needs train, test, FEATURES)
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    "Baseline (same as yesterday)": None,
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (deg 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
    "Polynomial Regression (deg 3)": make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression()),
    "Decision Tree (depth 3)": DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=1),
    "Decision Tree (depth 6)": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=1),
    "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=1),
}
rows = []
for name, m in models.items():
    p = test["yesterday"].values if m is None else np.clip(m.fit(train[FEATURES], train["qty"]).predict(test[FEATURES]), 0, None)
    agg = pd.DataFrame({"s": test["sku"].values, "y": test["qty"].values, "p": p}).groupby("s").sum()   # per-SKU total over test days
    rows.append({"Model": name,
                 "MAE": mean_absolute_error(test["qty"], p),
                 "RMSE": np.sqrt(mean_squared_error(test["qty"], p)),
                 "R2": r2_score(test["qty"], p),
                 "Accuracy % (3-day SKU totals)": 100 * (1 - (agg.y - agg.p).abs().sum() / agg.y.sum())})
comparison = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
comparison.round(3)
