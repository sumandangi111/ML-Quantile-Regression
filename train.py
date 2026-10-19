from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from common import FEATURES, TARGET, create_supervised_dataset, load_consumption_series

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "quantile_models.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.txt"
QUANTILES = (0.10, 0.50, 0.90)


def main():
    data = create_supervised_dataset(load_consumption_series())
    if len(data) < 100:
        raise ValueError(f"Not enough usable rows: {len(data)}")
    X, y = data[FEATURES], data[TARGET]
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    models, preds = {}, {}
    lines = ["ELECTRICITY QUANTILE REGRESSION METRICS", f"Train rows: {len(X_train):,}", f"Test rows: {len(X_test):,}", ""]
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=120, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[q], preds[q] = model, pred
        lines.append(f"Q{int(q*100)} pinball loss: {mean_pinball_loss(y_test, pred, alpha=q):.6f}")
        if q == 0.5:
            lines.append(f"Q50 MAE: {mean_absolute_error(y_test, pred):.6f} kW")
        lines.append(f"Q{int(q*100)} empirical coverage: {np.mean(y_test.to_numpy() <= pred):.4f} (target about {q:.2f})")
    actual = y_test.to_numpy()
    lines.append(f"Q10-Q90 interval coverage: {np.mean((actual >= preds[0.10]) & (actual <= preds[0.90])):.4f} (target about 0.80)")
    lines.append(f"Mean Q10-Q90 interval width: {np.mean(preds[0.90] - preds[0.10]):.6f} kW")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "quantiles": QUANTILES, "features": FEATURES}, MODEL_PATH)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nModels saved: {MODEL_PATH}\nMetrics saved: {REPORT_PATH}")

if __name__ == "__main__":
    main()
