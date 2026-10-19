from pathlib import Path
import joblib
import pandas as pd
from common import FEATURES, INTERVAL, load_consumption_series

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "quantile_models.joblib"


def has_history(ts, series):
    times = pd.date_range(end=ts - pd.Timedelta(minutes=1), periods=1440, freq=INTERVAL)
    return not series.reindex(times).isna().any()


def find_analogue(ts, series):
    idx = series.index
    conditions = [
        (idx.month == ts.month) & (idx.dayofweek == ts.dayofweek) & (idx.hour == ts.hour) & (idx.minute == ts.minute),
        (idx.month == ts.month) & (idx.hour == ts.hour) & (idx.minute == ts.minute),
        (idx.hour == ts.hour) & (idx.minute == ts.minute),
    ]
    for condition in conditions:
        candidates = sorted(idx[condition], key=lambda c: (abs(c.dayofyear - ts.dayofyear), abs((c - ts).total_seconds())))
        for candidate in candidates:
            if has_history(candidate, series):
                return candidate
    raise ValueError("No similar historical timestamp with complete previous-24-hour readings found.")


def predict(date_time):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Quantile models not found. Run `python train.py` first.")
    try:
        requested = pd.to_datetime(date_time, format="%Y-%m-%d %H:%M", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Use date/time format YYYY-MM-DD HH:MM.") from exc
    models = joblib.load(MODEL_PATH)["models"]
    series = load_consumption_series()
    if requested in series.index and has_history(requested, series):
        anchor, source = requested, "matching historical timestamp"
    else:
        anchor, source = find_analogue(requested, series), "similar historical timestamp"
    recent_times = pd.date_range(end=anchor - pd.Timedelta(minutes=1), periods=60, freq=INTERVAL)
    recent = series.reindex(recent_times)
    lags = [series.get(anchor - pd.Timedelta(minutes=1)), series.get(anchor - pd.Timedelta(hours=1)), series.get(anchor - pd.Timedelta(hours=24))]
    if any(pd.isna(x) for x in lags) or recent.isna().any():
        raise ValueError("Historical feature readings are incomplete; try another date/time.")
    row = pd.DataFrame([{
        "hour": requested.hour, "day": requested.day, "month": requested.month,
        "day_of_week": requested.dayofweek, "is_weekend": int(requested.dayofweek >= 5),
        "lag_1min": float(lags[0]), "lag_1hour": float(lags[1]),
        "lag_24hours": float(lags[2]), "rolling_mean_1hour": float(recent.mean()),
    }], columns=FEATURES)
    result = {q: float(models[q].predict(row)[0]) for q in (0.10, 0.50, 0.90)}
    print("\n=== ELECTRICITY QUANTILE PREDICTION ===")
    print("Requested:", requested.strftime("%Y-%m-%d %H:%M"))
    print("Reference:", anchor.strftime("%Y-%m-%d %H:%M"), f"({source})")
    print(f"Q10 lower estimate:  {result[0.10]:.3f} kW")
    print(f"Q50 median estimate: {result[0.50]:.3f} kW")
    print(f"Q90 upper estimate:  {result[0.90]:.3f} kW")
    if not result[0.10] <= result[0.50] <= result[0.90]:
        print("Warning: quantile crossing detected; estimates are not ordered.")
    if source == "similar historical timestamp":
        print("Note: historical-pattern estimate, not a full future forecast.")
    return result


def main():
    while True:
        print("\n=== ELECTRICITY QUANTILE REGRESSION ===\n1. Predict Q10/Q50/Q90\n2. Exit")
        choice = input("Choose (1-2): ").strip()
        if choice == "1":
            date = input("Date (YYYY-MM-DD): ").strip()
            time = input("Time (HH:MM, 24-hour): ").strip()
            try:
                predict(f"{date} {time}")
            except (ValueError, FileNotFoundError) as exc:
                print(f"Prediction error: {exc}")
        elif choice == "2":
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
