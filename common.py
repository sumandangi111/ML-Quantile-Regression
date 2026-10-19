from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "household_power_consumption.txt"
TARGET = "Global_active_power"
INTERVAL = "1min"
FEATURES = ["hour", "day", "month", "day_of_week", "is_weekend", "lag_1min", "lag_1hour", "lag_24hours", "rolling_mean_1hour"]


def load_consumption_series():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Put household_power_consumption.txt in data/.")
    df = pd.read_csv(DATA_PATH, sep=";", na_values=["?"], low_memory=False)
    required = ["Date", "Time", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df["datetime"] = pd.to_datetime(df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["datetime", TARGET]).sort_values("datetime").drop_duplicates("datetime", keep="last")
    if df.empty:
        raise ValueError("No valid historical readings found.")
    return df.set_index("datetime")[TARGET].resample(INTERVAL).mean().sort_index()


def create_supervised_dataset(series):
    frame = pd.DataFrame(index=series.index)
    frame[TARGET] = series
    frame["hour"] = frame.index.hour
    frame["day"] = frame.index.day
    frame["month"] = frame.index.month
    frame["day_of_week"] = frame.index.dayofweek
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    frame["lag_1min"] = series.shift(1)
    frame["lag_1hour"] = series.shift(60)
    frame["lag_24hours"] = series.shift(1440)
    frame["rolling_mean_1hour"] = series.shift(1).rolling(60, min_periods=60).mean()
    return frame.dropna(subset=FEATURES + [TARGET])


