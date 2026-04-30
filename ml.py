from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
import polars as pl
import os
import time

model_cache = {}
prediction_cache = {}

FEATURE_COLUMNS = [
    'RSI', 'MACD', 'SMA_50_Dist', 'SMA_100_Dist', 'Volatility',
    'returns_1d', 'returns_5d', 'momentum_pct',
    'volatility_5d', 'sma_ratio'
]
PREDICTION_WEIGHTS = np.array([1, 2, 3, 4, 5], dtype=np.float64)
PREDICTION_WEIGHT_SUM = PREDICTION_WEIGHTS.sum()
MODEL_CACHE_TTL_SECONDS = 3600
PREDICTION_CACHE_TTL_SECONDS = 300

def ensure_polars_frame(df):
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df, include_index=False)
    if isinstance(df, list):
        return pl.from_dicts(df)
    if isinstance(df, dict):
        return pl.DataFrame(df)
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def _prepare_data_reference(df):
    pdf = df if isinstance(df, pd.DataFrame) else ensure_polars_frame(df).to_pandas()
    close = pdf["Close"].to_numpy(dtype=np.float32, copy=False)
    rsi = pdf["RSI"].to_numpy(dtype=np.float32, copy=False)
    macd = pdf["MACD"].to_numpy(dtype=np.float32, copy=False)
    sma_50 = pdf["SMA_50"].to_numpy(dtype=np.float32, copy=False)
    sma_100 = pdf["SMA_100"].to_numpy(dtype=np.float32, copy=False)
    volatility = pdf["Volatility"].to_numpy(dtype=np.float32, copy=False)

    row_count = close.size
    if row_count < 35:
        return (
            np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    returns_1d = np.zeros(row_count, dtype=np.float32)
    returns_1d[1:] = (close[1:] / close[:-1]) - 1

    returns_5d = np.full(row_count, np.nan, dtype=np.float32)
    returns_5d[5:] = (close[5:] / close[:-5]) - 1

    momentum_pct = np.full(row_count, np.nan, dtype=np.float32)
    momentum_pct[5:] = (close[5:] / close[:-5]) - 1

    sma_50_dist = np.divide(
        close - sma_50, sma_50, out=np.zeros_like(close), where=sma_50 != 0
    )
    sma_100_dist = np.divide(
        close - sma_100, sma_100, out=np.zeros_like(close), where=sma_100 != 0
    )

    volatility_5d = np.asarray(
        pd.Series(close, copy=False).rolling(5).std(),
        dtype=np.float32,
    )

    sma_ratio = np.divide(
        sma_50,
        sma_100,
        out=np.full(row_count, np.nan, dtype=np.float32),
        where=sma_100 != 0,
    )

    target = np.full(row_count, np.nan, dtype=np.float32)
    target[:-30] = (close[30:] / close[:-30]) - 1

    feature_matrix = np.empty((row_count, len(FEATURE_COLUMNS)), dtype=np.float32)
    feature_matrix[:, 0] = rsi
    feature_matrix[:, 1] = macd
    feature_matrix[:, 2] = sma_50_dist
    feature_matrix[:, 3] = sma_100_dist
    feature_matrix[:, 4] = volatility
    feature_matrix[:, 5] = returns_1d
    feature_matrix[:, 6] = returns_5d
    feature_matrix[:, 7] = momentum_pct
    feature_matrix[:, 8] = volatility_5d
    feature_matrix[:, 9] = sma_ratio

    valid_mask = np.isfinite(target) & np.isfinite(feature_matrix).all(axis=1)
    return feature_matrix[valid_mask], target[valid_mask]


def _validate_prepare_data(df, feature_matrix, target_vector):
    if FEATURE_COLUMNS is None:
        return
    ref_features, ref_target = _prepare_data_reference(df)
    if ref_features.shape != feature_matrix.shape or ref_target.shape != target_vector.shape:
        raise ValueError("Polars ML preparation output shape mismatch")
    if not np.allclose(ref_features, feature_matrix, equal_nan=True, atol=1e-7, rtol=1e-7):
        raise ValueError("Polars ML feature matrix mismatch")
    if not np.allclose(ref_target, target_vector, equal_nan=True, atol=1e-7, rtol=1e-7):
        raise ValueError("Polars ML target vector mismatch")


def get_prediction_cache_key(df, ticker):
    frame = ensure_polars_frame(df)
    last_date = frame.select(pl.col("Date").last()).item()
    if hasattr(last_date, "value"):
        last_date = last_date.value
    return ticker, frame.height, last_date, float(frame.select(pl.col("Close").last()).item())

def get_model(ticker, x_train, y_train):
    now = time.time()
    if ticker in model_cache:
        model, timestamp = model_cache[ticker]
        if now - timestamp < MODEL_CACHE_TTL_SECONDS:
            return model

    # changed: ExtraTrees trains materially faster than RandomForest for this endpoint while keeping nonlinear signal capacity.
    model = ExtraTreesRegressor(
        n_estimators=96,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    model_cache[ticker] = (model, now)
    return model

def prepare_data(df):
    frame = ensure_polars_frame(df).with_columns(
        [
            pl.col("Close").cast(pl.Float32, strict=False),
            pl.col("RSI").cast(pl.Float32, strict=False),
            pl.col("MACD").cast(pl.Float32, strict=False),
            pl.col("SMA_50").cast(pl.Float32, strict=False),
            pl.col("SMA_100").cast(pl.Float32, strict=False),
            pl.col("Volatility").cast(pl.Float32, strict=False),
        ]
    )

    if frame.height < 35:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.empty(0, dtype=np.float32)

    feature_frame = (
        frame.lazy()
        .with_columns(
            [
                pl.col("Close").pct_change().fill_null(0.0).alias("returns_1d"),
                pl.col("Close").pct_change(5).alias("returns_5d"),
                pl.col("Close").pct_change(5).alias("momentum_pct"),
                pl.when(pl.col("SMA_50") != 0)
                .then((pl.col("Close") - pl.col("SMA_50")) / pl.col("SMA_50"))
                .otherwise(0.0)
                .alias("SMA_50_Dist"),
                pl.when(pl.col("SMA_100") != 0)
                .then((pl.col("Close") - pl.col("SMA_100")) / pl.col("SMA_100"))
                .otherwise(0.0)
                .alias("SMA_100_Dist"),
                pl.col("Close").rolling_std(window_size=5, min_samples=5).alias("volatility_5d"),
                pl.when(pl.col("SMA_100") != 0)
                .then(pl.col("SMA_50") / pl.col("SMA_100"))
                .otherwise(None)
                .alias("sma_ratio"),
                ((pl.col("Close").shift(-30) / pl.col("Close")) - 1.0).alias("__target"),
            ]
        )
        .select(
            [
                "RSI",
                "MACD",
                "SMA_50_Dist",
                "SMA_100_Dist",
                "Volatility",
                "returns_1d",
                "returns_5d",
                "momentum_pct",
                "volatility_5d",
                "sma_ratio",
                "__target",
            ]
        )
        .collect()
    )

    feature_matrix = feature_frame.select(FEATURE_COLUMNS).to_numpy().astype(np.float32, copy=False)
    target_vector = feature_frame.get_column("__target").to_numpy().astype(np.float32, copy=False)
    valid_mask = np.isfinite(target_vector) & np.isfinite(feature_matrix).all(axis=1)
    feature_matrix = feature_matrix[valid_mask]
    target_vector = target_vector[valid_mask]

    if os.getenv("VALIDATE_POLARS_ML") == "1":
        _validate_prepare_data(df, feature_matrix, target_vector)

    return feature_matrix, target_vector


def get_ml_predictions(data_matrix, ticker):
    cache_key = get_prediction_cache_key(data_matrix, ticker)
    now = time.time()
    cached = prediction_cache.get(cache_key)
    if cached and now - cached[0] < PREDICTION_CACHE_TTL_SECONDS:
        return cached[1]

    feature_matrix, target_vector = prepare_data(data_matrix)

    if len(feature_matrix) < 10:
        return 0.0

    split = max(1, int(len(feature_matrix) * 0.8))
    x_train = feature_matrix[:split]
    y_train = target_vector[:split]

    model = get_model(ticker, x_train, y_train)

    recent_window = min(5, len(feature_matrix))
    latest = feature_matrix[-recent_window:]
    preds = model.predict(latest)

    if recent_window == PREDICTION_WEIGHTS.size:
        weighted_return = (preds * PREDICTION_WEIGHTS).sum() / PREDICTION_WEIGHT_SUM
    else:
        weights = PREDICTION_WEIGHTS[-recent_window:]
        weighted_return = (preds * weights).sum() / weights.sum()
    
    result = float(np.clip(weighted_return, -0.20, 0.20))
    prediction_cache[cache_key] = (now, result)
    return result/100
