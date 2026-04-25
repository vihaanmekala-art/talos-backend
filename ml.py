from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import time

model_cache = {}
#changed: cache the final prediction for unchanged ticker/history inputs so repeat simulations skip feature engineering and inference.
prediction_cache = {}
#changed: hoisted feature names and weights so they are built once per process
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility',
    'returns_1d', 'returns_5d', 'momentum',
    'volatility_5d', 'sma_ratio'
]
PREDICTION_WEIGHTS = np.array([1, 2, 3, 4, 5], dtype=np.float64)
#changed: precompute the denominator so prediction averaging stays on vector math only
PREDICTION_WEIGHT_SUM = PREDICTION_WEIGHTS.sum()
MODEL_CACHE_TTL_SECONDS = 3600
PREDICTION_CACHE_TTL_SECONDS = 300

#changed: key the hot prediction cache by the latest visible market data so identical requests reuse one result.
def get_prediction_cache_key(df, ticker):
    last_date = df["Date"].iat[-1]
    if hasattr(last_date, "value"):
        last_date = last_date.value
    return ticker, len(df), last_date, float(df["Close"].iat[-1])

def get_model(ticker, x_train, y_train):
    now = time.time()

    if ticker in model_cache:
        model, timestamp = model_cache[ticker]

        if now - timestamp < MODEL_CACHE_TTL_SECONDS:  # 1 hour cache
            return model

    #changed: kept model construction centralized so cached models are reused
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    model_cache[ticker] = (model, now)
    return model
def prepare_data(df):
    #changed: build ML features directly from NumPy arrays so fresh predictions avoid copying the whole DataFrame.
    close = df["Close"].to_numpy(dtype=np.float32, copy=False)
    rsi = df["RSI"].to_numpy(dtype=np.float32, copy=False)
    macd = df["MACD"].to_numpy(dtype=np.float32, copy=False)
    sma_50 = df["SMA_50"].to_numpy(dtype=np.float32, copy=False)
    sma_100 = df["SMA_100"].to_numpy(dtype=np.float32, copy=False)
    volatility = df["Volatility"].to_numpy(dtype=np.float32, copy=False)

    row_count = close.size
    if row_count < 6:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.empty(0, dtype=np.float32)

    returns_1d = np.empty(row_count, dtype=np.float32)
    returns_1d[0] = np.nan
    returns_1d[1:] = close[1:] / close[:-1] - 1

    returns_5d = np.full(row_count, np.nan, dtype=np.float32)
    returns_5d[5:] = close[5:] / close[:-5] - 1

    momentum = np.full(row_count, np.nan, dtype=np.float32)
    momentum[5:] = close[5:] - close[:-5]

    #changed: compute the only rolling ML-only feature from a light Series instead of cloning the full market frame.
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
    target[:-30] = close[30:] / close[:-30] - 1

    feature_matrix = np.empty((row_count, len(FEATURE_COLUMNS)), dtype=np.float32)
    feature_matrix[:, 0] = rsi
    feature_matrix[:, 1] = macd
    feature_matrix[:, 2] = sma_50
    feature_matrix[:, 3] = sma_100
    feature_matrix[:, 4] = volatility
    feature_matrix[:, 5] = returns_1d
    feature_matrix[:, 6] = returns_5d
    feature_matrix[:, 7] = momentum
    feature_matrix[:, 8] = volatility_5d
    feature_matrix[:, 9] = sma_ratio

    valid_mask = np.isfinite(target) & np.isfinite(feature_matrix).all(axis=1)
    return feature_matrix[valid_mask], target[valid_mask]


def get_ml_predictions(df, ticker):
    #changed: reuse a recent prediction when the ticker's underlying history has not changed.
    cache_key = get_prediction_cache_key(df, ticker)
    now = time.time()
    cached = prediction_cache.get(cache_key)
    if cached and now - cached[0] < PREDICTION_CACHE_TTL_SECONDS:
        return cached[1]

    feature_matrix, target_vector = prepare_data(df)

    if len(feature_matrix) < 2 or len(target_vector) < 2:
        raise ValueError("Not enough clean data for prediction")

    # --- Time-based split (prevents leakage) ---
    split = max(1, int(len(feature_matrix) * 0.8))
    x_train = feature_matrix[:split]
    y_train = target_vector[:split]

    # --- Model ---
    model = get_model(ticker, x_train, y_train)

    # --- Prediction (use last 5 rows, weighted) ---
    #changed: slice the recent inference window from the shared feature matrix and trim weights when history is short.
    recent_window = min(5, len(feature_matrix))
    latest = feature_matrix[-recent_window:]
    preds = model.predict(latest)

    # Weighted average of the return predictions
    #changed: reuse the precomputed weights array without paying for shape mismatches on shorter histories.
    if recent_window == PREDICTION_WEIGHTS.size:
        weighted_return = (preds * PREDICTION_WEIGHTS).sum() / PREDICTION_WEIGHT_SUM
    else:
        weights = PREDICTION_WEIGHTS[-recent_window:]
        weighted_return = (preds * weights).sum() / weights.sum()
    
    # Clip to +/- 20% (0.20) for sanity since 30-day moves are larger than 5-day
    weighted_return = np.clip(weighted_return, -0.20, 0.20)
    
    # Return as a decimal (e.g., 0.05 for 5%)
    # Clip to +/- 20% (0.20) for sanity
    weighted_return = np.clip(weighted_return, -0.20, 0.20)
    
    # Cast to float and cache the result
    weighted_return = float(weighted_return)
    prediction_cache[cache_key] = (now, weighted_return)

    # Simply return the prediction as a decimal percent change
    return weighted_return
   
