from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
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

def get_prediction_cache_key(df, ticker):
    last_date = df["Date"].iat[-1]
    if hasattr(last_date, "value"):
        last_date = last_date.value
    return ticker, len(df), last_date, float(df["Close"].iat[-1])

def get_model(ticker, x_train, y_train):
    now = time.time()
    if ticker in model_cache:
        model, timestamp = model_cache[ticker]
        if now - timestamp < MODEL_CACHE_TTL_SECONDS:
            return model

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
    # build ML features directly from NumPy arrays
    close = df["Close"].to_numpy(dtype=np.float32, copy=False)
    rsi = df["RSI"].to_numpy(dtype=np.float32, copy=False)
    macd = df["MACD"].to_numpy(dtype=np.float32, copy=False)
    sma_50 = df["SMA_50"].to_numpy(dtype=np.float32, copy=False)
    sma_100 = df["SMA_100"].to_numpy(dtype=np.float32, copy=False)
    volatility = df["Volatility"].to_numpy(dtype=np.float32, copy=False)

    row_count = close.size
    if row_count < 35: # Increased minimum to ensure windows for SMAs and Targets
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), np.empty(0, dtype=np.float32)

    # 1d Returns
    returns_1d = np.zeros(row_count, dtype=np.float32)
    returns_1d[1:] = (close[1:] / close[:-1]) - 1

    # 5d Returns
    returns_5d = np.full(row_count, np.nan, dtype=np.float32)
    returns_5d[5:] = (close[5:] / close[:-5]) - 1

    # Momentum as a percentage change instead of raw price difference
    momentum_pct = np.full(row_count, np.nan, dtype=np.float32)
    momentum_pct[5:] = (close[5:] / close[:-5]) - 1

    # SMA Distances (Normalizing price to percentages)
    # This prevents the model from seeing absolute prices like $200
    sma_50_dist = np.divide(close - sma_50, sma_50, out=np.zeros_like(close), where=sma_50!=0)
    sma_100_dist = np.divide(close - sma_100, sma_100, out=np.zeros_like(close), where=sma_100!=0)

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

    # Target: 30-day forward return (The percentage change we want to predict)
    target = np.full(row_count, np.nan, dtype=np.float32)
    target[:-30] = (close[30:] / close[:-30]) - 1

    feature_matrix = np.empty((row_count, len(FEATURE_COLUMNS)), dtype=np.float32)
    feature_matrix[:, 0] = rsi
    feature_matrix[:, 1] = macd
    feature_matrix[:, 2] = sma_50_dist  # Fixed: Percentage instead of Price
    feature_matrix[:, 3] = sma_100_dist # Fixed: Percentage instead of Price
    feature_matrix[:, 4] = volatility
    feature_matrix[:, 5] = returns_1d
    feature_matrix[:, 6] = returns_5d
    feature_matrix[:, 7] = momentum_pct # Fixed: Percentage instead of Price
    feature_matrix[:, 8] = volatility_5d
    feature_matrix[:, 9] = sma_ratio

    valid_mask = np.isfinite(target) & np.isfinite(feature_matrix).all(axis=1)
    return feature_matrix[valid_mask], target[valid_mask]


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
    return result