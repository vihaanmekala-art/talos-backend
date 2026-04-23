from sklearn.ensemble import RandomForestRegressor
import numpy as np
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
    ml_df = df.copy()
    #changed: bind the close series once so all derived ML features reuse the same source values.
    close = ml_df["Close"]

    # --- Feature Engineering ---
    ml_df['returns_1d'] = close.pct_change()
    ml_df['returns_5d'] = close.pct_change(5)
    ml_df['momentum'] = close - close.shift(5)
    ml_df['volatility_5d'] = close.rolling(5).std()
    ml_df['sma_ratio'] = ml_df['SMA_50'] / ml_df['SMA_100']

    # --- Target (predict % return instead of raw price) ---
    ml_df['target'] = close.pct_change(5).shift(-5)

    ml_df.dropna(inplace=True)

    #changed: reuse the shared feature list instead of rebuilding it per call
    x = ml_df[FEATURE_COLUMNS]
    y = ml_df['target']

    return x, y, ml_df


def get_ml_predictions(df, ticker):
    #changed: reuse a recent prediction when the ticker's underlying history has not changed.
    cache_key = get_prediction_cache_key(df, ticker)
    now = time.time()
    cached = prediction_cache.get(cache_key)
    if cached and now - cached[0] < PREDICTION_CACHE_TTL_SECONDS:
        return cached[1]

    x, y, ml_df = prepare_data(df)

    if x.empty or y.empty:
        raise ValueError("Not enough clean data for prediction")

    # --- Time-based split (prevents leakage) ---
    split = int(len(x) * 0.8)
    #changed: materialize the feature matrix once so training and inference reuse the same compact NumPy buffer.
    feature_matrix = x.to_numpy(dtype=np.float32, copy=False)
    target_vector = y.to_numpy(dtype=np.float32, copy=False)
    x_train = feature_matrix[:split]
    y_train = target_vector[:split]

    # --- Model ---
    model = get_model(ticker, x_train, y_train)

    # --- Optional evaluation (you can log this) ---
    # Example metric (directional accuracy)
    
   
    # print(f"Directional Accuracy: {accuracy:.2f}")

    # --- Prediction (use last 5 rows, weighted) ---
    #changed: slice the recent inference window from the shared feature matrix and trim weights when history is short.
    recent_window = min(5, len(feature_matrix))
    latest = feature_matrix[-recent_window:]
    preds = model.predict(latest)

    # Weighted average of the return predictions
    #changed: reuse the precomputed weights array without paying for shape mismatches on shorter histories.
    weights = PREDICTION_WEIGHTS[-recent_window:]
    weighted_return = (preds * weights).sum() / weights.sum()
    
    # Clip to +/- 15% (0.15) for sanity
    weighted_return = np.clip(weighted_return, -0.15, 0.15)
    
    # Return as a percentage move (e.g., 0.025 for 2.5%)
    weighted_return = float(weighted_return)
    prediction_cache[cache_key] = (now, weighted_return)
    return weighted_return
