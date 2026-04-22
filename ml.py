from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

model_cache = {}
# changed: hoisted feature names and weights so they are built once per process
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility',
    'returns_1d', 'returns_5d', 'momentum',
    'volatility_5d', 'sma_ratio'
]
PREDICTION_WEIGHTS = np.array([1, 2, 3, 4, 5], dtype=np.float64)
# changed: precompute the denominator so prediction averaging stays on vector math only
PREDICTION_WEIGHT_SUM = PREDICTION_WEIGHTS.sum()
MODEL_CACHE_TTL_SECONDS = 3600

def get_model(ticker, x_train, y_train):
    now = time.time()

    if ticker in model_cache:
        model, timestamp = model_cache[ticker]

        if now - timestamp < MODEL_CACHE_TTL_SECONDS:  # 1 hour cache
            return model

    # changed: kept model construction centralized so cached models are reused
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

    # --- Feature Engineering ---
    ml_df['returns_1d'] = ml_df['Close'].pct_change()
    ml_df['returns_5d'] = ml_df['Close'].pct_change(5)
    ml_df['momentum'] = ml_df['Close'] - ml_df['Close'].shift(5)
    ml_df['volatility_5d'] = ml_df['Close'].rolling(5).std()
    ml_df['sma_ratio'] = ml_df['SMA_50'] / ml_df['SMA_100']

    # --- Target (predict % return instead of raw price) ---
    ml_df['target'] = ml_df['Close'].pct_change(5).shift(-5)

    ml_df.dropna(inplace=True)

    # changed: reuse the shared feature list instead of rebuilding it per call
    x = ml_df[FEATURE_COLUMNS]
    y = ml_df['target']

    return x, y, ml_df


def get_ml_predictions(df, ticker):
    x, y, ml_df = prepare_data(df)

    if x.empty or y.empty:
        raise ValueError("Not enough clean data for prediction")

    # --- Time-based split (prevents leakage) ---
    split = int(len(x) * 0.8)
    # changed: hand scikit-learn compact arrays directly to reduce conversion overhead
    x_train = x.iloc[:split].to_numpy(dtype=np.float32, copy=False)
    y_train = y.iloc[:split].to_numpy(dtype=np.float32, copy=False)

    # --- Model ---
    model = get_model(ticker, x_train, y_train)

    # --- Optional evaluation (you can log this) ---
    # Example metric (directional accuracy)
    
   
    # print(f"Directional Accuracy: {accuracy:.2f}")

    # --- Prediction (use last 5 rows, weighted) ---
    # changed: predict from the existing feature slice without rebuilding a frame
    latest = x.iloc[-5:].to_numpy(dtype=np.float32, copy=False)
    preds = model.predict(latest)

    # Weighted average of the return predictions
    # changed: reuse the precomputed weights array
    weighted_return = (preds * PREDICTION_WEIGHTS).sum() / PREDICTION_WEIGHT_SUM
    
    # Clip to +/- 15% (0.15) for sanity
    weighted_return = np.clip(weighted_return, -0.15, 0.15)
    
    # Return as a percentage move (e.g., 0.025 for 2.5%)
    return float(weighted_return)
