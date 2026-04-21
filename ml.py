from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

model_cache = {}

def get_model(ticker, x_train, y_train):
    now = time.time()

    if ticker in model_cache:
        model, timestamp = model_cache[ticker]

        if now - timestamp < 3600:  # 1 hour cache
            return model

    model = RandomForestRegressor(n_estimators=200,
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

    features = [
        'RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility',
        'returns_1d', 'returns_5d', 'momentum',
        'volatility_5d', 'sma_ratio'
    ]

    x = ml_df[features]
    y = ml_df['target']

    return x, y, ml_df


def get_ml_predictions(df, ticker):
    x, y, ml_df = prepare_data(df)

    if x.empty or y.empty:
        raise ValueError("Not enough clean data for prediction")

    # --- Time-based split (prevents leakage) ---
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # --- Model ---
    model = get_model(ticker, x_train, y_train)

    # --- Optional evaluation (you can log this) ---
    # Example metric (directional accuracy)
    
   
    # print(f"Directional Accuracy: {accuracy:.2f}")

    # --- Prediction (use last 5 rows, weighted) ---
    latest = ml_df[x.columns].iloc[-5:]
    preds = model.predict(latest)

    weights = np.array([1, 2, 3, 4, 5])
    weighted_pred = (preds * weights).sum() / weights.sum()
    weighted_pred = np.clip(weighted_pred, -0.5, 0.5)
    # --- Convert return → price ---
    current_price = df['Close'].iloc[-1]
    predicted_price = current_price * (1 + weighted_pred)
    predicted_price = predicted_price / 100.0  # Scale down to realistic range

    return float(predicted_price)