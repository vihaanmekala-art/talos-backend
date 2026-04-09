from sklearn.ensemble import RandomForestRegressor

def prepare_data(df):
    ml_df = df.copy()
    ml_df['target'] = ml_df['Close'].shift(-5)  # Predicting 5 days into the future
    ml_df.dropna(inplace=True)
    features = ['RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility']
    x = ml_df[features]
    y = ml_df['target']

    return x, y
def get_ml_predictions(df):
    x, y = prepare_data(df)
    if x.empty or y.empty:
        raise ValueError("Not enough clean data for prediction")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    latest_indicators = df[['RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility']].iloc[[-1]]
    predictions = model.predict(latest_indicators)[0]
    return float(predictions)