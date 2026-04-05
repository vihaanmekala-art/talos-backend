import requests
import os
from dotenv import load_dotenv

load_dotenv()


BASE_URL = "https://data.alpaca.markets/v2"
HEADERS = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET")
}

def get_multiple_prices(symbols):
   
    url = f"{BASE_URL}/stocks/quotes/latest?symbols={symbols}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

# Try this:
print(get_multiple_prices("AAPL,TSLA,MSFT"))