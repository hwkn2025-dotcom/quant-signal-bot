import os
import requests
from datetime import datetime, timedelta, timezone

def get_env(name, required=True, default=None):
    value = os.getenv(name)

    if value is None or value.strip() == "":
        if required:
            raise ValueError(f"Missing GitHub Secret: {name}")
        return default

    return value.strip()

def send_telegram(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message
    }

    response = requests.post(url, json=data, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Telegram API error: {response.status_code} - {response.text}")

def get_alpaca_daily_bars(symbol, api_key, api_secret):
    url = "https://data.alpaca.markets/v2/stocks/bars"

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=450)

    params = {
        "symbols": symbol,
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "all",
        "feed": "iex",
        "limit": 1000
    }

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Alpaca API error: {response.status_code} - {response.text}")

    data = response.json()
    bars = data.get("bars", {}).get(symbol, [])

    if len(bars) == 0:
        raise RuntimeError(f"No data returned for {symbol}")

    return bars

if __name__ == "__main__":
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")
    total_cash = float(get_env("TOTAL_CASH", required=False, default="10000"))

    alpaca_api_key = get_env("ALPACA_API_KEY")
    alpaca_api_secret = get_env("ALPACA_API_SECRET")

    symbol = "QQQ"
    bars = get_alpaca_daily_bars(symbol, alpaca_api_key, alpaca_api_secret)

    last_bar = bars[-1]
    last_date = last_bar["t"][:10]
    last_close = last_bar["c"]

    message = f"""✅ Alpaca 資料測試成功

📌 標的：{symbol}
📅 最新日期：{last_date}
💵 最新收盤價：${last_close:.2f}
📊 抓到資料筆數：{len(bars)}

💰 目前設定總資金：
${total_cash:,.2f}

下一步：接入完整策略，開始計算權重、金額、股數。
"""

    send_telegram(bot_token, chat_id, message)
    print("Alpaca data test sent successfully.")
