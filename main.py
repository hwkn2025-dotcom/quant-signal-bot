import os
import requests
from datetime import datetime

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TOTAL_CASH = float(os.getenv("TOTAL_CASH", "10000"))

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    response = requests.post(url, json=data)
    response.raise_for_status()

if __name__ == "__main__":
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    message = f"""
✅ Telegram 通知測試成功

💰 你的策略總資金設定：
${TOTAL_CASH:,.2f}

⏰ 執行時間：
{now}

下一步就可以開始接策略訊號。
"""

    send_telegram(message)
    print("Telegram message sent.")
