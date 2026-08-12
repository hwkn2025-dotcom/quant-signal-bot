import os
import json
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np


# =========================
# 策略參數
# =========================

TECH = [
    "FNGS", "QQQ", "IYW", "SMH", "TSM", "STX", "NVDA",
    "SOXX", "XLK", "VUG", "IWF", "PLTR", "WDC"
]

DEFENSIVE = [
    "BIL", "SHY", "IEF", "GLD", "TLT", "XLV", "VDE"
]

ALL_SYMBOLS = sorted(set(TECH + DEFENSIVE + ["QQQ", "FNGS"]))

TOP_BULL = 4
TOP_NTECH = 2
TOP_DEF = 3

BULL_MULT = 1.03
CHANGE_THRESH = 0.20
STRONG = 0.10
EXPOSURE = 1.0

STATE_FILE = Path("state.json")


# =========================
# 基本工具
# =========================

def get_env(name, required=True, default=None):
    value = os.getenv(name)

    if value is None or value.strip() == "":
        if required:
            raise ValueError(f"缺少 GitHub Secret：{name}")
        return default

    return value.strip()


def clamp(x, low, high):
    return max(low, min(high, x))


def fmt_money(x):
    return f"${x:,.2f}"


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def send_telegram(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    chunks = [message[i:i + 3900] for i in range(0, len(message), 3900)]

    for chunk in chunks:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": chunk
            },
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Telegram 發送失敗：{response.status_code} - {response.text}"
            )


# =========================
# Alpaca 抓資料
# =========================

def fetch_alpaca_daily_bars(symbols):
    api_key = get_env("ALPACA_API_KEY")
    api_secret = get_env("ALPACA_API_SECRET")

    url = "https://data.alpaca.markets/v2/stocks/bars"

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=900)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }

    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "all",
        "feed": "iex",
        "limit": 10000
    }

    all_bars = {}
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token
        elif "page_token" in params:
            params.pop("page_token", None)

        response = requests.get(url, headers=headers, params=params, timeout=60)

        if response.status_code == 401:
            raise RuntimeError(
                "Alpaca 401 授權失敗：ALPACA_API_KEY 或 ALPACA_API_SECRET 錯誤、貼反、過期，或 GitHub Secret 沒更新成功。"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Alpaca API 錯誤：{response.status_code} - {response.text}"
            )

        data = response.json()

        for sym, bars in data.get("bars", {}).items():
            all_bars.setdefault(sym, []).extend(bars)

        page_token = data.get("next_page_token")

        if not page_token:
            break

    records = []

    for sym, bars in all_bars.items():
        for b in bars:
            records.append({
                "date": pd.to_datetime(b["t"]).date(),
                "symbol": sym,
                "close": float(b["c"])
            })

    if not records:
        raise RuntimeError("Alpaca 沒有回傳任何資料。")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["date", "symbol"], keep="last")

    close = (
        df.pivot_table(
            index="date",
            columns="symbol",
            values="close",
            aggfunc="last"
        )
        .sort_index()
    )

    for required in ["QQQ", "FNGS"]:
        if required not in close.columns or close[required].dropna().empty:
            raise RuntimeError(f"缺少必要資料：{required}")

    return close


# =========================
# 市況判斷
# =========================

def market_regime(close):
    q = close["QQQ"].dropna()
    f = close["FNGS"].dropna()

    if len(q) < 200:
        return "bear", ["QQQ 資料不足 200 日，保守判定 bear"]

    q_now = q.iloc[-1]
    q_sma200 = q.iloc[-200:].mean()
    q_sma150 = q.iloc[-150:].mean()
    q_sma100 = q.iloc[-100:].mean()
    q_sma50 = q.iloc[-50:].mean()

    reasons = []

    if q_now < q_sma200:
        reasons.append("QQQ < SMA200")
        return "bear", reasons

    if len(f) >= 100:
        f_now = f.iloc[-1]
        f_sma100 = f.iloc[-100:].mean()

        if q_now < q_sma100 and f_now < f_sma100:
            reasons.append("QQQ 與 FNGS 皆低於 SMA100")
            return "bear", reasons

    ret21 = q_now / q.iloc[-22] - 1

    if q_now < q_sma50 and ret21 < -0.08:
        reasons.append("QQQ < SMA50 且 21 日跌幅 < -8%")
        return "bear", reasons

    if q_now > q_sma150 * 1.03:
        reasons.append("QQQ > SMA150 × 1.03")
        return "bull", reasons

    reasons.append("未達 bull，也未觸發 bear")
    return "neutral", reasons


# =========================
# 評分
# =========================

def score_candidates(close, candidates):
    strong_scores = {}
    all_scores = {}

    for t in candidates:
        if t not in close.columns:
            continue

        s = close[t].dropna()

        if len(s) < 127:
            continue

        p_now = s.iloc[-1]

        r1 = p_now / s.iloc[-22] - 1
        r3 = p_now / s.iloc[-64] - 1
        r6 = p_now / s.iloc[-127] - 1

        momentum = 0.3 * r1 + 0.4 * r3 + 0.3 * r6

        daily_ret = s.pct_change().dropna()

        if len(daily_ret) < 126:
            continue

        volatility = daily_ret.iloc[-126:].std(ddof=1) * math.sqrt(252)

        if volatility <= 0 or not np.isfinite(volatility):
            continue

        sma100 = s.iloc[-100:].mean()
        trend = p_now / sma100 - 1

        drawdown = p_now / s.iloc[-63:].max() - 1

        score = momentum / volatility + trend + drawdown

        vol63 = daily_ret.iloc[-63:].std(ddof=1)

        if not np.isfinite(score):
            continue

        row = {
            "score": float(score),
            "volatility": float(volatility),
            "vol63": float(vol63) if np.isfinite(vol63) else None,
            "price": float(p_now)
        }

        all_scores[t] = row

        if score > STRONG:
            strong_scores[t] = row

    return strong_scores, all_scores


def top_n(scores, n):
    return [
        k for k, v in sorted(
            scores.items(),
            key=lambda item: item[1]["score"],
            reverse=True
        )[:n]
    ]


def get_score_value(ticker, strong_scores, all_scores):
    if ticker in strong_scores:
        return max(strong_scores[ticker]["score"], 0.0001)

    if ticker in all_scores:
        return max(all_scores[ticker]["score"], STRONG)

    return STRONG


def allocate_by_score(selected, total, strong_scores, all_scores):
    if not selected or total <= 0:
        return {}

    raw = []

    for t in selected:
        raw.append((t, get_score_value(t, strong_scores, all_scores)))

    score_sum = sum(v for _, v in raw)

    weights = {}

    if score_sum <= 0:
        equal = total / len(raw)
        for t, _ in raw:
            weights[t] = weights.get(t, 0) + equal
        return weights

    for t, v in raw:
        weights[t] = weights.get(t, 0) + total * v / score_sum

    return weights


def vol63_for(close, ticker):
    if ticker not in close.columns:
        return None

    s = close[ticker].dropna()

    if len(s) < 64:
        return None

    r = s.pct_change().dropna()

    if len(r) < 10:
        return None

    v = r.iloc[-63:].std(ddof=1)

    if not np.isfinite(v) or v <= 0:
        return None

    return float(v)


def allocate_inverse_vol(selected, total, close):
    if not selected:
        return {}

    raw = []

    for t in selected:
        v = vol63_for(close, t)

        if v is None or v <= 0:
            inv = 0
        else:
            inv = 1 / v

        raw.append((t, inv))

    inv_sum = sum(v for _, v in raw)

    weights = {}

    if inv_sum <= 0:
        equal = total / len(raw)
        for t, _ in raw:
            weights[t] = weights.get(t, 0) + equal
        return weights

    for t, inv in raw:
        weights[t] = weights.get(t, 0) + total * inv / inv_sum

    return weights


def merge_weights(a, b):
    result = dict(a)

    for k, v in b.items():
        result[k] = result.get(k, 0) + v

    return result


def volatility_scale(close):
    q = close["QQQ"].dropna()
    r = q.pct_change().dropna()

    if len(r) < 252:
        return 1.0

    current_vol = r.iloc[-21:].std(ddof=1) * math.sqrt(252)
    avg_vol = r.iloc[-252:].std(ddof=1) * math.sqrt(252)

    if current_vol <= 0 or not np.isfinite(current_vol):
        return 1.0

    return clamp(avg_vol / current_vol, 0.5, 1.0)


# =========================
# 計算目標配置
# =========================

def compute_target_weights(close):
    regime, reasons = market_regime(close)
    vol_scale = volatility_scale(close)

    tech_strong, tech_all = score_candidates(close, TECH)
    def_strong, def_all = score_candidates(close, DEFENSIVE)

    selected_info = {}

    if regime == "bull":
        selected = top_n(tech_strong, TOP_BULL)

        while len(selected) < TOP_BULL:
            selected.append("QQQ")

        total = EXPOSURE * vol_scale * BULL_MULT

        weights = allocate_by_score(
            selected,
            total,
            tech_strong,
            tech_all
        )

        selected_info["科技"] = selected

    elif regime == "neutral":
        selected_tech = top_n(tech_strong, TOP_NTECH)

        while len(selected_tech) < TOP_NTECH:
            selected_tech.append("QQQ")

        tech_total = 0.5 * vol_scale

        tech_weights = allocate_by_score(
            selected_tech,
            tech_total,
            tech_strong,
            tech_all
        )

        selected_def = top_n(def_strong, TOP_DEF)

        while len(selected_def) < TOP_DEF:
            selected_def.append("BIL")

        def_total = 0.5

        def_weights = allocate_by_score(
            selected_def,
            def_total,
            def_strong,
            def_all
        )

        weights = merge_weights(tech_weights, def_weights)

        selected_info["科技"] = selected_tech
        selected_info["防禦"] = selected_def

    else:
        selected_def = top_n(def_strong, TOP_DEF)

        while len(selected_def) < TOP_DEF:
            selected_def.append("BIL")

        weights = allocate_inverse_vol(
            selected_def,
            EXPOSURE,
            close
        )

        selected_info["防禦"] = selected_def

    weights = {
        k: float(v)
        for k, v in weights.items()
        if abs(v) > 0.000001
    }

    details = {
        "regime": regime,
        "reasons": reasons,
        "vol_scale": vol_scale,
        "selected": selected_info
    }

    return weights, details


# =========================
# 狀態保存
# =========================

def load_state():
    if not STATE_FILE.exists():
        return {}

    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def max_weight_change(new_weights, old_weights):
    keys = set(new_weights.keys()) | set(old_weights.keys())

    if not keys:
        return 1.0

    return max(
        abs(new_weights.get(k, 0) - old_weights.get(k, 0))
        for k in keys
    )


# =========================
# 報告
# =========================

def add_cash_if_needed(weights):
    result = dict(weights)
    total_weight = sum(result.values())

    if total_weight < 0.999999:
        result["CASH"] = 1.0 - total_weight

    return result


def get_latest_and_prev_price(close, ticker):
    if ticker == "CASH":
        return None, None

    if ticker not in close.columns:
        return None, None

    s = close[ticker].dropna()

    if len(s) < 2:
        return None, None

    return float(s.iloc[-1]), float(s.iloc[-2])


def format_report(
    close,
    target_weights,
    effective_weights,
    details,
    total_cash,
    max_diff,
    action_text
):
    latest_date = close.index[-1]

    display_weights = add_cash_if_needed(effective_weights)

    exposure = sum(
        w for t, w in effective_weights.items()
        if t != "CASH"
    )

    lines = []

    lines.append("📊 策略每日報告")
    lines.append("")
    lines.append(f"📅 資料日期：{latest_date}")
    lines.append(f"📌 市況：{details['regime'].upper()}")
    lines.append(f"⚙️ 波動縮放 vol_scale：{details['vol_scale']:.2f}")
    lines.append(f"🔔 換倉判斷：{action_text}")
    lines.append(f"📏 最大權重差：{fmt_pct(max_diff)} / 門檻 {fmt_pct(CHANGE_THRESH)}")
    lines.append("")
    lines.append(f"💰 設定總資金：{fmt_money(total_cash)}")
    lines.append(f"📦 總曝險：約 {fmt_pct(exposure)}")

    if exposure > 1.0001:
        lines.append("⚠️ 注意：曝險超過 100%，代表策略有使用約 3% 超額曝險 / 槓桿概念。")

    lines.append("")
    lines.append("【策略選出】")

    for bucket, selected in details["selected"].items():
        lines.append(f"{bucket}：{', '.join(selected)}")

    lines.append("")
    lines.append("【目前應持有配置】")
    lines.append("格式：標的｜權重｜金額｜估算股數｜今日P/L")

    portfolio_pl = 0.0

    sorted_items = sorted(
        display_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for ticker, weight in sorted_items:
        amount = total_cash * weight

        if ticker == "CASH":
            lines.append(
                f"{ticker}｜{fmt_pct(weight)}｜{fmt_money(amount)}｜-｜$0.00"
            )
            continue

        latest_price, prev_price = get_latest_and_prev_price(close, ticker)

        if latest_price is None or prev_price is None:
            lines.append(
                f"{ticker}｜{fmt_pct(weight)}｜{fmt_money(amount)}｜無價格資料｜無法計算P/L"
            )
            continue

        shares = amount / latest_price
        daily_ret = latest_price / prev_price - 1
        daily_pl = amount * daily_ret
        portfolio_pl += daily_pl

        sign = "+" if daily_pl >= 0 else ""

        lines.append(
            f"{ticker}｜{fmt_pct(weight)}｜{fmt_money(amount)}｜{shares:.4f}股｜{sign}{fmt_money(daily_pl)} ({fmt_pct(daily_ret)})"
        )

    portfolio_ret = portfolio_pl / total_cash if total_cash > 0 else 0
    estimated_value = total_cash + portfolio_pl

    lines.append("")
    lines.append("【投資組合每日變化】")

    sign = "+" if portfolio_pl >= 0 else ""

    lines.append(f"今日總 P/L：{sign}{fmt_money(portfolio_pl)} ({fmt_pct(portfolio_ret)})")
    lines.append(f"今日估算資金：{fmt_money(estimated_value)}")
    lines.append("")
    lines.append("註：此 P/L 是依照系統目前配置，用最新收盤價估算的每日變化。")
    lines.append("註：本系統只通知，不會自動下單。")

    return "\n".join(lines)


# =========================
# 主流程
# =========================

def main():
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")
    total_cash = float(get_env("TOTAL_CASH", required=False, default="10000"))

    close = fetch_alpaca_daily_bars(ALL_SYMBOLS)

    target_weights, details = compute_target_weights(close)

    state = load_state()
    old_weights = state.get("weights", {})

    first_run = len(old_weights) == 0

    latest_date = close.index[-1]
    current_month = str(latest_date)[:7]
    last_checked_month = state.get("last_checked_month")

    is_monthly_check = first_run or (last_checked_month != current_month)

    if first_run:
        max_diff = 1.0
    else:
        max_diff = max_weight_change(target_weights, old_weights)

    trade_action = first_run or (is_monthly_check and max_diff >= CHANGE_THRESH)

    if trade_action:
        effective_weights = target_weights
        if first_run:
            action_text = "初始化：建立第一份策略配置"
        else:
            action_text = "觸發換倉：本月檢查且最大權重變動達門檻"
    else:
        effective_weights = old_weights if old_weights else target_weights

        if not is_monthly_check:
            action_text = "非本月第一次檢查：不換倉，只回報每日 P/L"
        else:
            action_text = "本月檢查：最大權重變動未達 20%，不換倉"

    if is_monthly_check:
        state = {
            "weights": effective_weights,
            "last_checked_month": current_month,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "regime": details["regime"]
        }
        save_state(state)

    message = format_report(
        close=close,
        target_weights=target_weights,
        effective_weights=effective_weights,
        details=details,
        total_cash=total_cash,
        max_diff=max_diff,
        action_text=action_text
    )

    send_telegram(bot_token, chat_id, message)

    print("Quant report sent successfully.")


if __name__ == "__main__":
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    try:
        main()
    except Exception as e:
        error_message = f"""❌ 策略機器人執行失敗

原因：
{str(e)}

如果是 Alpaca 401，代表 ALPACA_API_KEY / ALPACA_API_SECRET 錯誤、貼反、過期，或 GitHub Secrets 沒更新成功。
"""

        if bot_token and chat_id:
            try:
                send_telegram(bot_token, chat_id, error_message)
            except Exception:
                pass

        raise
