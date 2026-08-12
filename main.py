import os
import json
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np


# =========================
# 策略設定
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
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


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
        else:
            params.pop("page_token", None)

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=60
        )

        if response.status_code == 401:
            raise RuntimeError(
                "Alpaca 401 授權失敗：請確認 ALPACA_API_KEY / ALPACA_API_SECRET 是否正確、有沒有貼反、是否用 Paper Trading API Key。"
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

    if "QQQ" not in close.columns or close["QQQ"].dropna().empty:
        raise RuntimeError("缺少必要資料：QQQ")

    if "FNGS" not in close.columns or close["FNGS"].dropna().empty:
        raise RuntimeError("缺少必要資料：FNGS")

    return close


# =========================
# 策略邏輯
# =========================

def market_regime(close):
    q = close["QQQ"].dropna()
    f = close["FNGS"].dropna()

    if len(q) < 200:
        return "bear", ["QQQ 資料不足 200 日，保守判定熊市"]

    q_now = q.iloc[-1]
    q_sma200 = q.iloc[-200:].mean()
    q_sma150 = q.iloc[-150:].mean()
    q_sma100 = q.iloc[-100:].mean()
    q_sma50 = q.iloc[-50:].mean()

    if q_now < q_sma200:
        return "bear", ["QQQ 低於 200 日均線"]

    if len(f) >= 100:
        f_now = f.iloc[-1]
        f_sma100 = f.iloc[-100:].mean()

        if q_now < q_sma100 and f_now < f_sma100:
            return "bear", ["QQQ 與 FNGS 都低於 100 日均線"]

    ret21 = q_now / q.iloc[-22] - 1

    if q_now < q_sma50 and ret21 < -0.08:
        return "bear", ["QQQ 低於 50 日均線，且近 21 日跌幅超過 8%"]

    if q_now > q_sma150 * 1.03:
        return "bull", ["QQQ 高於 150 日均線 3% 以上"]

    return "neutral", ["未達牛市條件，也未觸發熊市條件"]


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

        if not np.isfinite(score):
            continue

        all_scores[t] = {
            "score": float(score),
            "price": float(p_now)
        }

        if score > STRONG:
            strong_scores[t] = {
                "score": float(score),
                "price": float(p_now)
            }

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
# 虛擬投資組合追蹤
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


def is_valid_state(state):
    return (
        isinstance(state, dict)
        and "holdings" in state
        and "cash" in state
        and "last_value" in state
        and "last_date" in state
        and "initial_cash" in state
    )


def latest_price(close, ticker):
    if ticker not in close.columns:
        raise RuntimeError(f"缺少價格資料：{ticker}")

    s = close[ticker].dropna()

    if len(s) == 0:
        raise RuntimeError(f"沒有可用價格：{ticker}")

    return float(s.iloc[-1])


def prev_price(close, ticker):
    if ticker not in close.columns:
        return None

    s = close[ticker].dropna()

    if len(s) < 2:
        return None

    return float(s.iloc[-2])


def build_positions_from_weights(target_weights, total_value, close):
    holdings = {}
    invested = 0.0

    for ticker, weight in target_weights.items():
        price = latest_price(close, ticker)
        amount = total_value * weight
        shares = amount / price

        holdings[ticker] = shares
        invested += amount

    cash = total_value - invested

    return holdings, cash


def portfolio_value(holdings, cash, close):
    total = float(cash)

    for ticker, shares in holdings.items():
        price = latest_price(close, ticker)
        total += float(shares) * price

    return total


def max_weight_change(new_weights, old_weights):
    keys = set(new_weights.keys()) | set(old_weights.keys())

    if not keys:
        return 1.0

    return max(
        abs(new_weights.get(k, 0) - old_weights.get(k, 0))
        for k in keys
    )


def build_position_rows(holdings, cash, close, total_value, include_daily_pl):
    rows = []

    for ticker, shares in holdings.items():
        price = latest_price(close, ticker)
        value = shares * price
        weight = value / total_value if total_value != 0 else 0

        p_prev = prev_price(close, ticker)

        if include_daily_pl and p_prev is not None:
            daily_pl = shares * (price - p_prev)
            daily_ret = price / p_prev - 1
        else:
            daily_pl = None
            daily_ret = None

        rows.append({
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": value,
            "weight": weight,
            "daily_pl": daily_pl,
            "daily_ret": daily_ret
        })

    if abs(cash) > 0.01:
        rows.append({
            "ticker": "CASH" if cash >= 0 else "MARGIN",
            "shares": None,
            "price": None,
            "value": cash,
            "weight": cash / total_value if total_value != 0 else 0,
            "daily_pl": 0.0 if include_daily_pl else None,
            "daily_ret": 0.0 if include_daily_pl else None
        })

    rows = sorted(rows, key=lambda x: x["value"], reverse=True)

    return rows


def regime_text(regime):
    r = regime.upper()

    if r == "BULL":
        return "BULL 牛市"

    if r == "NEUTRAL":
        return "NEUTRAL 中性"

    return "BEAR 熊市"


def format_report(
    latest_date,
    details,
    action_text,
    initial_cash,
    previous_value,
    current_value,
    daily_pl,
    cumulative_pl,
    max_diff,
    rows,
    include_daily_pl,
    note
):
    cumulative_ret = cumulative_pl / initial_cash if initial_cash else 0

    if daily_pl is None or previous_value is None:
        daily_line = "今日 P/L：首次建立 / 同日重跑，不計算"
    else:
        daily_ret = daily_pl / previous_value if previous_value else 0
        sign = "+" if daily_pl >= 0 else ""
        daily_line = f"今日 P/L：{sign}{fmt_money(daily_pl)} ({fmt_pct(daily_ret)})"

    cum_sign = "+" if cumulative_pl >= 0 else ""

    exposure = sum(
        row["weight"]
        for row in rows
        if row["ticker"] not in ["CASH"]
    )

    lines = []

    lines.append(f"📊 策略追蹤報告｜{latest_date}")
    lines.append("")
    lines.append(f"📌 市況：{regime_text(details['regime'])}")
    lines.append(f"🔔 狀態：{action_text}")
    lines.append(f"🧠 原因：{'、'.join(details.get('reasons', []))}")
    lines.append("")
    lines.append(f"💰 初始資金：{fmt_money(initial_cash)}")
    lines.append(f"💼 目前組合價值：{fmt_money(current_value)}")
    lines.append(f"📈 {daily_line}")
    lines.append(f"📊 累積 P/L：{cum_sign}{fmt_money(cumulative_pl)} ({fmt_pct(cumulative_ret)})")
    lines.append("")
    lines.append(f"📦 目前曝險：約 {fmt_pct(exposure)}")
    lines.append(f"⚙️ 波動縮放：{details['vol_scale']:.2f}")
    lines.append(f"📏 最大權重變化：{fmt_pct(max_diff)}")
    lines.append("")

    if note:
        lines.append(f"⚠️ {note}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━")

    if include_daily_pl:
        lines.append("📌 目前持倉與今日損益")
    else:
        lines.append("📌 目前建立 / 新建立持倉")
        lines.append("今日不計算單檔 P/L，從下一個交易日開始追蹤。")

    lines.append("")

    for i, row in enumerate(rows, start=1):
        ticker = row["ticker"]

        lines.append(f"{i}. {ticker}")
        lines.append(f"比例：{fmt_pct(row['weight'])}")
        lines.append(f"金額：{fmt_money(row['value'])}")

        if ticker in ["CASH", "MARGIN"]:
            lines.append("股數：-")
            lines.append("價格：-")
        else:
            lines.append(f"股數：{row['shares']:.4f} 股")
            lines.append(f"價格：{fmt_money(row['price'])}")

        if include_daily_pl:
            dpl = row["daily_pl"]
            dret = row["daily_ret"]

            if dpl is None:
                lines.append("今日：無法計算")
            else:
                sign = "+" if dpl >= 0 else ""
                lines.append(f"今日：{sign}{fmt_money(dpl)} ({fmt_pct(dret)})")
        else:
            lines.append("今日：從下一個交易日開始計算")

        lines.append("")

    lines.append("━━━━━━━━━━━━━━")
    lines.append("註：本系統只通知，不會自動下單。")
    lines.append("註：這是虛擬策略追蹤，不是 Alpaca 真實成交損益。")
    lines.append("註：第一次建立配置不算 P/L；之後每天用已建立股數追蹤每日與累積損益。")

    return "\n".join(lines)


# =========================
# 主流程
# =========================

def main():
    bot_token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")

    configured_cash = float(
        get_env("TOTAL_CASH", required=False, default="10000")
    )

    close = fetch_alpaca_daily_bars(ALL_SYMBOLS)
    latest_date_obj = close.index[-1]
    latest_date = str(latest_date_obj)
    current_month = latest_date[:7]

    target_weights, details = compute_target_weights(close)

    state = load_state()

    # 如果 state.json 是舊版本，會自動重新建立
    if not is_valid_state(state):
        holdings, cash = build_positions_from_weights(
            target_weights,
            configured_cash,
            close
        )

        current_value = portfolio_value(holdings, cash, close)

        state = {
            "initial_cash": configured_cash,
            "start_date": latest_date,
            "last_date": latest_date,
            "last_value": current_value,
            "holdings": holdings,
            "cash": cash,
            "target_weights": target_weights,
            "last_checked_month": current_month,
            "last_rebalance_month": current_month,
            "updated_at_utc": datetime.now(timezone.utc).isoformat()
        }

        save_state(state)

        rows = build_position_rows(
            holdings=holdings,
            cash=cash,
            close=close,
            total_value=current_value,
            include_daily_pl=False
        )

        message = format_report(
            latest_date=latest_date,
            details=details,
            action_text="✅ 第一次建立配置，今天不計算 P/L",
            initial_cash=configured_cash,
            previous_value=None,
            current_value=current_value,
            daily_pl=None,
            cumulative_pl=0.0,
            max_diff=1.0,
            rows=rows,
            include_daily_pl=False,
            note="這是第一天建立虛擬持倉。明天開始才會用今天建立的股數計算每日與累積 P/L。"
        )

        send_telegram(bot_token, chat_id, message)
        print("First portfolio created.")
        return

    initial_cash = float(state["initial_cash"])
    old_holdings = {
        k: float(v)
        for k, v in state["holdings"].items()
    }
    old_cash = float(state["cash"])
    old_last_value = float(state["last_value"])
    old_last_date = str(state["last_date"])
    old_target_weights = state.get("target_weights", {})

    current_value_before_rebalance = portfolio_value(
        old_holdings,
        old_cash,
        close
    )

    same_data_date = latest_date == old_last_date

    if same_data_date:
        daily_pl = None
        cumulative_pl = current_value_before_rebalance - initial_cash

        rows = build_position_rows(
            holdings=old_holdings,
            cash=old_cash,
            close=close,
            total_value=current_value_before_rebalance,
            include_daily_pl=False
        )

        message = format_report(
            latest_date=latest_date,
            details=details,
            action_text="ℹ️ 資料日期沒有更新，同一天重跑，不重複計算 P/L",
            initial_cash=initial_cash,
            previous_value=None,
            current_value=current_value_before_rebalance,
            daily_pl=None,
            cumulative_pl=cumulative_pl,
            max_diff=0.0,
            rows=rows,
            include_daily_pl=False,
            note="今天已經建立 / 更新過，不會重複計算損益。"
        )

        send_telegram(bot_token, chat_id, message)
        print("Same date, no update.")
        return

    # 新交易日：先用舊持倉計算今天損益
    daily_pl = current_value_before_rebalance - old_last_value
    cumulative_pl = current_value_before_rebalance - initial_cash

    last_checked_month = state.get("last_checked_month")
    is_monthly_check = last_checked_month != current_month

    if is_monthly_check:
        max_diff = max_weight_change(target_weights, old_target_weights)
    else:
        max_diff = 0.0

    should_rebalance = is_monthly_check and max_diff >= CHANGE_THRESH

    if should_rebalance:
        # 用今日收盤價，把目前組合價值換成新策略持倉
        new_holdings, new_cash = build_positions_from_weights(
            target_weights,
            current_value_before_rebalance,
            close
        )

        current_value_after_rebalance = portfolio_value(
            new_holdings,
            new_cash,
            close
        )

        state = {
            "initial_cash": initial_cash,
            "start_date": state.get("start_date", latest_date),
            "last_date": latest_date,
            "last_value": current_value_after_rebalance,
            "holdings": new_holdings,
            "cash": new_cash,
            "target_weights": target_weights,
            "last_checked_month": current_month,
            "last_rebalance_month": current_month,
            "updated_at_utc": datetime.now(timezone.utc).isoformat()
        }

        save_state(state)

        rows = build_position_rows(
            holdings=new_holdings,
            cash=new_cash,
            close=close,
            total_value=current_value_after_rebalance,
            include_daily_pl=False
        )

        message = format_report(
            latest_date=latest_date,
            details=details,
            action_text="✅ 觸發換倉，已用今日收盤價建立新虛擬持倉",
            initial_cash=initial_cash,
            previous_value=old_last_value,
            current_value=current_value_after_rebalance,
            daily_pl=daily_pl,
            cumulative_pl=cumulative_pl,
            max_diff=max_diff,
            rows=rows,
            include_daily_pl=False,
            note="今日 P/L 是換倉前舊持倉的結果；新持倉從下一個交易日開始計算 P/L。"
        )

        send_telegram(bot_token, chat_id, message)
        print("Rebalanced.")
        return

    # 不換倉，沿用原本股數
    state["last_date"] = latest_date
    state["last_value"] = current_value_before_rebalance
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    if is_monthly_check:
        state["last_checked_month"] = current_month
        action_text = "⏸️ 本月檢查未達換倉門檻，沿用原持倉"
        note = "策略本月有檢查，但最大權重變化未達 20%，所以不換倉。"
    else:
        action_text = "⏸️ 持續追蹤，不換倉"
        note = ""

    save_state(state)

    rows = build_position_rows(
        holdings=old_holdings,
        cash=old_cash,
        close=close,
        total_value=current_value_before_rebalance,
        include_daily_pl=True
    )

    message = format_report(
        latest_date=latest_date,
        details=details,
        action_text=action_text,
        initial_cash=initial_cash,
        previous_value=old_last_value,
        current_value=current_value_before_rebalance,
        daily_pl=daily_pl,
        cumulative_pl=cumulative_pl,
        max_diff=max_diff,
        rows=rows,
        include_daily_pl=True,
        note=note
    )

    send_telegram(bot_token, chat_id, message)
    print("Daily tracking report sent.")


if __name__ == "__main__":
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    try:
        main()
    except Exception as e:
        error_message = f"""❌ 策略機器人執行失敗

原因：
{str(e)}

常見原因：
1. Alpaca API Key / Secret 錯誤或貼反
2. GitHub Secrets 沒設定 ALPACA_API_KEY / ALPACA_API_SECRET
3. requirements.txt 沒有 pandas 或 numpy
4. Alpaca 當下資料尚未更新
"""

        if bot_token and chat_id:
            try:
                send_telegram(bot_token, chat_id, error_message)
            except Exception:
                pass

        raise
