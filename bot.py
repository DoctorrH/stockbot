import asyncio
import os
import io
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from vnstock.api.quote import Quote

# --- CẤU HÌNH HỆ THỐNG ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf8', line_buffering=True)

VN100_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    "ANV", "ASM", "BAF", "BFC", "BMP", "BSI", "C4G", "CII", "CMG", "CSM",
    "D2D", "DCM", "DGC", "DHA", "DHC", "DIG", "DPM", "DPR", "DRC", "DXG",
    "EIB", "EVF", "FRT", "FTS", "GEE", "GEX", "GEG", "GMD", "HAG", "HAH",
    "HCM", "HHS", "HNG", "IDI", "IJC", "KBC", "KDC", "KDH", "LCG", "LPB",
    "MSB", "NKG", "NLG", "NT2", "OCB", "ORS", "PAN", "PDR", "PHR", "PNJ",
    "PTB", "PVD", "PVT", "QNS", "REE", "SCR", "SCS", "SIP", "SZC", "TCH",
    "TLG", "VCG", "VCI", "VDS", "VGC", "VHC", "VIX", "VPI", "VSH", "KSB",
]

# --- MODELS ---

@dataclass(frozen=True)
class SignalResult:
    symbol: str
    exchange: str
    close: float
    pct_change: float
    rsi14: float
    ma20: float
    ma50: float
    ma200: float
    ma20_distance_pct: float
    vol: float
    vol_avg20: float
    rvol: float
    special_label: str
    priority_level: int
    rs_score: float
    is_weekly_ok: bool
    w_weeks: int
    spread: float
    reason: str

@dataclass(frozen=True)
class EvalOutcome:
    symbol: str
    exchange: str
    info_line: str
    skip_reason: str
    signal: Optional[SignalResult]

# --- UTILS & DATA (COPIED FROM BACKTEST) ---

def log(level: str, message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns and "time" not in out.columns:
        out.rename(columns={"date": "time"}, inplace=True)
    out["time"] = pd.to_datetime(out["time"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["time", "close", "volume"]).sort_values("time").reset_index(drop=True)

def get_history(symbol: str, source: str, length: int) -> pd.DataFrame:
    q = Quote(symbol=symbol, source=source)
    now = datetime.now()
    start_date = (now - timedelta(days=int(length) * 1.6)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    try:
        df = q.history(start=start_date, end=end_date, interval="1D")
        return normalize_ohlcv(df)
    except: return pd.DataFrame()

def load_history_with_fallback(symbol: str, sources: List[str], length: int) -> Tuple[pd.DataFrame, Optional[str]]:
    for src in sources:
        df = get_history(symbol, src, length)
        if not df.empty: return df, src
    return pd.DataFrame(), None

def check_weekly_status(df: pd.DataFrame, target_date: Optional[datetime] = None) -> Tuple[bool, int]:
    if len(df) < 150: return False, 0
    df_weekly = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    if len(df_weekly) < 20: return False, 0
    df_weekly["ma20_w"] = df_weekly["close"].rolling(window=20).mean()
    df_weekly = df_weekly.dropna(subset=["ma20_w"])
    if df_weekly.empty: return False, 0
    last_row = df_weekly.iloc[-1]
    is_above = last_row["close"] > last_row["ma20_w"]
    consecutive = 0
    for i in range(len(df_weekly)-1, -1, -1):
        if df_weekly["close"].iloc[i] > df_weekly["ma20_w"].iloc[i]: consecutive += 1
        else: break
    return is_above, consecutive

def calculate_rs_score(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> float:
    if stock_df.empty or index_df.empty or len(stock_df) < 50 or len(index_df) < 50: return 0.0
    s_now, s_50 = stock_df["close"].iloc[-1], stock_df["close"].iloc[-50]
    i_now, i_50 = index_df["close"].iloc[-1], index_df["close"].iloc[-50]
    if s_50 == 0 or i_50 == 0: return 0.0
    return (s_now / s_50) / (i_now / i_50)

# --- MARKET PROTECTIONS ---

async def check_market_kill_switch(sources: List[str], tickers: List[str]) -> Tuple[bool, str]:
    log("INFO", "Kiểm tra Market Kill Switch...")
    idx_df, _ = await asyncio.to_thread(load_history_with_fallback, "VNINDEX", sources, 50)
    if not idx_df.empty and len(idx_df) >= 2:
        idx_df["rsi"] = rsi(idx_df["close"], 14)
        last, prev = idx_df.iloc[-1], idx_df.iloc[-2]
        pct = (last["close"]/prev["close"] - 1)*100
        rsi_drop = prev["rsi"] - last["rsi"]
        if pct < -2.0 or rsi_drop > 5.0:
            return True, f"🚨 VNINDEX giảm {pct:.2f}% | RSI rơi {rsi_drop:.2f}đ"
    return False, ""

# --- CORE EVALUATION (COPIED LOGIC FROM BACKTEST) ---

def evaluate_symbol(symbol: str, exchange: str, sources: List[str], length: int = 220, **kwargs) -> EvalOutcome:
    df, used_source = load_history_with_fallback(symbol, sources, length)
    if not used_source: return EvalOutcome(symbol, exchange, "", "không lấy được dữ liệu", None)
    
    df["ma20"] = sma(df["close"], 20)
    df["ma50"] = sma(df["close"], 50)
    df["ma200"] = sma(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    df["vol_avg20"] = df["volume"].shift(1).rolling(window=20).mean()
    df["perf"] = df["close"].pct_change() * 100
    
    if len(df) < 50 or not np.isfinite(df["ma20"].iloc[-1]):
        return EvalOutcome(symbol, exchange, "", "thiếu dữ liệu kỹ thuật", None)
        
    last, prev = df.iloc[-1], df.iloc[-2]
    
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    pc = float(prev["close"])
    perf_today = float(last["perf"])
    vol_avg20 = float(last["vol_avg20"])
    rvol = float(last["volume"] / vol_avg20) if vol_avg20 > 0 else 0
    rs_score = kwargs.get("rs_score", 0.0)
    is_weekly_ok, w_weeks = check_weekly_status(df)
    candle_spread = (h - l) / pc * 100 if pc > 0 else 0
    ma20_distance_pct = (c / last["ma20"] - 1) * 100
    
    info_line = f"{symbol}: Giá={c:.2f}, RS={rs_score:.2f}, RVOL={rvol:.2f}"

    label = ""
    priority = 4

    if is_weekly_ok:
        # 1. RS THẤP
        if rs_score < 1.15:
            if perf_today > 3.0: label, priority = "⚠️ NỔ GIẢ (RS THẤP)", 3
            else: label, priority = "💤 CHỜ DÒNG TIỀN", 4
        
        # 2. RŨ BỎ (SHAKEOUT)
        elif rs_score > 1.3 and perf_today < 0:
            if perf_today < -2.5 and rvol > 0.8: label, priority = "👀 THEO DÕI THÊM", 4
            elif rvol < 0.8: label, priority = "💎 RŨ BỎ CHUẨN (MUA GOM)", 1
            elif -2.0 < perf_today < 0 and 0.8 <= rvol < 1.1: label, priority = "🔥 RŨ BỎ LINH HOẠT", 1
            else: label, priority = "💎 RŨ BỎ KỸ THUẬT", 2

        # 3. SIÊU CỔ ĐANG CHẠY
        elif rs_score > 1.5:
            if perf_today >= -2.0: label, priority = "🔥 SIÊU CỔ ĐANG CHẠY", 1
        
        # 4. ĐIỂM NỔ & CẠN CUNG
        elif 1.25 <= rs_score <= 1.5:
            if perf_today > 2.0 and rvol > 1.5: label, priority = "🚀 XÁC NHẬN ĐIỂM NỔ", 1
            elif perf_today > 2.0 and rvol < 1.0: label, priority = "🚀 CẠN CUNG BỨT PHÁ", 1 if rs_score >= 1.35 else 3
            elif abs(perf_today) < 1.0 and rvol < 0.8: label, priority = "💤 TÍCH LŨY KIỆT VOL", 2
        
        # 5. NỀN DÀI & DÒNG TIỀN ĐỘT BIẾN
        elif 1.2 <= rs_score < 1.25 and w_weeks >= 4 and rvol < 0.8: label, priority = "💎 GOM HÀNG NỀN DÀI", 2
        elif 1.15 <= rs_score < 1.25 and rvol > 2.5: label, priority = "🚀 DÒNG TIỀN ĐỘT BIẾN (HẠNG 2)", 2

        # CHẶN BẪY
        if rvol > 5.0: label, priority = "⚠️ CAO TRÀO MUA (RỦI RO)", 3
        elif candle_spread > 8.0 and rvol > 1.8: label, priority = "⚠️ BIẾN ĐỘNG LỎNG", 3
        elif perf_today > 2.0 and rvol < 0.7 and rs_score < 1.3: label, priority = "⚠️ TĂNG THIẾU VOL", 3

    # CẢNH BÁO QUÁ MUA
    if rs_score > 2.0 and ma20_distance_pct > 20.0: label, priority = "⚠️ QUÁ MUA (KHÔNG ĐU)", 3
    elif ma20_distance_pct > 15.0 and rs_score <= 1.5: label, priority = "⚠️ QUÁ ĐIỂM MUA", 3

    if not label: label, priority = "👀 THEO DÕI THÊM", 4
    
    sig = SignalResult(
        symbol=symbol, exchange=exchange, close=c, pct_change=perf_today,
        rsi14=last["rsi14"], ma20=last["ma20"], ma50=last["ma50"], ma200=df["ma200"].iloc[-1],
        ma20_distance_pct=ma20_distance_pct, vol=last["volume"], vol_avg20=vol_avg20,
        rvol=rvol, special_label=label, priority_level=priority,
        rs_score=rs_score, is_weekly_ok=is_weekly_ok, w_weeks=w_weeks, spread=candle_spread,
        reason=f"P{priority}"
    )
    
    return EvalOutcome(symbol, exchange, info_line, "", sig if priority <= 3 else None)

# --- BOT INTERFACE ---

def format_telegram_message(results: List[SignalResult], scanned: int, source: str) -> str:
    header = f"<b>🚀 STOCK BOT V10 - SCAN {datetime.now().strftime('%d/%m %H:%M')}</b>\n"
    header += f"<i>Universe: VN100 | Quét: {scanned} mã | Nguồn: {source}</i>\n\n"
    
    lines = [header]
    for r in sorted(results, key=lambda x: (x.priority_level, -x.rs_score)):
        emoji = "🚀" if r.priority_level == 1 else ("💎" if r.priority_level == 2 else "⚠️")
        msg = (
            f"{emoji} <b>{r.symbol}</b> | {r.special_label}\n"
            f"───────────────────\n"
            f"💰 Giá: <b>{r.close:,.2f}</b> ({r.pct_change:+.2f}%)\n"
            f"📊 RS: <b>{r.rs_score:.2f}</b> | RVOL: <b>{r.rvol:.2f}</b>\n"
            f"📏 Spread: <b>{r.spread:.2f}%</b> | MA20: <b>{r.ma20:,.2f}</b>\n"
            f"📍 Cách MA20: <b>{r.ma20_distance_pct:+.2f}%</b> | Tuần: {'✅' if r.is_weekly_ok else '❌'} ({r.w_weeks}w)\n"
            f"───────────────────\n"
        )
        lines.append(msg)
    return "".join(lines)

async def scan_once_and_send():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    sources = ["KBS", "VCI", "TCBS", "SSI"]
    
    log("INFO", "Bắt đầu quét V10 (Logic 1:1 từ Backtest)...")
    
    # 1. Kill Switch
    is_killed, kill_msg = await check_market_kill_switch(sources, VN100_TICKERS)
    if is_killed:
        log("KILL", kill_msg)
        await send_telegram_message(token, chat_id, f"⚠️ <b>DỪNG QUÉT KHẨN CẤP</b>\n\n{kill_msg}")
        return

    # 2. RS Ranking (220 phiên)
    idx_df, _ = await asyncio.to_thread(load_history_with_fallback, "VNINDEX", sources, 220)
    rs_results = []
    for t in VN100_TICKERS:
        df_t, _ = await asyncio.to_thread(load_history_with_fallback, t, sources, 220)
        rs_results.append((t, calculate_rs_score(df_t, idx_df)))
        await asyncio.sleep(0.4)
    
    rs_results.sort(key=lambda x: x[1], reverse=True)
    top_20_count = int(len(VN100_TICKERS) * 0.2)
    top_20_tickers = {x[0] for x in rs_results[:top_20_count]}
    rs_map = {x[0]: x[1] for x in rs_results}
    
    results = []
    for t in VN100_TICKERS:
        if t not in top_20_tickers: continue
        log("SCAN", f"Đang phân tích {t}...")
        outcome = await asyncio.to_thread(evaluate_symbol, t, "VN100", sources, rs_score=rs_map.get(t,0))
        if outcome.signal: results.append(outcome.signal)
        await asyncio.sleep(1.2)
        
    if results:
        msg = format_telegram_message(results, len(VN100_TICKERS), "VN100")
        await send_telegram_message(token, chat_id, msg)
        log("INFO", f"Gửi {len(results)} tín hiệu thành công.")
    else:
        log("INFO", "Không tìm thấy mã đạt tiêu chuẩn.")

async def send_telegram_message(token, chat_id, text):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=15)
    except: pass

async def main():
    await scan_once_and_send()

if __name__ == "__main__":
    asyncio.run(main())
