import asyncio
import os
import io
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

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
    rvol_label: str
    filter_label: str
    special_label: str
    priority_level: int
    rs_score: float
    is_weekly_ok: bool
    w_weeks: int
    spread: float
    warning: str
    reason: str

@dataclass(frozen=True)
class EvalOutcome:
    symbol: str
    exchange: str
    info_line: str
    skip_reason: str
    signal: Optional[SignalResult]

# --- UTILS & DATA ---

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

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = sma(out["close"], 20)
    out["ma50"] = sma(out["close"], 50)
    out["ma200"] = sma(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)
    out["vol_avg20_prev"] = out["volume"].shift(1).rolling(window=20, min_periods=20).mean()
    out["perf"] = out["close"].pct_change() * 100
    return out

def check_weekly_status(df: pd.DataFrame) -> Tuple[bool, int]:
    if len(df) < 150: return False, 0
    df_weekly = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    if len(df_weekly) < 20: return False, 0
    df_weekly["ma20_w"] = df_weekly["close"].rolling(window=20).mean()
    df_weekly = df_weekly.dropna(subset=["ma20_w"])
    if df_weekly.empty: return False, 0
    is_above = df_weekly["close"].iloc[-1] > df_weekly["ma20_w"].iloc[-1]
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

async def check_market_kill_switch(sources: List[str], tickers: List[str], timeout: int) -> Tuple[bool, str]:
    log("INFO", "Kiểm tra Market Kill Switch...")
    # 1. VNINDEX Check
    idx_df, _ = await asyncio.to_thread(load_history_with_fallback, "VNINDEX", sources, 50)
    if not idx_df.empty and len(idx_df) >= 2:
        idx_df["rsi"] = rsi(idx_df["close"], 14)
        last, prev = idx_df.iloc[-1], idx_df.iloc[-2]
        pct = (last["close"]/prev["close"] - 1)*100
        rsi_drop = prev["rsi"] - last["rsi"]
        if pct < -2.0 or rsi_drop > 5.0:
            return True, f"🚨 THỊ TRƯỜNG NGUY HIỂM: VNINDEX giảm {pct:.2f}% | RSI rơi {rsi_drop:.2f}đ"
            
    # 2. Breadth Check (Giảm sàn diện rộng)
    log("INFO", "Kiểm tra độ rộng thị trường...")
    floor_count = 0
    for sym in tickers[:30]: # Quét nhanh 30 mã đại diện
        df, _ = await asyncio.to_thread(load_history_with_fallback, sym, sources, 2)
        if not df.empty and len(df) >= 2:
            pct = (df["close"].iloc[-1]/df["close"].iloc[-2] - 1)*100
            if pct <= -6.8: floor_count += 1
    
    if floor_count >= 3:
        return True, f"💀 CẢNH BÁO SẬP DIỆN RỘNG: Phát hiện {floor_count} mã giảm sàn trong nhóm dẫn dắt."
        
    return False, ""

# --- CORE EVALUATION V10 ---

def evaluate_symbol(symbol: str, exchange: str, sources: List[str], length: int = 260, **kwargs) -> EvalOutcome:
    df, used_source = load_history_with_fallback(symbol, sources, length)
    if not used_source: return EvalOutcome(symbol, exchange, "", "không lấy được dữ liệu", None)
    if len(df) < 220: return EvalOutcome(symbol, exchange, "", "thiếu dữ liệu MA200", None)
    
    df = calc_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    
    vol_avg20 = float(last["vol_avg20_prev"])
    if vol_avg20 <= 200_000: return EvalOutcome(symbol, exchange, "", f"thanh khoản thấp ({vol_avg20:.0f})", None)
    
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    pc = float(prev["close"])
    perf_today = last["perf"]
    rvol = float(last["volume"] / vol_avg20) if vol_avg20 > 0 else 0
    rs_score = kwargs.get("rs_score", 0.0)
    is_weekly_ok, w_weeks = check_weekly_status(df)
    candle_spread = (h - l) / pc * 100 if pc > 0 else 0
    ma20_distance_pct = (c / last["ma20"] - 1) * 100
    
    info_line = f"{symbol}: Giá={c:.2f}, RS={rs_score:.2f}, RVOL={rvol:.2f}, Perf={perf_today:.2f}%"

    priority_level = 4
    priority_label = ""

    if is_weekly_ok:
        if rs_score < 1.15:
            if perf_today > 3.0: priority_label, priority_level = "⚠️ NỔ GIẢ (RS THẤP)", 3
            else: priority_label, priority_level = "💤 CHỜ DÒNG TIỀN", 4
        elif rs_score > 1.3 and perf_today < 0:
            if perf_today < -2.5 and rvol > 0.8: priority_label, priority_level = "👀 THEO DÕI THÊM", 4
            elif rvol < 0.8: priority_label, priority_level = "💎 RŨ BỎ CHUẨN (MUA GOM)", 1
            elif -2.0 < perf_today < 0 and 0.8 <= rvol < 1.1: priority_label, priority_level = "🔥 RŨ BỎ LINH HOẠT", 1
        elif perf_today > 2.0 and rvol > 1.5:
            if rs_score > 1.5: priority_label, priority_level = "🚀 SIÊU CỔ XÁC NHẬN NỔ", 1
            else: priority_label, priority_level = "🚀 XÁC NHẬN ĐIỂM NỔ", 2
        elif abs(perf_today) < 1.0 and rvol < 0.8:
            if rs_score >= 1.35: priority_label, priority_level = "🚀 CẠN CUNG BỨT PHÁ", 1
            else: priority_label, priority_level = "🚀 CẠN CUNG BỨT PHÁ", 3
        elif abs(perf_today) < 1.0 and rvol < 0.6:
            priority_label, priority_level = "💤 TÍCH LŨY KIỆT VOL", 2
        elif 1.2 <= rs_score < 1.25 and w_weeks >= 4 and rvol < 0.8:
            priority_label, priority_level = "💎 GOM HÀNG NỀN DÀI", 2
        elif 1.15 <= rs_score < 1.25 and rvol > 2.5:
            priority_label, priority_level = "🚀 DÒNG TIỀN ĐỘT BIẾN (HẠNG 2)", 2

    # CHẶN BẪY
    if rvol > 5.0: priority_label, priority_level = "⚠️ CAO TRÀO MUA (RỦI RO)", 3
    elif candle_spread > 8.0 and rvol > 1.8: priority_label, priority_level = "⚠️ BIẾN ĐỘNG LỎNG (RỦI RO)", 3
    elif rs_score > 2.0 and ma20_distance_pct > 20.0: priority_label, priority_level = "⚠️ QUÁ MUA (KHÔNG ĐU)", 3
    elif abs(ma20_distance_pct) > 15.0 and rs_score <= 1.5: priority_label, priority_level = "⚠️ QUÁ ĐIỂM MUA", 3

    if not priority_label: priority_label, priority_level = "👀 THEO DÕI THÊM", 4
    
    reason = f"P{priority_level} | RS={rs_score:.2f} | W={is_weekly_ok} | RVOL={rvol:.2f}"
    
    sig = SignalResult(
        symbol=symbol, exchange=exchange, close=c, pct_change=perf_today,
        rsi14=last["rsi14"], ma20=last["ma20"], ma50=last["ma50"], ma200=last["ma200"],
        ma20_distance_pct=ma20_distance_pct, vol=last["volume"], vol_avg20=vol_avg20,
        rvol=rvol, rvol_label="", filter_label="", special_label=priority_label,
        priority_level=priority_level, rs_score=rs_score, is_weekly_ok=is_weekly_ok,
        w_weeks=w_weeks, spread=candle_spread, warning="", reason=reason
    )
    
    return EvalOutcome(symbol, exchange, info_line, "", sig if priority_level <= 2 else None)

# --- BOT INTERFACE ---

def format_telegram_message(results: List[SignalResult], scanned: int, source: str) -> str:
    header = f"<b>🚀 STOCK BOT V10 - SCAN {datetime.now().strftime('%d/%m %H:%M')}</b>\n"
    header += f"<i>Universe: VN100 | Quét: {scanned} mã | Nguồn: {source}</i>\n\n"
    
    lines = [header]
    for r in sorted(results, key=lambda x: x.priority_level):
        emoji = "🚀" if r.priority_level == 1 else "💎"
        msg = (
            f"{emoji} <b>{r.symbol}</b> | {r.special_label}\n"
            f"───────────────────\n"
            f"💰 Giá: <b>{r.close:,.2f}</b> ({r.pct_change:+.2f}%)\n"
            f"📊 RS: <b>{r.rs_score:.2f}</b> | RVOL: <b>{r.rvol:.2f}</b>\n"
            f"📏 Spread: <b>{r.spread:.2f}%</b> | MA20: <b>{r.ma20:.2f}</b>\n"
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
    
    log("INFO", "Bắt đầu quét V10 chuyên sâu...")
    
    # 1. Kill Switch Check
    is_killed, kill_msg = await check_market_kill_switch(sources, VN100_TICKERS, 30)
    if is_killed:
        log("KILL", kill_msg)
        await send_telegram_message(token, chat_id, f"⚠️ <b>DỪNG QUÉT KHẨN CẤP</b>\n\n{kill_msg}")
        return

    # 2. RS Ranking
    index_df, _ = await asyncio.to_thread(load_history_with_fallback, "VNINDEX", sources, 60)
    rs_results = []
    for t in VN100_TICKERS:
        df_t, _ = await asyncio.to_thread(load_history_with_fallback, t, sources, 60)
        rs_results.append((t, calculate_rs_score(df_t, index_df)))
        await asyncio.sleep(0.4)
    
    rs_results.sort(key=lambda x: x[1], reverse=True)
    top_20_tickers = {x[0] for x in rs_results[:20]}
    rs_map = {x[0]: x[1] for x in rs_results}
    
    results = []
    scanned_count = 0
    for t in VN100_TICKERS:
        if t not in top_20_tickers: continue
        scanned_count += 1
        log("SCAN", f"Đang phân tích {t}...")
        outcome = await asyncio.to_thread(evaluate_symbol, t, "VN100", sources, rs_score=rs_map.get(t,0))
        if outcome.signal: results.append(outcome.signal)
        await asyncio.sleep(1.2)
        
    if results:
        msg = format_telegram_message(results, len(VN100_TICKERS), ",".join(sources))
        await send_telegram_message(token, chat_id, msg)
        log("INFO", f"Gửi {len(results)} tín hiệu thành công.")
    else:
        log("INFO", "Không tìm thấy mã đạt tiêu chuẩn V10.")

async def send_telegram_message(token, chat_id, text):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=15)
    except Exception as e: log("ERROR", f"Lỗi gửi Telegram: {e}")

# --- BOT HANDLERS ---

async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot V10 đang hoạt động. Sử dụng /scan để quét ngay.")

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Đang bắt đầu quét VN100 chuẩn V10. Vui lòng đợi...")
    await scan_once_and_send()

# --- MAIN ENTRY ---

async def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    scan_once = os.getenv("SCAN_ONCE", "0") == "1"

    if scan_once:
        await scan_once_and_send()
        return

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("test", cmd_test))
    app.add_handler(CommandHandler("scan", cmd_scan))
    
    log("INFO", "Bot V10 đã sẵn sàng (Long-running mode).")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    
    while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
