import asyncio
import os
import io
import sys
import html
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

SECTORS = {
    "Ngân hàng": ["ACB", "BID", "CTG", "HDB", "MBB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "EIB", "MSB", "OCB", "LPB"],
    "Chứng khoán": ["SSI", "VCI", "VND", "HCM", "VIX", "FTS", "BSI", "ORS", "VDS"],
    "Thép": ["HPG", "HSG", "NKG", "SMC", "TLH"],
    "Bất động sản": ["VIC", "VHM", "VRE", "NVL", "PDR", "DIG", "DXG", "KBC", "KDH", "NLG", "SCR", "IJC", "TCH"],
    "Bán lẻ": ["MWG", "PNJ", "FRT", "DGW", "PET"],
    "Công nghệ & Viễn thông": ["FPT", "CMG", "LCG", "CTR", "VGI"],
    "Dầu khí": ["GAS", "PLX", "POW", "PVD", "PVS", "PVT", "BSR"],
    "Hóa chất & Phân bón": ["DGC", "DCM", "DPM", "BFC", "CSV"],
}

STRATEGY_MAP = {
    "💎 RŨ BỎ CHUẨN (MUA GOM)": "Mua gom 30-50% vị thế. Cắt lỗ nếu đóng cửa thủng MA50.",
    "🔥 RŨ BỎ LINH HOẠT": "Mua test 30% vị thế quanh nền. Hàng về lỗ > 4% cắt dứt khoát.",
    "💎 RŨ BỎ KỸ THUẬT": "Mua thăm dò tỷ trọng nhỏ. Đợi Vol nổ để gia tăng.",
    "🔥 SIÊU CỔ ĐANG CHẠY": "Nắm giữ chặt. Trailing stop (chặn lãi) theo MA10 hoặc đáy nến tuần.",
    "🚀 XÁC NHẬN ĐIỂM NỔ": "Mua đủ vị thế (Full size). Cắt lỗ khi thủng nửa cây nến bùng nổ.",
    "🚀 CẠN CUNG BỨT PHÁ": "Mua 50% gia tăng. Đợi dòng tiền lớn xác nhận vượt đỉnh.",
    "💤 TÍCH LŨY KIỆT VOL": "Nằm vùng 20-30% vốn. Tuyệt đối không mua đuổi giá xanh.",
    "💎 GOM HÀNG NỀN DÀI": "Gom dần từng phần theo biên dưới của nền. Kiên nhẫn nắm giữ.",
    "🚀 DÒNG TIỀN ĐỘT BIẾN (HẠNG 2)": "Đánh T+ tỷ trọng vừa phải. Chốt lời chủ động khi rướn giá.",
    "⚠️ MẤT GIA TỐC TĂNG": "Dừng mua mới. Sẵn sàng chốt lời 1/2 nếu thủng MA20.",
    "👀 THỊ TRƯỜNG XẤU (ĐỨNG NGOÀI)": "Rủi ro hệ thống. Ôm tiền mặt, không bắt dao rơi."
}

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
    trend_type: str
    sector_name: str
    recommended_size: str
    stop_loss_price: float
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

def calculate_rs_score(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> Tuple[float, float]:
    if stock_df.empty or index_df.empty or len(stock_df) < 55 or len(index_df) < 55: 
        return 0.0, 0.0
    
    # RS Today (Tỷ lệ thay đổi 50 phiên của Cổ phiếu / Chỉ số)
    s_now, s_50 = stock_df["close"].iloc[-1], stock_df["close"].iloc[-50]
    i_now, i_50 = index_df["close"].iloc[-1], index_df["close"].iloc[-50]
    rs_today = (s_now / s_50) / (i_now / i_50) if (s_50 != 0 and i_50 != 0) else 0.0
    
    # RS 5 days ago (Tỷ lệ thay đổi 50 phiên kết thúc cách đây 5 ngày)
    s_5, s_55 = stock_df["close"].iloc[-6], stock_df["close"].iloc[-55]
    i_5, i_55 = index_df["close"].iloc[-6], index_df["close"].iloc[-55]
    rs_5_ago = (s_5 / s_55) / (i_5 / i_55) if (s_55 != 0 and i_55 != 0) else 0.0
    
    rs_momentum = rs_today - rs_5_ago
    return rs_today, rs_momentum

def calculate_sector_rs(rs_map: dict) -> dict:
    sector_scores = {}
    for sector, tickers in SECTORS.items():
        # rs_map lưu dạng {ticker: (rs_score, rs_momentum)}
        scores = [rs_map[t][0] for t in tickers if t in rs_map]
        if scores:
            sector_scores[sector] = sum(scores) / len(scores)
        else:
            sector_scores[sector] = 1.0
    return sector_scores

# --- MARKET PROTECTIONS ---

async def check_market_kill_switch(sources: List[str], tickers: List[str]) -> Tuple[bool, str, bool]:
    log("INFO", "Kiểm tra trạng thái thị trường VNINDEX...")
    idx_df, _ = await asyncio.to_thread(load_history_with_fallback, "VNINDEX", sources, 50)
    if idx_df.empty or len(idx_df) < 20:
        return False, "Không lấy được dữ liệu VNINDEX", False
        
    idx_df["ma20"] = sma(idx_df["close"], 20)
    idx_df["rsi"] = rsi(idx_df["close"], 14)
    last, prev = idx_df.iloc[-1], idx_df.iloc[-2]
    
    pct = (last["close"]/prev["close"] - 1)*100
    rsi_drop = prev["rsi"] - last["rsi"]
    is_weak = last["close"] < last["ma20"]
    
    # Tạo nội dung thông báo tình hình VNINDEX
    status_msg = f"VNINDEX: {last['close']:,.2f} ({pct:+.2f}%) | RSI: {last['rsi']:.1f} | MA20: {last['ma20']:,.2f}"
    if is_weak:
        status_msg += "\n⚠️ <b>CẢNH BÁO:</b> Thị trường yếu (Dưới MA20). Ưu tiên hạ tỷ trọng giải ngân!"
    else:
        status_msg += "\n✅ <b>Trạng thái:</b> Thị trường khỏe (Trên MA20)."

    # Logic Kill Switch (Dừng khẩn cấp)
    if pct < -2.0 or rsi_drop > 5.0:
        kill_reason = f"🚨 VNINDEX GIẢM MẠNH ({pct:.2f}%) hoặc RSI RƠI SỐC ({rsi_drop:.2f}đ)."
        return True, kill_reason, is_weak
        
    return False, status_msg, is_weak

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
    
    # Các tham số nâng cấp
    rs_score = kwargs.get("rs_score", 0.0)
    rs_momentum = kwargs.get("rs_momentum", 0.0)
    is_market_downtrend = kwargs.get("is_market_downtrend", False)
    
    # Logic Sóng ngành & Tỷ trọng
    sector_name = "KHÁC"
    for s, t_list in SECTORS.items():
        if symbol in t_list:
            sector_name = s
            break
    
    sector_rs = kwargs.get("sector_rs_map", {}).get(sector_name, 1.0)
    
    is_weekly_ok, w_weeks = check_weekly_status(df)
    candle_spread = (h - l) / pc * 100 if pc > 0 else 0
    ma20_distance_pct = (c / last["ma20"] - 1) * 100
    
    info_line = f"{symbol}: Giá={c:.2f}, RS={rs_score:.2f}, Momentum={rs_momentum:.3f}"

    label = ""
    priority = 4

    if is_weekly_ok:
        # 1. RS THẤP
        if rs_score < 1.15:
            if perf_today > 3.0: label, priority = "⚠️ NỔ GIẢ (RS THẤP)", 3
            else: label, priority = "💤 CHỜ DÒNG TIỀN", 4
        
        # 2. RŨ BỎ (SHAKEOUT) - Bổ sung MA50
        elif rs_score > 1.3 and perf_today < 0:
            # Giá phải nằm trên MA50 mới được tính là rũ bỏ an toàn
            if c < last["ma50"]: 
                label, priority = "👀 GÃY NỀN TRUNG HẠN", 4
            elif perf_today < -2.5 and rvol > 0.8: 
                label, priority = "👀 THEO DÕI THÊM", 4
            elif rvol < 0.8: 
                label, priority = "💎 RŨ BỎ CHUẨN (MUA GOM)", 1
            elif -2.0 < perf_today < 0 and 0.8 <= rvol < 1.1: 
                label, priority = "🔥 RŨ BỎ LINH HOẠT", 1
            else: 
                label, priority = "💎 RŨ BỎ KỸ THUẬT", 2

        # 3. SIÊU CỔ ĐANG CHẠY
        elif rs_score > 1.5:
            if perf_today >= -2.0: 
                label, priority = "🔥 SIÊU CỔ ĐANG CHẠY", 1
                if rs_momentum <= 0: label, priority = "⚠️ MẤT GIA TỐC TĂNG", 3
        
        # 4. ĐIỂM NỔ & CẠN CUNG
        elif 1.25 <= rs_score <= 1.5:
            if perf_today > 2.0 and rvol > 1.5: 
                label, priority = "🚀 XÁC NHẬN ĐIỂM NỔ", 1
                if rs_momentum <= 0: label, priority = "⚠️ MẤT GIA TỐC TĂNG", 3
            elif perf_today > 2.0 and rvol < 1.0: 
                label, priority = "🚀 CẠN CUNG BỨT PHÁ", 1 if rs_score >= 1.35 else 3
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

    # BẢO VỆ RỦI RO HỆ THỐNG: Thị trường xấu (Downtrend MA20)
    if is_market_downtrend and priority == 1:
        # Ngoại lệ: Chỉ giữ lại Priority 1 nếu là RŨ BỎ CHUẨN và giá vẫn > MA50
        if label == "💎 RŨ BỎ CHUẨN (MUA GOM)" and c > last["ma50"]:
            pass # Giữ nguyên Priority 1
        else:
            label, priority = "👀 THỊ TRƯỜNG XẤU (ĐỨNG NGOÀI)", 4

    # QUẢN TRỊ TỶ TRỌNG VỐN (Recommended Size)
    recommended_size = "Quan sát (0%)"
    if priority in [1, 2]:
        if sector_rs >= 1.05:
            recommended_size = "ĐÁNH LỚN (30-50%) - Có sóng ngành bảo kê"
        else:
            recommended_size = "ĐÁNH NHỎ (Dưới 15%) - Đi ngược bầy đàn, rủi ro T+"

    # TÍNH GIÁ CẮT LỖ (Stop Loss)
    sl_price = 0.0
    if label == "💎 RŨ BỎ CHUẨN (MUA GOM)":
        sl_price = last["ma50"]
    elif label == "🚀 XÁC NHẬN ĐIỂM NỔ":
        sl_price = c - (h - l) * 0.5
    elif label == "🔥 RŨ BỎ LINH HOẠT":
        sl_price = c * 0.96
    elif priority <= 2:
        sl_price = c * 0.93

    if not label: label, priority = "👀 THEO DÕI THÊM", 4
    
    # PHÂN LOẠI CẤU TRÚC XU HƯỚNG (Ngắn hạn vs Trung/Dài hạn)
    trend_type = "📉 CHƯA RÕ XU HƯỚNG"
    m20 = last["ma20"]
    m50 = last["ma50"]
    m200 = df["ma200"].iloc[-1]
    
    if c > m20:
        if m20 > m50 > m200 and w_weeks >= 4:
            trend_type = "📈 FORM TRUNG DÀI HẠN"
        else:
            trend_type = "⚡ FORM ĐÁNH NGẮN (T+)"

    sig = SignalResult(
        symbol=symbol, exchange=exchange, close=c, pct_change=perf_today,
        rsi14=last["rsi14"], ma20=last["ma20"], ma50=last["ma50"], ma200=df["ma200"].iloc[-1],
        ma20_distance_pct=ma20_distance_pct, vol=last["volume"], vol_avg20=vol_avg20,
        rvol=rvol, special_label=label, priority_level=priority,
        rs_score=rs_score, is_weekly_ok=is_weekly_ok, w_weeks=w_weeks, spread=candle_spread,
        trend_type=trend_type,
        sector_name=sector_name,
        recommended_size=recommended_size,
        stop_loss_price=sl_price,
        reason=f"P{priority}"
    )
    
    return EvalOutcome(symbol, exchange, info_line, "", sig if priority <= 3 else None)

# --- BOT INTERFACE ---

def format_telegram_message(results: List[SignalResult], scanned: int, source: str, market_info: str = "", top_sectors_info: str = "") -> str:
    header = f"<b>🚀 STOCK BOT V10 - SCAN {datetime.now().strftime('%d/%m %H:%M')}</b>\n"
    header += f"<i>Universe: VN100 | Quét: {scanned} mã | Nguồn: {source}</i>\n"
    if market_info:
        header += f"───────────────────\n📊 <b>Tình hình VN-Index:</b>\n{market_info}\n"
    if top_sectors_info:
        header += f"───────────────────\n🔥 <b>TOP 3 NGÀNH DẪN DẮT:</b>\n{top_sectors_info}\n"
    header += "───────────────────\n\n"
    
    lines = [header]
    if not results:
        lines.append("📭 <i>Không tìm thấy mã nào đạt tiêu chuẩn trong phiên này.</i>")
    else:
        for r in sorted(results, key=lambda x: (x.priority_level, -x.rs_score)):
            emoji = "🚀" if r.priority_level == 1 else ("💎" if r.priority_level == 2 else "⚠️")
            strategy = STRATEGY_MAP.get(r.special_label, "Quan sát rủi ro, không mở vị thế mua mới.")
            
            # Thoát ký tự HTML cho các chuỗi văn bản để tránh lỗi parse Telegram
            safe_label = html.escape(r.special_label)
            safe_trend = html.escape(r.trend_type)
            safe_size = html.escape(r.recommended_size)
            safe_strategy = html.escape(strategy)
            
            sl_val = f"{r.stop_loss_price:,.2f}" if r.stop_loss_price > 0 else "Theo cấu trúc"
            
            msg = (
                f"{emoji} <b>{r.symbol}</b> ({html.escape(r.sector_name)}) | {safe_label}\n"
                f"───────────────────\n"
                f"💰 Giá: <b>{r.close:,.2f}</b> ({r.pct_change:+.2f}%)\n"
                f"📊 RS Mã: <b>{r.rs_score:.2f}</b> | RVOL: <b>{r.rvol:.2f}</b>\n"
                f"📍 Cách MA20: <b>{r.ma20_distance_pct:+.2f}%</b> | Cấu trúc: <b>{safe_trend}</b>\n"
                f"⚖️ Tỷ trọng: <b>{safe_size}</b>\n"
                f"💡 Hành động: <b>{safe_strategy}</b>\n"
                f"🛡️ Cắt lỗ tại: <b>{sl_val}</b>\n"
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
    
    # 1. Kiểm tra trạng thái thị trường & Kill Switch (Đèn giao thông)
    is_killed, market_msg, is_market_downtrend = await check_market_kill_switch(sources, VN100_TICKERS)
    if is_killed:
        log("KILL", market_msg)
        await send_telegram_message(token, chat_id, f"⚠️ <b>DỪNG QUÉT KHẨN CẤP</b>\n\n<b>Lý do:</b> {market_msg}")
        return

    # 2. RS Ranking & Momentum (220 phiên)
    idx_df, _ = await asyncio.to_thread(load_history_with_fallback, "VNINDEX", sources, 220)
    rs_results = []
    for t in VN100_TICKERS:
        df_t, _ = await asyncio.to_thread(load_history_with_fallback, t, sources, 220)
        score, momentum = calculate_rs_score(df_t, idx_df)
        rs_results.append((t, score, momentum))
        await asyncio.sleep(0.4)
    
    rs_results.sort(key=lambda x: x[1], reverse=True)
    top_20_count = int(len(VN100_TICKERS) * 0.2)
    top_20_tickers = {x[0] for x in rs_results[:top_20_count]}
    
    # Map kết quả RS và Momentum
    rs_map = {x[0]: (x[1], x[2]) for x in rs_results}
    
    # Bước mới: Tính điểm Sóng ngành
    sector_rs_map = calculate_sector_rs(rs_map)
    
    # Xử lý dữ liệu Top 3 Ngành dẫn dắt
    sorted_sectors = sorted(sector_rs_map.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_sectors[:3]
    top_sectors_info = "\n".join([f"  🏆 {name}: RS {score:.2f}" for name, score in top_3])
    
    results = []
    for t in VN100_TICKERS:
        if t not in top_20_tickers: continue
        log("SCAN", f"Đang phân tích {t}...")
        
        score, momentum = rs_map.get(t, (0.0, 0.0))
        outcome = await asyncio.to_thread(
            evaluate_symbol, t, "VN100", sources, 
            rs_score=score, 
            rs_momentum=momentum,
            is_market_downtrend=is_market_downtrend,
            sector_rs_map=sector_rs_map
        )
        if outcome.signal: results.append(outcome.signal)
        await asyncio.sleep(1.2)
        
    # Gửi báo cáo cuối cùng bao gồm tình hình thị trường và top ngành
    msg = format_telegram_message(results, len(VN100_TICKERS), "VN100", market_info=market_msg, top_sectors_info=top_sectors_info)
    await send_telegram_message(token, chat_id, msg)
    if results:
        log("INFO", f"Gửi {len(results)} tín hiệu thành công.")
    else:
        log("INFO", "Không tìm thấy mã đạt tiêu chuẩn.")

async def send_telegram_message(token, chat_id, text):
    import requests
    if not token or not chat_id:
        log("ERROR", "Thiếu TELEGRAM_TOKEN hoặc TELEGRAM_CHAT_ID trong file .env")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            log("INFO", f"Đã gửi tin nhắn đến Telegram (ChatID: {chat_id})")
        else:
            log("ERROR", f"Telegram API báo lỗi: {response.status_code} - {response.text}")
    except Exception as e:
        log("ERROR", f"Không thể kết nối đến Telegram: {e}")

async def main():
    await scan_once_and_send()

if __name__ == "__main__":
    asyncio.run(main())
