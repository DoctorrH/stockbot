import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo
from typing import Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


from vnstock.api.quote import Quote


def get_history(symbol: str, source: str, length: int) -> pd.DataFrame:
    """
    Lấy dữ liệu lịch sử theo chuẩn Migration 2025.
    """
    q = Quote(symbol=symbol, source=source)
    now = datetime.now()
    # Lấy dư 60% số ngày để bù cuối tuần/lễ
    start_date = (now - timedelta(days=int(length) * 1.6)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    try:
        df = q.history(start=start_date, end=end_date, interval="1D")
        return normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()

# Danh sách VN100 cố định (100 mã) dùng làm input quét.
# Lưu ý: thành phần VN100 có thể thay đổi theo kỳ review của HOSE.
VN100_TICKERS: List[str] = [
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


def env_get(name: str, default: str = "", *, fallbacks: Optional[List[str]] = None) -> str:
    """
    Lấy biến môi trường (headless-friendly). Có hỗ trợ fallback tên cũ để tương thích.
    """
    if name in os.environ and os.environ[name].strip():
        return os.environ[name].strip()
    for fb in fallbacks or []:
        if fb in os.environ and os.environ[fb].strip():
            return os.environ[fb].strip()
    return default


def init_vnstock_user() -> None:
    """
    Nếu có API key, đăng ký user để tăng hạn mức request/phút.
    """
    api_key = env_get("VNSTOCK_API_KEY", "")
    if not api_key:
        return


def get_source_candidates() -> List[str]:
    """
    Danh sách nguồn dữ liệu theo thứ tự ưu tiên.
    Mặc định ưu tiên VCI để tránh lỗi RetryError thường gặp ở KBS.
    """
    raw = env_get("VNSTOCK_SOURCES", "").strip()
    if raw:
        arr = [s.strip().upper() for s in raw.split(",") if s.strip()]
    else:
        primary = env_get("VNSTOCK_SOURCE", "VCI").strip().upper() or "VCI"
        fallback_default = ["VCI", "TCBS", "SSI", "KBS"]
        arr = [primary] + [s for s in fallback_default if s != primary]

    out: List[str] = []
    for s in arr:
        if s not in out:
            out.append(s)
    return out


def get_request_timeout_seconds() -> int:
    try:
        v = int(env_get("REQUEST_TIMEOUT_SECONDS", "30") or "30")
        return max(5, v)
    except Exception:
        return 30


def log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


async def run_blocking_with_timeout(label: str, func, *args, timeout_seconds: int):
    """
    Chạy hàm sync trong thread và timeout cứng để tránh treo CI.
    """
    log("START", f"{label} (timeout={timeout_seconds}s)")
    try:
        result = await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=timeout_seconds)
        log("DONE", label)
        return result
    except asyncio.TimeoutError:
        log("TIMEOUT", f"{label} > {timeout_seconds}s")
        return None
    except Exception as e:
        log("ERROR", f"{label}: {type(e).__name__}: {e}")
        return None
    try:
        from vnstock import register_user  # type: ignore

        register_user(api_key=api_key)
    except Exception:
        # Không chặn bot nếu bước đăng ký không khả dụng ở phiên bản hiện tại
        return


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
    rs_score: float  # Chỉ số RS
    is_weekly_ok: bool  # Trạng thái MA20 tuần
    w_weeks: int  # Số tuần liên tiếp trên MA20 tuần
    spread: float  # Biên độ nến
    warning: str
    reason: str


@dataclass(frozen=True)
class EvalOutcome:
    symbol: str
    exchange: str
    info_line: str
    skip_reason: str
    signal: Optional[SignalResult]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI theo Wilder (EMA smoothing).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def analyze_candle_shape(o: float, h: float, l: float, c: float, po: float, pc: float) -> Optional[str]:
    """
    Phân tích hình dạng nến dựa trên các quy tắc kỹ thuật.
    """
    total_length = h - l
    if total_length == 0:
        return None
    
    body = abs(c - o)
    low_shadow = min(o, c) - l
    
    # Pin Bar (Hammer): (low_shadow > 0.6 * total_length) AND (body nằm ở 1/3 phía trên)
    if (low_shadow > 0.6 * total_length) and (min(o, c) >= l + (2/3) * total_length):
        return "Pin Bar"
    
    # Doji: (body < 0.1 * total_length)
    if body < 0.1 * total_length:
        return "Doji"
        
    # Engulfing: (body > prev_body) AND (close > open) AND (prev_close < prev_open)
    prev_body = abs(pc - po)
    if (body > prev_body) and (c > o) and (pc < po):
        return "Engulfing"
        
    return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá DataFrame OHLCV từ vnstock về các cột:
    ['time', 'open', 'high', 'low', 'close', 'volume']
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # Một số nguồn trả về 'date' thay vì 'time'
    if "time" not in df.columns and "date" in df.columns:
        df.rename(columns={"date": "time"}, inplace=True)

    # Đảm bảo volume dạng số
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Parse time
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Sắp xếp tăng dần theo thời gian
    if "time" in df.columns:
        df = df.sort_values("time")

    need = {"open", "high", "low", "close", "volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.dropna(subset=["close", "volume"])
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def load_history(symbol: str, source: str, length: int) -> pd.DataFrame:
    return get_history(symbol, source, length)


def load_history_with_fallback(symbol: str, sources: List[str], length: int) -> tuple[pd.DataFrame, Optional[str]]:
    for src in sources:
        try:
            df = load_history(symbol=symbol, source=src, length=length)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, src
        except Exception:
            continue
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


def load_intraday(symbol: str, source: str, date_yyyy_mm_dd: str) -> pd.DataFrame:
    """
    Lấy dữ liệu khớp lệnh trong ngày bằng chuẩn Migration 2025.
    """
    q = Quote(symbol=symbol, source=source)
    try:
        df = q.intraday(date=date_yyyy_mm_dd)
        if isinstance(df, pd.DataFrame):
            return df.copy()
    except Exception as e:
        log("ERROR", f"Lỗi load_intraday {symbol}: {e}")
    return pd.DataFrame()


def calculate_rs_score(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> float:
    """
    Tính RS = (Price_now / Price_50) / (Index_now / Index_50)
    """
    if len(stock_df) < 50 or len(index_df) < 50:
        return 0.0
    
    stock_now = stock_df["close"].iloc[-1]
    stock_50 = stock_df["close"].iloc[-50]
    index_now = index_df["close"].iloc[-1]
    index_50 = index_df["close"].iloc[-50]
    
    if stock_50 == 0 or index_50 == 0:
        return 0.0
        
    return (stock_now / stock_50) / (index_now / index_50)


def check_weekly_status(df: pd.DataFrame) -> Tuple[bool, int]:
    """
    Kiểm tra xem giá hiện tại có nằm trên đường MA20 tuần không và đếm số tuần liên tiếp.
    Trả về: (is_above, consecutive_weeks)
    """
    if len(df) < 150: # Cần khoảng 30 tuần dữ liệu
        return False, 0
        
    # Resample sang khung tuần
    df_weekly = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    if len(df_weekly) < 20:
        return False, 0
        
    df_weekly["ma20_w"] = df_weekly["close"].rolling(window=20).mean()
    df_weekly = df_weekly.dropna(subset=["ma20_w"])
    
    if df_weekly.empty:
        return False, 0
        
    is_above = df_weekly["close"].iloc[-1] > df_weekly["ma20_w"].iloc[-1]
    
    consecutive = 0
    for i in range(len(df_weekly) - 1, -1, -1):
        if df_weekly["close"].iloc[i] > df_weekly["ma20_w"].iloc[i]:
            consecutive += 1
        else:
            break
            
    return is_above, consecutive


def load_intraday_with_fallback(symbol: str, sources: List[str], date_yyyy_mm_dd: str) -> tuple[pd.DataFrame, Optional[str]]:
    for src in sources:
        try:
            df = load_intraday(symbol=symbol, source=src, date_yyyy_mm_dd=date_yyyy_mm_dd)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, src
        except Exception:
            continue
    return pd.DataFrame(), None


def intraday_volume_upto(df_intraday: pd.DataFrame, cutoff_hhmm: str) -> Optional[float]:
    if df_intraday is None or len(df_intraday) == 0:
        return None

    df = df_intraday.copy()
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("time") or cols.get("datetime") or cols.get("date")
    vol_col = cols.get("volume") or cols.get("vol")
    if not time_col or not vol_col:
        return None

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df = df.dropna(subset=[time_col, vol_col])
    if df.empty:
        return None

    day = df[time_col].iloc[0].date()
    cutoff = pd.Timestamp(f"{day} {cutoff_hhmm}:00")
    df = df[df[time_col] <= cutoff]
    if df.empty:
        return None

    return float(df[vol_col].sum())


async def check_market_kill_switch(sources: List[str], tickers: List[str], request_timeout: int) -> tuple[bool, str]:
    """
    Kiểm tra các điều kiện an toàn của thị trường.
    Trả về (is_killed, alert_message).
    """
    # 1. Kiểm tra VN-Index
    log("INFO", "Đang kiểm tra Market Kill Switch (VN-Index)...")
    for idx_symbol in ("VNINDEX", "VN-INDEX"):
        res_h = await run_blocking_with_timeout(
            f"Lấy dữ liệu {idx_symbol}",
            load_history_with_fallback,
            idx_symbol,
            sources,
            50,
            timeout_seconds=request_timeout,
        )
        if res_h:
            df, _ = res_h
            if len(df) >= 2:
                df["rsi"] = rsi(df["close"], 14)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                pct_change = ((last["close"] / prev["close"]) - 1) * 100
                rsi_now = last["rsi"]
                rsi_prev = prev["rsi"]
                rsi_drop = rsi_prev - rsi_now
                
                if pct_change < -2.0 or rsi_drop > 5.0:
                    msg = f"🚨 THỊ TRƯỜNG RỦI RO CAO\nVN-Index giảm {pct_change:.2f}% | RSI giảm {rsi_drop:.2f} điểm"
                    return True, msg
            break

    # 2. Kiểm tra Độ rộng thị trường (Mã giảm sàn)
    log("INFO", "Đang kiểm tra Độ rộng thị trường (VN100)...")
    floor_count = 0
    scanned_count = 0
    threshold = len(tickers) * 0.05
    
    for sym in tickers:
        scanned_count += 1
        res = await run_blocking_with_timeout(
            f"Breadth check {sym}",
            load_history_with_fallback,
            sym,
            sources,
            2,
            timeout_seconds=request_timeout,
        )
        if res:
            df_b, _ = res
            if len(df_b) >= 2:
                c = df_b["close"].iloc[-1]
                p = df_b["close"].iloc[-2]
                pct = (c / p - 1) * 100
                if pct <= -6.9:
                    floor_count += 1
        
        if floor_count > threshold:
            msg = f"💀 CẢNH BÁO SẬP DIỆN RỘNG\nSố mã giảm sàn: {floor_count} (>5% danh sách quét)"
            return True, msg
            
        # Nghỉ ngắn hơn vì đây là bước check tiền thị trường
        await asyncio.sleep(1.0)
        if scanned_count % 20 == 0:
            log("INFO", f"Đã check breadth {scanned_count}/{len(tickers)} mã...")

    return False, ""


def evaluate_symbol(symbol: str, exchange: str, sources: List[str], length: int = 260) -> EvalOutcome:
    df, used_source = load_history_with_fallback(symbol=symbol, sources=sources, length=length)
    if not used_source:
        return EvalOutcome(symbol=symbol, exchange=exchange, info_line="", skip_reason="không lấy được dữ liệu lịch sử từ mọi nguồn", signal=None)
    if df.empty or len(df) < 220:
        return EvalOutcome(symbol=symbol, exchange=exchange, info_line="", skip_reason="không đủ dữ liệu để tính MA200", signal=None)

    df = calc_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bộ lọc thanh khoản cơ bản
    vol_avg20 = float(last["vol_avg20_prev"]) if pd.notna(last["vol_avg20_prev"]) else np.nan
    if not np.isfinite(vol_avg20) or vol_avg20 <= 200_000:
        return EvalOutcome(
            symbol=symbol,
            exchange=exchange,
            info_line="",
            skip_reason=f"không thỏa thanh khoản AvgVol20 ({vol_avg20:.0f}) <= 200000",
            signal=None,
        )

    close = float(last["close"])
    prev_close = float(prev["close"])
    pct_change = ((close / prev_close) - 1) * 100 if prev_close else 0.0
    ma20_now = float(last["ma20"]) if pd.notna(last["ma20"]) else np.nan
    ma50_now = float(last["ma50"]) if pd.notna(last["ma50"]) else np.nan
    ma200_now = float(last["ma200"]) if pd.notna(last["ma200"]) else np.nan
    rsi_now = float(last["rsi14"]) if pd.notna(last["rsi14"]) else np.nan
    vol_now = float(last["volume"])

    info_line = (
        f"{symbol}: Giá={close:.2f}, MA20={ma20_now:.2f}, MA50={ma50_now:.2f}, "
        f"MA200={ma200_now:.2f}, RSI={rsi_now:.2f}, %Tăng={pct_change:.2f}, "
        f"Vol={vol_now:.0f}, AvgVol20={vol_avg20:.0f}, src={used_source}"
    )

    if not (np.isfinite(ma20_now) and np.isfinite(ma50_now) and np.isfinite(ma200_now) and np.isfinite(rsi_now)):
        return EvalOutcome(symbol=symbol, exchange=exchange, info_line=info_line, skip_reason="chỉ báo không hợp lệ (NaN)", signal=None)

    # Điều kiện chiến lược + an toàn tối đa
    def get_row_data(idx):
        if idx < 0 or idx >= len(df):
            return None
        r = df.iloc[idx]
        c = float(r["close"])
        v = float(r["volume"])
        m20 = float(r["ma20"]) if pd.notna(r["ma20"]) else np.nan
        m50 = float(r["ma50"]) if pd.notna(r["ma50"]) else np.nan
        m200 = float(r["ma200"]) if pd.notna(r["ma200"]) else np.nan
        rsi_val = float(r["rsi14"]) if pd.notna(r["rsi14"]) else np.nan
        va20 = float(r["vol_avg20_prev"]) if pd.notna(r["vol_avg20_prev"]) else np.nan
        p = float(r["perf"]) if pd.notna(r["perf"]) else 0.0
        rv = v / va20 if va20 > 0 else np.nan
        return {
            "close": c, "volume": v, "ma20": m20, "ma50": m50, "ma200": m200,
            "rsi": rsi_val, "vol_avg20": va20, "perf": p, "rvol": rv
        }

    curr = get_row_data(-1)
    prev1 = get_row_data(-2)
    prev2 = get_row_data(-3)

    if curr is None or not np.isfinite(curr["ma20"]):
        return EvalOutcome(symbol=symbol, exchange=exchange, info_line=info_line, skip_reason="chỉ báo không hợp lệ (NaN)", signal=None)

    # Step 1 Check (Watchlist)
    def is_step1(data):
        if data is None: return False
        # RVOL 0.5 - 1.0 AND Price near MA20 (+-2%) AND Avg Volume > 200k
        cond_rvol = 0.5 <= data["rvol"] < 1.0
        cond_ma20 = abs(data["close"] / data["ma20"] - 1) <= 0.02
        cond_vol = data["vol_avg20"] > 200000
        return cond_rvol and cond_ma20 and cond_vol

    is_watchlist = is_step1(curr)
    is_breakout = False
    if curr["perf"] > 2.0 and prev1 and prev2:
        cond_prev1_sideways = abs(prev1["perf"]) <= 0.5
        cond_prev2_sideways = abs(prev2["perf"]) <= 0.5
        if cond_prev1_sideways and cond_prev2_sideways and is_step1(prev1) and is_step1(prev2):
            is_breakout = True

    if is_breakout:
        filter_label = "🚀 BREAKOUT SIGNAL"
    elif is_watchlist:
        filter_label = "👀 Watchlist (Step 1)"
    else:
        return EvalOutcome(symbol=symbol, exchange=exchange, info_line=info_line, skip_reason="không thỏa bộ lọc 2 lớp (Watchlist/Breakout)", signal=None)

    rvol = curr["rvol"]
    if rvol > 2.0:
        rvol_label = "🔥 DÒNG TIỀN ĐỘT BIẾN"
    elif rvol >= 1.5:
        rvol_label = "⭐ TIỀN VÀO MẠNH"
    elif rvol < 1.0:
        rvol_label = "⚠️ TIỀN YẾU"
    else:
        rvol_label = "Bình thường"

    # Phân tích hình dạng nến
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    candle_shape = analyze_candle_shape(o, h, l, c, po, pc)
    
    # Tính toán đặc điểm nến cho logic mới
    body = abs(c - o)
    candle_range = h - l
    upper_shadow = h - max(o, c)
    
    ma20_distance_pct = (curr["close"] / curr["ma20"] - 1) * 100
    
    # Phân tích hình dạng nến
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    candle_shape = analyze_candle_shape(o, h, l, c, po, pc)
    
    # Lọc Đa khung thời gian: MA20 tuần
    is_weekly_ok, w_weeks = check_weekly_status(df)
    
    # Candle Spread: (High - Low) / Close_yesterday * 100
    candle_spread = (h - l) / pc * 100 if pc > 0 else 0
    
    # Xác định Mức độ ưu tiên và Nhãn mới theo chuẩn 9.5 nâng cao
    priority_level = 4
    priority_label = ""
    
    perf_today = curr["perf"]
    rvol = curr["rvol"]
    m20 = curr["ma20"]
    rs_score = kwargs.get("rs_score", 0.0)

    if is_weekly_ok:
        # 1. BỘ LỌC CỨNG: RS THẤP (Loại bỏ hàng yếu)
        if rs_score < 1.15:
            if perf_today > 3.0:
                priority_label = "⚠️ NỔ GIẢ (RS THẤP)"
                priority_level = 3
            else:
                priority_label = "💤 CHỜ DÒNG TIỀN (KIÊN NHẪN)"
                priority_level = 4
        
        # 2. NHÓM SIÊU CỔ & RS MẠNH (Nhận diện Rũ bỏ sớm - Cập nhật 3 cấp độ)
        elif rs_score > 1.3 and perf_today < 0:
            # Quy tắc loại bỏ: Giảm sâu kèm vol lớn không phải rũ bỏ
            if perf_today < -2.5 and rvol > 0.8:
                priority_label = "👀 THEO DÕI THÊM"
                priority_level = 4
            # Cấp độ 1: Rũ bỏ chuẩn (Cực kỳ an toàn)
            elif rvol < 0.8:
                priority_label = "💎 RŨ BỎ CHUẨN (MUA GOM)"
                priority_level = 1
            # Cấp độ 2: Rũ bỏ linh hoạt (Bắt siêu cổ)
            elif -2.0 < perf_today < 0 and 0.8 <= rvol < 1.1:
                priority_label = "🔥 RŨ BỎ LINH HOẠT (THEO DÕI MUA)"
                priority_level = 1
            else:
                priority_label = "👀 THEO DÕI THÊM"
                priority_level = 4

        # 3. NHÓM SIÊU CỔ (RS > 1.5) - Ưu tiên tuyệt đối khi giá giữ vững
        elif rs_score > 1.5:
            if perf_today >= -2.0:
                priority_label = "🔥 SIÊU CỔ ĐANG CHẠY"
                priority_level = 1
        
        # 4. NHÓM RS MẠNH (1.25 - 1.5) - Các trạng thái khác
        elif 1.25 <= rs_score <= 1.5:
            # Điểm nổ chuẩn
            if perf_today > 2.0 and rvol > 1.5:
                priority_label = "🚀 XÁC NHẬN ĐIỂM NỔ TIN CẬY"
                priority_level = 1
            # Bứt phá cạn cung
            elif perf_today > 2.0 and rvol < 1.0:
                priority_label = "🚀 CẠN CUNG BỨT PHÁ"
                priority_level = 1 if rs_score >= 1.35 else 3
            # Tích lũy kiệt Vol
            elif abs(perf_today) < 1.0 and rvol < 0.8:
                priority_label = "💤 TÍCH LŨY KIỆT VOL (THEO DÕI)"
                priority_level = 2
        
        # Nhóm RS trung bình tích lũy nền dài
        elif 1.2 <= rs_score < 1.25 and is_weekly_ok and w_weeks >= 4 and rvol < 0.8:
            priority_label = "💎 GOM HÀNG NỀN DÀI"
            priority_level = 2
            
        # Nhóm RS hạng 2 có dòng tiền đột biến (Mới)
        elif 1.15 <= rs_score < 1.25 and is_weekly_ok and rvol > 2.5:
            priority_label = "🚀 DÒNG TIỀN ĐỘT BIẾN (HẠNG 2)"
            priority_level = 2
        
        # 4. CHẶN BẪY VOLUME (Climax) - Né bẫy D2D
        if rvol > 5.0:
            priority_label = "⚠️ CAO TRÀO MUA (RỦI RO)"
            priority_level = 3
            
        # 5. BỘ LỌC BIẾN ĐỘ LỎNG (Volatility Filter) - Né bẫy CII
        elif candle_spread > 8.0 and rvol > 1.8:
            priority_label = "⚠️ BIẾN ĐỘNG LỎNG (HƯNG PHẤN QUÁ ĐÀ)"
            priority_level = 3

        # 6. CẢNH BÁO THIẾU VOL (Hàng điều tiết)
        elif perf_today > 2.0 and rvol < 0.7 and rs_score < 1.3:
            priority_label = "⚠️ BẬT TĂNG THIẾU VOL (HÀNG ĐIỀU TIẾT)"
            priority_level = 3

    # Cảnh báo quá điểm mua
    # Ngưỡng Quá mua cực đại cho Siêu cổ (RS > 2.0)
    if rs_score > 2.0 and ma20_distance_pct > 20.0:
        priority_label = "⚠️ QUÁ MUA (KHÔNG ĐU ĐUỔI)"
        priority_level = 3
    # Cảnh báo quá điểm mua thông thường
    elif abs(ma20_distance_pct) > 15.0 and rs_score <= 1.5:
        priority_label = "⚠️ QUÁ ĐIỂM MUA"
        priority_level = 3

    if not priority_label:
        priority_label = "👀 THEO DÕI THÊM"
        priority_level = 4

    special_label = priority_label if priority_label else (candle_shape or "")
    
    reason = f"P{priority_level} | RS={kwargs.get('rs_score', 0):.2f} | W_MA20={is_weekly_ok} | RVOL={rvol:.2f}"
    
    sig = SignalResult(
            symbol=symbol,
            exchange=exchange,
            close=curr["close"],
            pct_change=curr["perf"],
            rsi14=curr["rsi"],
            ma20=curr["ma20"],
            ma50=curr["ma50"],
            ma200=curr["ma200"],
            ma20_distance_pct=ma20_distance_pct,
            vol=curr["volume"],
            vol_avg20=curr["vol_avg20"],
            rvol=rvol,
            rvol_label=rvol_label,
            filter_label=filter_label,
            special_label=special_label,
            priority_level=priority_level,
            rs_score=kwargs.get("rs_score", 0.0),
            is_weekly_ok=is_weekly_ok,
            w_weeks=w_weeks,
            spread=candle_spread,
            warning="",
            reason=reason,
        )
    return EvalOutcome(symbol=symbol, exchange=exchange, info_line=info_line, skip_reason="", signal=sig)


def format_message(results: List[SignalResult], scanned: int, source: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"VN Trend+Momentum Scanner - {now}\nNguồn dữ liệu: {source}\nĐã quét: {scanned} mã\n"
    if not results:
        return header + "\nKết thúc quét: Không có điểm mua an toàn hôm nay."

    # Sắp xếp theo priority_level (tăng dần) rồi đến rs_score (giảm dần)
    results_sorted = sorted(results, key=lambda x: (x.priority_level, -x.rs_score))

    lines: List[str] = [header]
    for r in results_sorted:
        emoji = "🚀" if "XÁC NHẬN" in r.special_label else ("💎" if "SIÊU CỔ" in r.special_label else "👀")
        msg = (
            f"{emoji} <b>{r.symbol}</b> | {r.special_label or r.filter_label}\n"
            f"───────────────────\n"
            f"💰 Giá: <b>{r.close:,.2f}</b> ({r.pct_change:+.2f}%)\n"
            f"📊 RS: <b>{r.rs_score:.2f}</b> | RVOL: <b>{r.rvol:.2f}</b>\n"
            f"📏 Spread: <b>{r.spread:.2f}%</b> | MA20: <b>{r.ma20:.2f}</b>\n"
            f"📍 Cách MA20: <b>{r.ma20_distance_pct:+.2f}%</b> | Tuần: {'✅' if r.is_weekly_ok else '❌'} ({r.w_weeks}w)\n"
            f"───────────────────\n"
        )
        lines.append(msg)
        if r.warning:
            lines.append(f"⚠️ {r.warning}")
        lines.append("") # Khoảng trống giữa các mã
    return "\n".join(lines)


async def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    # Dùng Bot API trực tiếp khi gửi “push” theo chat_id cấu hình sẵn
    from telegram import Bot

    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")


async def scan_once_and_send() -> None:
    load_dotenv()
    init_vnstock_user()

    # Ưu tiên TELEGRAM_TOKEN theo yêu cầu; fallback TELEGRAM_BOT_TOKEN để tương thích.
    token = env_get("TELEGRAM_TOKEN", fallbacks=["TELEGRAM_BOT_TOKEN"])
    chat_id = env_get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise RuntimeError("Thiếu TELEGRAM_TOKEN hoặc TELEGRAM_CHAT_ID trong biến môi trường")

    request_timeout = get_request_timeout_seconds()
    source_candidates = get_source_candidates()
    source = ",".join(source_candidates)
    length = 220
    log("INFO", f"Bắt đầu quét | sources={source} | universe=VN100 cố định | length={length}")
    log("INFO", "Đang khởi động bộ lọc 9.5 điểm (Vui lòng đợi khoảng 2 phút do giới hạn API)...")

    symbols: List[tuple[str, str]] = []
    for s in VN100_TICKERS:
        s2 = s.strip().upper()
        if s2.isalpha() and 2 <= len(s2) <= 5:
            symbols.append((s2, "VN100"))

    log("INFO", f"Tổng số mã VN100 sẽ quét: {len(symbols)}")

    # Bước 1: Tiền xử lý - Tính RS Ranking
    log("INFO", "Bắt đầu bước Tiền xử lý (Pre-processing): Tính RS Ranking...")
    index_df, _ = load_history_with_fallback("VNINDEX", source_candidates, 60)
    
    rs_results = []
    for sym, _ in symbols:
        df_rs, _ = load_history_with_fallback(sym, source_candidates, 60)
        score = calculate_rs_score(df_rs, index_df)
        rs_results.append((sym, score))
        await asyncio.sleep(0.5)
        
    # Lấy top 20%
    rs_results.sort(key=lambda x: x[1], reverse=True)
    top_rs_count = int(len(symbols) * 0.2)
    top_rs_tickers = {x[0] for x in rs_results[:top_rs_count]}
    rs_map = {x[0]: x[1] for x in rs_results}
    
    log("INFO", f"Đã xác định {len(top_rs_tickers)} mã mạnh nhất thị trường (Top 20% RS).")

    results: List[SignalResult] = []
    scanned = 0
    count_signals = 0
    kill_switch_res = await check_market_kill_switch(
        source_candidates,
        VN100_TICKERS,
        request_timeout,
    )
    is_killed, kill_msg = kill_switch_res
    if is_killed:
        log("KILL", f"Kích hoạt Kill Switch: {kill_msg}")
        header = f"VN Trend+Momentum Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        full_msg = f"{header}\n{kill_msg}\n\n⚠️ TÍN HIỆU DỪNG: Ngừng quét toàn bộ để đảm bảo an toàn vốn."
        await send_telegram_message(token=token, chat_id=chat_id, text=full_msg)
        return

    for sym, ex in symbols:
        if sym not in top_rs_tickers:
            continue
            
        scanned += 1
        log("SCAN", f"Đang quét mã mạnh {sym} ({ex})... [{scanned}/{len(top_rs_tickers)}]")
        try:
            outcome = await run_blocking_with_timeout(
                f"Phân tích {sym}",
                evaluate_symbol,
                sym,
                ex,
                source_candidates,
                length,
                rs_score=rs_map.get(sym, 0.0), # Truyền RS score vào
                timeout_seconds=request_timeout,
            )
            if not outcome:
                continue
            
            if outcome.info_line:
                log("INFO", outcome.info_line)

            if outcome.signal:
                results.append(outcome.signal)
                count_signals += 1
                log("HIT", f"{sym}: thỏa điều kiện mua.")
            else:
                reason = outcome.skip_reason or "không thỏa điều kiện"
                log("SKIP", f"{sym} {reason}")

            # Rate-limit guard
            await asyncio.sleep(2)

        except Exception:
            # Bỏ qua mã lỗi dữ liệu để không dừng toàn bộ vòng quét
            log("ERROR", f"Bỏ qua mã {sym}")
            continue

        # Nghỉ 1 giây sau mỗi mã để giảm nhịp truy cập quá nhanh
        await asyncio.sleep(2)

        # Thở nhẹ thêm để giảm rủi ro rate-limit (ngoài sleep(2) theo từng API call)
        if scanned % 10 == 0:
            await asyncio.sleep(1.2)

    log("INFO", f"Quét xong. Số mã đạt điều kiện: {len(results)}")
    if not results:
        msg = "--- 📭 KHÔNG CÓ MÃ NÀO ĐỦ ĐIỀU KIỆN (CHUẨN 9.5 ĐIỂM) TRONG PHIÊN HÔM NAY ---"
        log("INFO", msg)
        await send_telegram_message(token=token, chat_id=chat_id, text=msg)
        return
        
    results = sorted(results, key=lambda x: (x.exchange, x.symbol))
    msg = format_message(results=results, scanned=scanned, source=source)
    await send_telegram_message(token=token, chat_id=chat_id, text=msg)


def latest_price_from_intraday(df_intraday: pd.DataFrame) -> Optional[float]:
    if df_intraday is None or len(df_intraday) == 0:
        return None
    cols = {c.lower(): c for c in df_intraday.columns}
    price_col = cols.get("price") or cols.get("close") or cols.get("last")
    if not price_col:
        return None
    s = pd.to_numeric(df_intraday[price_col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /test: lấy giá hiện tại 3 mã để kiểm tra bot hoạt động.
    """
    load_dotenv()
    init_vnstock_user()

    source_candidates = get_source_candidates()
    request_timeout = get_request_timeout_seconds()
    cutoff_hhmm = datetime.now().strftime("%H:%M")
    today = datetime.now().strftime("%Y-%m-%d")

    symbols = ["FPT", "SSI", "HPG"]
    parts = [f"Test OK ({today} {cutoff_hhmm}) | Sources: {', '.join(source_candidates)}"]

    for sym in symbols:
        try:
            # Ưu tiên intraday (giá “hiện tại”)
            res_i = await run_blocking_with_timeout(
                f"/test intraday {sym}",
                load_intraday_with_fallback,
                sym,
                source_candidates,
                today,
                timeout_seconds=request_timeout,
            )
            if res_i is None:
                parts.append(f"- {sym}: lỗi timeout")
                continue
            df_i, used_intraday = res_i
            px = latest_price_from_intraday(df_i)
            # Rate-limit guard
            await asyncio.sleep(2)

            # Fallback: lấy close gần nhất nếu intraday không có
            if px is None:
                res_h = await run_blocking_with_timeout(
                    f"/test history {sym}",
                    load_history_with_fallback,
                    sym,
                    source_candidates,
                    5,
                    timeout_seconds=request_timeout,
                )
                if res_h is None:
                    parts.append(f"- {sym}: lỗi timeout")
                    continue
                df_h, used_history = res_h
                if not df_h.empty:
                    px = float(df_h["close"].iloc[-1])
                    used_intraday = used_history
                await asyncio.sleep(2)

            if px is None:
                parts.append(f"- {sym}: (không lấy được giá)")
            else:
                src_label = used_intraday or "unknown"
                parts.append(f"- {sym}: {px:.2f} (src={src_label})")
        except Exception as e:
            parts.append(f"- {sym}: lỗi {type(e).__name__}")

    await update.effective_message.reply_text("\n".join(parts))


async def cmd_check_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /check_all: chế độ demo để xem MA20, MA50, MA200, RSI của FPT/HPG.
    """
    load_dotenv()
    init_vnstock_user()
    sources = get_source_candidates()
    request_timeout = get_request_timeout_seconds()
    length = 220

    demo_symbols = ["FPT", "HPG"]
    lines = [f"Demo chi so | Sources: {', '.join(sources)}"]

    for sym in demo_symbols:
        res_h = await run_blocking_with_timeout(
            f"/check_all history {sym}",
            load_history_with_fallback,
            sym,
            sources,
            length,
            timeout_seconds=request_timeout,
        )
        if res_h is None:
            lines.append(f"- {sym}: timeout khi lấy dữ liệu")
            continue
        df, used = res_h
        await asyncio.sleep(2)
        if df.empty or len(df) < 220 or not used:
            lines.append(f"- {sym}: khong du du lieu")
            continue

        x = calc_indicators(df).iloc[-1]
        close = float(x["close"])
        ma20 = float(x["ma20"]) if pd.notna(x["ma20"]) else np.nan
        ma50 = float(x["ma50"]) if pd.notna(x["ma50"]) else np.nan
        ma200 = float(x["ma200"]) if pd.notna(x["ma200"]) else np.nan
        rsi14 = float(x["rsi14"]) if pd.notna(x["rsi14"]) else np.nan

        lines.append(
            f"- {sym}: Close={close:.2f}, MA20={ma20:.2f}, MA50={ma50:.2f}, "
            f"MA200={ma200:.2f}, RSI14={rsi14:.2f} (src={used})"
        )

    await update.effective_message.reply_text("\n".join(lines))


async def main() -> None:
    load_dotenv()

    token = env_get("TELEGRAM_TOKEN", fallbacks=["TELEGRAM_BOT_TOKEN"])
    if not token:
        log("ERROR", "Thiếu TELEGRAM_TOKEN trong biến môi trường")
        return

    # Chế độ chạy 1 lần (phù hợp cho GitHub Actions/Task Scheduler)
    # Set SCAN_ONCE=1 để quét và gửi xong thì thoát.
    scan_once_flag = env_get("SCAN_ONCE", "").lower() in {"1", "true", "yes", "y"}

    # Tạo Application để quản lý lifecycle kết nối Telegram (headless-friendly)
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("test", cmd_test))
    application.add_handler(CommandHandler("check_all", cmd_check_all))

    # Tuỳ chọn: chạy quét theo lịch nội bộ (nếu bạn chạy bot 24/7)
    daily_hhmm = env_get("DAILY_SCAN_HHMM", "")
    if daily_hhmm:
        try:
            hh, mm = [int(x) for x in daily_hhmm.split(":")]
            tz = ZoneInfo("Asia/Ho_Chi_Minh")

            async def _job(_: ContextTypes.DEFAULT_TYPE) -> None:
                try:
                    await scan_once_and_send()
                except Exception as e:
                    log("ERROR", f"Lỗi job daily scan: {type(e).__name__}: {e}")

            application.job_queue.run_daily(
                _job,
                time=datetime.now(tz).replace(hour=hh, minute=mm, second=0, microsecond=0).timetz(),
            )
        except Exception:
            pass

    await application.initialize()
    await application.start()
    try:
        if scan_once_flag:
            # Dành cho GitHub Actions: quét xong là thoát
            try:
                await scan_once_and_send()
            except Exception as e:
                log("ERROR", f"Lỗi scan_once: {type(e).__name__}: {e}")
            return

        # Chạy bot Telegram (polling) để nhận lệnh /test
        if application.updater is None:
            raise RuntimeError("Updater không khả dụng. Hãy nâng/cài đúng python-telegram-bot.")

        await application.updater.start_polling()

        # Chạy vô hạn cho đến khi bị dừng (Ctrl+C / SIGTERM)
        await asyncio.Event().wait()
    finally:
        # Đóng kết nối an toàn trước khi thoát
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
    log("EXIT", "Bot đã hoàn tất và thoát an toàn.")
