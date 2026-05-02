import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


def _import_vnstock():
    """
    vnstock3 đã đổi tên gói trên PyPI thành 'vnstock' (nhưng nhiều nơi vẫn cài 'vnstock3').
    Đoạn import này cố gắng tương thích cả hai.
    """
    try:
        from vnstock import Quote  # type: ignore

        return Quote
    except Exception:
        # Một số bản vẫn dùng module vnstock sau khi cài vnstock3, nên fallback này là “best effort”
        from vnstock import Quote  # type: ignore

        return Quote


Quote = _import_vnstock()

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
    vol: float
    vol_avg20: float
    warning: str
    reason: str


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
    """
    Lấy OHLCV theo ngày cho 1 mã. Dùng length (số phiên lùi lại) để tránh phụ thuộc ngày hệ thống.
    """
    q = Quote(symbol=symbol, source=source)
    try:
        df = q.history(length=str(length), interval="1D")
    except Exception:
        df = q.history(length=str(length), interval="d")
    return normalize_ohlcv(df)


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
    return out


def load_intraday(symbol: str, source: str, date_yyyy_mm_dd: str) -> pd.DataFrame:
    """
    Lấy dữ liệu khớp lệnh trong ngày (intraday). Tham số có thể khác nhau theo nguồn,
    nên thử vài cách phổ biến để tương thích.
    """
    q = Quote(symbol=symbol, source=source)
    last_err: Optional[Exception] = None
    for kwargs in ({"date": date_yyyy_mm_dd}, {"trading_date": date_yyyy_mm_dd}):
        try:
            df = q.intraday(**kwargs)
            if isinstance(df, pd.DataFrame):
                return df.copy()
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return pd.DataFrame()


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


def is_market_bad(sources: List[str], length: int = 10, threshold_pct: float = -1.0) -> bool:
    """
    VN-Index giảm mạnh (>1%) thì trả True để chèn cảnh báo thận trọng.
    """
    for idx_symbol in ("VNINDEX", "VN-INDEX"):
        df, _src = load_history_with_fallback(symbol=idx_symbol, sources=sources, length=length)
        if df.empty or len(df) < 2:
            continue
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        if prev == 0:
            continue
        pct = ((last / prev) - 1) * 100
        if pct <= threshold_pct:
            return True
        return False
    return False


def evaluate_symbol(symbol: str, exchange: str, sources: List[str], length: int = 260) -> Optional[SignalResult]:
    df, used_source = load_history_with_fallback(symbol=symbol, sources=sources, length=length)
    if not used_source:
        return None
    if df.empty or len(df) < 220:
        return None

    df = calc_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bộ lọc thanh khoản cơ bản
    vol_avg20 = float(last["vol_avg20_prev"]) if pd.notna(last["vol_avg20_prev"]) else np.nan
    if not np.isfinite(vol_avg20) or vol_avg20 <= 200_000:
        return None

    close = float(last["close"])
    prev_close = float(prev["close"])
    pct_change = ((close / prev_close) - 1) * 100 if prev_close else 0.0
    ma20_now = float(last["ma20"]) if pd.notna(last["ma20"]) else np.nan
    ma50_now = float(last["ma50"]) if pd.notna(last["ma50"]) else np.nan
    ma200_now = float(last["ma200"]) if pd.notna(last["ma200"]) else np.nan
    rsi_now = float(last["rsi14"]) if pd.notna(last["rsi14"]) else np.nan
    vol_now = float(last["volume"])

    if not (np.isfinite(ma20_now) and np.isfinite(ma50_now) and np.isfinite(ma200_now) and np.isfinite(rsi_now)):
        return None

    # Điều kiện chiến lược + an toàn tối đa
    prev_ma20 = float(prev["ma20"]) if pd.notna(prev["ma20"]) else np.nan
    if not np.isfinite(prev_ma20):
        return None

    # Trend: giá > MA50 và MA50 > MA200
    cond_close_above_ma50 = close > ma50_now
    cond_ma50_above_ma200 = ma50_now > ma200_now
    # Cross-up MA20 hôm nay
    cond_cross_up_ma20 = (prev_close <= prev_ma20) and (close > ma20_now)
    # Không mua đuổi: không cao quá 5% so với MA20
    cond_not_chase = close <= 1.05 * ma20_now
    # Tín hiệu sức mạnh nến + volume
    cond_price_jump = pct_change > 2.0
    cond_vol_surge = vol_now > 1.5 * vol_avg20
    # RSI vừa mạnh lên
    cond_rsi_range = 50 <= rsi_now <= 60

    if all(
        [
            cond_close_above_ma50,
            cond_ma50_above_ma200,
            cond_cross_up_ma20,
            cond_not_chase,
            cond_price_jump,
            cond_vol_surge,
            cond_rsi_range,
        ]
    ):
        reason = f"Gia vuot MA20, RSI dep, xu huong MA50/MA200 on dinh (src={used_source})"
        return SignalResult(
            symbol=symbol,
            exchange=exchange,
            close=close,
            pct_change=pct_change,
            rsi14=rsi_now,
            ma20=ma20_now,
            ma50=ma50_now,
            ma200=ma200_now,
            vol=vol_now,
            vol_avg20=vol_avg20,
            warning="",
            reason=reason,
        )

    return None


def format_message(results: List[SignalResult], scanned: int, source: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"VN Trend+Momentum Scanner - {now}\nNguồn dữ liệu: {source}\nĐã quét: {scanned} mã\n"
    if not results:
        return header + "\nKết thúc quét: Không có điểm mua an toàn hôm nay."

    lines: List[str] = [header]
    for r in results:
        lines.append(
            f"🚀 PHÁT HIỆN ĐIỂM MUA: {r.symbol} - Giá: {r.close:.2f}. "
            f"Lý do: Giá vượt MA20, RSI đẹp, xu hướng trung hạn (MA50) ổn định."
        )
        if r.warning:
            lines.append(r.warning)
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
    # Chỉ lấy mức dữ liệu tối thiểu cần thiết cho MA20/MA50/MA200/RSI + vol20
    length = 220
    log("INFO", f"Bắt đầu quét | sources={source} | universe=VN100 cố định | length={length}")
    log("INFO", "Đang bắt đầu quét danh sách VN100 cố định (100 mã)")

    symbols: List[tuple[str, str]] = []
    for s in VN100_TICKERS:
        s2 = s.strip().upper()
        if s2.isalpha() and 2 <= len(s2) <= 5:
            symbols.append((s2, "VN100"))

    log("INFO", f"Tổng số mã VN100 sẽ quét: {len(symbols)}")

    results: List[SignalResult] = []
    scanned = 0
    market_bad_res = await run_blocking_with_timeout(
        "Đánh giá thị trường chung VN-Index",
        is_market_bad,
        source_candidates,
        10,
        -1.0,
        timeout_seconds=request_timeout,
    )
    market_bad = bool(market_bad_res) if market_bad_res is not None else False

    for sym, ex in symbols:
        scanned += 1
        log("SCAN", f"Đang quét mã {sym} ({ex})... [{scanned}/{len(symbols)}]")
        try:
            r0 = await run_blocking_with_timeout(
                f"Phân tích {sym}",
                evaluate_symbol,
                sym,
                ex,
                source_candidates,
                length,
                timeout_seconds=request_timeout,
            )
            if not r0:
                log("SKIP", f"Bỏ qua mã {sym}")
                continue

            # Rate-limit guard: sau mỗi lần gọi history() (nằm trong evaluate_symbol/load_history)
            await asyncio.sleep(2)

            results.append(
                SignalResult(
                    symbol=r0.symbol,
                    exchange=r0.exchange,
                    close=r0.close,
                    pct_change=r0.pct_change,
                    rsi14=r0.rsi14,
                    ma20=r0.ma20,
                    ma50=r0.ma50,
                    ma200=r0.ma200,
                    vol=r0.vol,
                    vol_avg20=r0.vol_avg20,
                    warning="Thị trường chung đang xấu, hãy thận trọng" if market_bad else "",
                    reason=r0.reason,
                )
            )
            log("HIT", f"{sym}: thỏa điều kiện mua.")
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
