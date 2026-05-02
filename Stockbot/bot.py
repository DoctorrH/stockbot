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
        from vnstock import Listing, Quote  # type: ignore

        return Listing, Quote
    except Exception:
        # Một số bản vẫn dùng module vnstock sau khi cài vnstock3, nên fallback này là “best effort”
        from vnstock import Listing, Quote  # type: ignore

        return Listing, Quote


Listing, Quote = _import_vnstock()


def init_vnstock_user() -> None:
    """
    Nếu có API key, đăng ký user để tăng hạn mức request/phút.
    """
    api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
    if not api_key:
        return
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
    vol: float
    vol_avg20: float
    intraday_vol: float
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


def safe_get_symbols(exchange: str, source: str = "KBS") -> List[str]:
    """
    Lấy danh sách mã theo sàn (HOSE/HNX).
    Tuỳ phiên bản vnstock, Listing có thể có .hose()/.hnx() hoặc trả về DataFrame có cột sàn.
    """
    ex = exchange.strip().upper()
    listing = Listing(source=source)

    # 1) Ưu tiên API trực tiếp nếu có
    direct_map = {
        "HOSE": ("hose", "hose_symbols", "symbols_hose"),
        "HNX": ("hnx", "hnx_symbols", "symbols_hnx"),
    }
    for attr in direct_map.get(ex, ()):
        if hasattr(listing, attr):
            try:
                syms = getattr(listing, attr)()
                if isinstance(syms, (list, tuple, pd.Series)):
                    return [str(s).strip().upper() for s in syms if str(s).strip()]
                if isinstance(syms, pd.DataFrame) and "symbol" in syms.columns:
                    return [str(s).strip().upper() for s in syms["symbol"].tolist()]
            except Exception:
                pass

    # 2) Fallback: lấy toàn bộ rồi lọc theo cột exchange
    try:
        all_syms = listing.all_symbols()
        if isinstance(all_syms, (list, tuple, pd.Series)):
            # Không có metadata về sàn → không thể lọc chắc chắn. Trả về danh sách như là “all”.
            return [str(s).strip().upper() for s in all_syms if str(s).strip()]

        if isinstance(all_syms, pd.DataFrame):
            cols = {c.lower(): c for c in all_syms.columns}
            sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("code")
            exch_col = cols.get("exchange") or cols.get("floor") or cols.get("market")

            if sym_col and exch_col:
                ex_df = all_syms[all_syms[exch_col].astype(str).str.upper().str.contains(ex)]
                return [str(s).strip().upper() for s in ex_df[sym_col].tolist() if str(s).strip()]

            if sym_col:
                return [str(s).strip().upper() for s in all_syms[sym_col].tolist() if str(s).strip()]
    except Exception:
        pass

    return []


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


def evaluate_symbol(symbol: str, exchange: str, source: str, length: int = 130) -> Optional[SignalResult]:
    df = load_history(symbol=symbol, source=source, length=length)
    if df.empty or len(df) < 60:
        return None

    df["ma20"] = sma(df["close"], 20)
    df["ma50"] = sma(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)

    # Trung bình khối lượng 20 phiên: dùng 20 phiên TRƯỚC phiên hiện tại để tránh “nhúng” volume hôm nay
    df["vol_avg20_prev"] = df["volume"].shift(1).rolling(window=20, min_periods=20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bộ lọc volume 20 phiên > 300k
    vol_avg20 = float(last["vol_avg20_prev"]) if pd.notna(last["vol_avg20_prev"]) else np.nan
    if not np.isfinite(vol_avg20) or vol_avg20 <= 300_000:
        return None

    close = float(last["close"])
    prev_close = float(prev["close"])
    pct_change = ((close / prev_close) - 1) * 100 if prev_close else 0.0
    ma20_now = float(last["ma20"]) if pd.notna(last["ma20"]) else np.nan
    ma50_now = float(last["ma50"]) if pd.notna(last["ma50"]) else np.nan
    rsi_now = float(last["rsi14"]) if pd.notna(last["rsi14"]) else np.nan
    vol_now = float(last["volume"])

    if not (np.isfinite(ma20_now) and np.isfinite(ma50_now) and np.isfinite(rsi_now)):
        return None

    # Điều kiện lọc xu hướng
    cond_close_above_ma50 = close > ma50_now

    # Điều kiện điểm mua
    prev_ma20 = float(prev["ma20"]) if pd.notna(prev["ma20"]) else np.nan
    cond_cross_up_ma20 = np.isfinite(prev_ma20) and (prev_close <= prev_ma20) and (close > ma20_now)

    cond_rsi_range = 45 <= rsi_now <= 60

    # Xác nhận volume intraday sẽ check ở scan_once_and_send (cần cutoff)

    if all([cond_close_above_ma50, cond_cross_up_ma20, cond_rsi_range]):
        reason = "Close>MA50, cross-up MA20, RSI(14) 45-60"
        return SignalResult(
            symbol=symbol,
            exchange=exchange,
            close=close,
            pct_change=pct_change,
            rsi14=rsi_now,
            ma20=ma20_now,
            ma50=ma50_now,
            vol=vol_now,
            vol_avg20=vol_avg20,
            intraday_vol=0.0,
            reason=reason,
        )

    return None


def format_message(results: List[SignalResult], scanned: int, source: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"VN Trend+Momentum Scanner - {now}\nNguồn dữ liệu: {source}\nĐã quét: {scanned} mã\n"
    if not results:
        return header + "\nKhông có mã nào thỏa điều kiện hôm nay."

    lines: List[str] = []
    lines.append("Mã      Sàn   Giá      %Tăng   Vol@cutoff/Avg20   Lý do")
    lines.append("-" * 78)
    for r in results:
        ratio = (r.intraday_vol / r.vol_avg20) if r.vol_avg20 else 0.0
        sym = r.symbol.ljust(6)
        ex = r.exchange.ljust(5)
        price = f"{r.close:.2f}".rjust(7)
        chg = f"{r.pct_change:+.2f}%".rjust(7)
        vr = f"{ratio:.2f}x".rjust(6)
        lines.append(f"{sym}  {ex} {price}  {chg}  {vr}            {r.reason}")

    body = "\n".join(lines)
    return header + "\n```\n" + body + "\n```"


async def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    # Dùng Bot API trực tiếp khi gửi “push” theo chat_id cấu hình sẵn
    from telegram import Bot

    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")


async def scan_once_and_send() -> None:
    load_dotenv()
    init_vnstock_user()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong file .env")

    source = os.getenv("VNSTOCK_SOURCE", "KBS").strip() or "KBS"
    length = int(os.getenv("HISTORY_LENGTH", "130").strip() or "130")
    exchanges = [x.strip().upper() for x in os.getenv("EXCHANGES", "HOSE,HNX").split(",") if x.strip()]
    cutoff_hhmm = os.getenv("VOLUME_CUTOFF_HHMM", "14:25").strip() or "14:25"
    volume_ratio_min = float(os.getenv("VOLUME_RATIO_MIN", "0.80").strip() or "0.80")
    today = datetime.now().strftime("%Y-%m-%d")

    symbols: List[tuple[str, str]] = []
    for ex in exchanges:
        for s in safe_get_symbols(exchange=ex, source=source):
            if s.isalpha() and 2 <= len(s) <= 5:
                symbols.append((s, ex))

    results: List[SignalResult] = []
    scanned = 0

    for sym, ex in symbols:
        scanned += 1
        try:
            r0 = evaluate_symbol(symbol=sym, exchange=ex, source=source, length=length)
            if not r0:
                continue

            # Rate-limit guard: sau mỗi lần gọi history() (nằm trong evaluate_symbol/load_history)
            await asyncio.sleep(2)

            intraday_df = load_intraday(symbol=sym, source=source, date_yyyy_mm_dd=today)
            intraday_vol = intraday_volume_upto(intraday_df, cutoff_hhmm=cutoff_hhmm)
            if intraday_vol is None:
                continue

            # Rate-limit guard: sau mỗi lần gọi intraday()
            await asyncio.sleep(2)

            if intraday_vol < volume_ratio_min * r0.vol_avg20:
                continue

            results.append(
                SignalResult(
                    symbol=r0.symbol,
                    exchange=r0.exchange,
                    close=r0.close,
                    pct_change=r0.pct_change,
                    rsi14=r0.rsi14,
                    ma20=r0.ma20,
                    ma50=r0.ma50,
                    vol=r0.vol,
                    vol_avg20=r0.vol_avg20,
                    intraday_vol=float(intraday_vol),
                    reason=r0.reason + f" + Vol@{cutoff_hhmm} ≥ {int(volume_ratio_min * 100)}% Avg20",
                )
            )
        except Exception:
            # Bỏ qua mã lỗi dữ liệu để không dừng toàn bộ vòng quét
            continue

        # Thở nhẹ thêm để giảm rủi ro rate-limit (ngoài sleep(2) theo từng API call)
        if scanned % 10 == 0:
            await asyncio.sleep(0.5)

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

    source = os.getenv("VNSTOCK_SOURCE", "KBS").strip() or "KBS"
    cutoff_hhmm = datetime.now().strftime("%H:%M")
    today = datetime.now().strftime("%Y-%m-%d")

    symbols = ["FPT", "SSI", "HPG"]
    parts = [f"Test OK ({today} {cutoff_hhmm}) | Source: {source}"]

    for sym in symbols:
        try:
            # Ưu tiên intraday (giá “hiện tại”)
            df_i = load_intraday(symbol=sym, source=source, date_yyyy_mm_dd=today)
            px = latest_price_from_intraday(df_i)
            # Rate-limit guard
            await asyncio.sleep(2)

            # Fallback: lấy close gần nhất nếu intraday không có
            if px is None:
                df_h = load_history(symbol=sym, source=source, length=5)
                if not df_h.empty:
                    px = float(df_h["close"].iloc[-1])
                await asyncio.sleep(2)

            if px is None:
                parts.append(f"- {sym}: (không lấy được giá)")
            else:
                parts.append(f"- {sym}: {px:.2f}")
        except Exception as e:
            parts.append(f"- {sym}: lỗi {type(e).__name__}")

    await update.effective_message.reply_text("\n".join(parts))


async def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong file .env")

    # Chế độ chạy 1 lần (phù hợp cho GitHub Actions/Task Scheduler)
    # Set SCAN_ONCE=1 để quét và gửi xong thì thoát.
    scan_once_flag = os.getenv("SCAN_ONCE", "").strip().lower() in {"1", "true", "yes", "y"}
    if scan_once_flag:
        await scan_once_and_send()
        return

    # Chạy bot Telegram (polling) để nhận lệnh /test
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("test", cmd_test))

    # Tuỳ chọn: nếu bạn vẫn muốn script tự “push” kết quả quét theo lịch nội bộ
    # (không cần Task Scheduler), bật DAILY_SCAN_HHMM. Ví dụ: DAILY_SCAN_HHMM=14:25
    daily_hhmm = os.getenv("DAILY_SCAN_HHMM", "").strip()
    if daily_hhmm:
        try:
            hh, mm = [int(x) for x in daily_hhmm.split(":")]
            tz = ZoneInfo("Asia/Ho_Chi_Minh")

            async def _job(_: ContextTypes.DEFAULT_TYPE) -> None:
                await scan_once_and_send()

            app.job_queue.run_daily(_job, time=datetime.now(tz).replace(hour=hh, minute=mm, second=0, microsecond=0).timetz())
        except Exception:
            pass

    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()


if __name__ == "__main__":
    asyncio.run(main())
