# ╔══════════════════════════════════════════════════════════════╗
# ║   INSTITUTIONAL STOCK SCREENER  —  v4.0 (single-file)        ║
# ║   อัปเกรดจาก v3.5 ตาม 12 ข้อกำหนด:                           ║
# ║   1. ATR + Support Zone Detection                             ║
# ║   2. VWAP (intraday-style rolling)                            ║
# ║   3. Break of Structure (BOS)                                 ║
# ║   4. VCP Volatility Contraction                               ║
# ║   5. Multi-Timeframe Weekly EMA40 Alignment                   ║
# ║   6. RS เป็นเงื่อนไขบังคับ                                     ║
# ║   7. God-Tier Signals + Stop Loss                             ║
# ║   8. Mobile-Friendly Card View                                ║
# ║   9. Position Sizing Calculator                               ║
# ║   10. Market Regime Filter (Top-Down)                         ║
# ║   11. Fundamental Confluence (Growth Stock badge)             ║
# ║   12. Mini User Guide                                         ║
# ╚══════════════════════════════════════════════════════════════╝

import datetime
import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("screener")


# ════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════

def log_err(context: str, e: Exception) -> None:
    logger.warning("%s -> %s: %s", context, type(e).__name__, e)


def retry(times: int = 3, base_delay: float = 0.6, exceptions=(Exception,)):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < times - 1:
                        msg = str(e)
                        is_rate_limit = ("Rate limit" in msg or "Too Many Requests" in msg
                                         or "429" in msg or "RateLimitError" in type(e).__name__)
                        delay = (8 * (2 ** attempt) + random.uniform(0, 2)) if is_rate_limit \
                            else (base_delay * (2 ** attempt) + random.uniform(0, 0.3))
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return deco


def to_date_indexed(s: pd.Series) -> pd.Series:
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out = s.copy()
    out.index = idx.normalize()
    return out


# ════════════════════════════════════════════════════════
# DISK CACHE & PERSISTENCE
# ════════════════════════════════════════════════════════

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
WATCHLIST_PATH = os.path.join(CACHE_DIR, "watchlist.json")
SIGNALS_DIR = os.path.join(CACHE_DIR, "last_signals")
os.makedirs(SIGNALS_DIR, exist_ok=True)


def _next_refresh_time(now: datetime.datetime) -> datetime.datetime:
    bkk = ZoneInfo("Asia/Bangkok")
    now_bkk = now.astimezone(bkk)
    cutoff_today = now_bkk.replace(hour=4, minute=0, second=0, microsecond=0)
    return cutoff_today if now_bkk >= cutoff_today else cutoff_today - datetime.timedelta(days=1)


def cache_key(universe: str, tickers: tuple, period: str, interval: str) -> str:
    raw = f"{universe}|{period}|{interval}|{','.join(sorted(tickers))}"
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe_name = "".join(c for c in universe if c.isalnum())[:20]
    return f"{safe_name}_{h}"


def load_disk_cache(universe, tickers, period, interval):
    key = cache_key(universe, tickers, period, interval)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_at = datetime.datetime.fromisoformat(payload["saved_at"])
        cutoff = _next_refresh_time(datetime.datetime.now(ZoneInfo("Asia/Bangkok")))
        if saved_at < cutoff:
            return None
        return pd.DataFrame(payload["data"])
    except Exception as e:
        log_err(f"load_disk_cache({universe})", e)
        return None


def save_disk_cache(universe, tickers, period, interval, df):
    key = cache_key(universe, tickers, period, interval)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        payload = {
            "saved_at": datetime.datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
            "universe": universe,
            "data": df.to_dict(orient="records"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str, ensure_ascii=False)
    except Exception as e:
        log_err(f"save_disk_cache({universe})", e)


def cache_age_label(universe, tickers, period, interval):
    key = cache_key(universe, tickers, period, interval)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_at = datetime.datetime.fromisoformat(payload["saved_at"])
        now = datetime.datetime.now(ZoneInfo("Asia/Bangkok"))
        delta_min = int((now - saved_at).total_seconds() / 60)
        if delta_min < 60:
            return f"สแกนล่าสุด {delta_min} นาทีที่แล้ว"
        elif delta_min < 1440:
            return f"สแกนล่าสุด {delta_min // 60} ชม.ที่แล้ว"
        return f"สแกนล่าสุด {saved_at.strftime('%d/%m %H:%M')}"
    except Exception as e:
        log_err(f"cache_age_label({universe})", e)
        return ""


def clear_cache_for(universe, tickers, period, interval):
    key = cache_key(universe, tickers, period, interval)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def load_watchlist():
    if not os.path.exists(WATCHLIST_PATH):
        return []
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_err("load_watchlist", e)
        return []


def save_watchlist(items):
    try:
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
    except Exception as e:
        log_err("save_watchlist", e)


def _signals_path(universe):
    safe = "".join(c for c in universe if c.isalnum())[:30] or "default"
    return os.path.join(SIGNALS_DIR, f"{safe}.json")


def load_last_signals(universe):
    path = _signals_path(universe)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_err("load_last_signals", e)
        return {}


def save_last_signals(universe, mapping):
    path = _signals_path(universe)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
    except Exception as e:
        log_err("save_last_signals", e)


# ════════════════════════════════════════════════════════
# UNIVERSES
# ════════════════════════════════════════════════════════

@st.cache_data(ttl=86400)
def fetch_sp500():
    try:
        import requests
        from io import StringIO
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = sorted([str(s).strip().replace(".", "-") for s in df[col].dropna()])
        if len(tickers) > 400:
            return tickers
    except Exception as e:
        log_err("fetch_sp500(github-csv)", e)
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted([s.replace(".", "-") for s in t["Symbol"].tolist()])
    except Exception as e:
        log_err("fetch_sp500(wikipedia)", e)
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "PG",
                "UNH", "JNJ", "XOM", "WMT", "MA", "HD", "CVX", "MRK", "ABBV", "KO"]


@st.cache_data(ttl=86400)
def fetch_nasdaq100():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if "ticker" in cols or "symbol" in cols:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                tk = [str(x).replace(".", "-") for x in t[col].dropna() if len(str(x)) <= 6]
                if len(tk) > 50:
                    return sorted(tk)
    except Exception as e:
        log_err("fetch_nasdaq100(wikipedia)", e)
    return sorted(["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AVGO","COST",
        "NFLX","AMD","ADBE","CSCO","QCOM","TXN","AMAT","INTU","ISRG","HON","AMGN"])


@st.cache_data(ttl=86400)
def fetch_set():
    base = ["ADVANC","AOT","AWC","BANPU","BBL","BDMS","BEM","BGRIM","BH","BJC",
            "BTS","CBG","CENTEL","CK","CPALL","CPF","CPN","CRC","DELTA","EA",
            "EGCO","GULF","HANA","HMPRO","INTUCH","IVL","JMT","KBANK","KCE",
            "KKP","KTB","KTC","LH","MAKRO","MBK","MINT","MTC","OR","OSP",
            "PTT","PTTEP","PTTGC","RATCH","SCB","SCC","SCGP","SIRI","SPALI",
            "THAI","TISCO","TOP","TRUE","TU","VGI","WHAUP","WORK"]
    mai = ["2S","ACAP","AMA","BFC","BFIT","CRANE","CSP","DCC","EARTH","EPG",
           "GENCO","HAPPY","HOME","ITEL","JWD","LEO","MASTER","MFEC","KISS"]
    return sorted([f"{t}.BK" for t in base + mai])


@st.cache_data(ttl=86400)
def fetch_etfs():
    return sorted(["XLK","XLV","XLF","XLE","XLI","XLB","XLP","XLU","XLRE","XLC","XLY",
        "QQQ","QQQM","SOXX","SMH","IWM","EEM","GLD","SLV","TLT","HYG","VYM","SCHD"])


SECTOR_MAP = {
    "Technology | เทคโนโลยี":     ["AAPL","MSFT","NVDA","GOOGL","META","AVGO","ORCL","AMD","QCOM","TXN"],
    "Healthcare | สุขภาพ":        ["UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN"],
    "Financials | การเงิน":       ["JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","USB","PNC"],
    "Consumer | สินค้าอุปโภค":    ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","MAR"],
    "Industrials | อุตสาหกรรม":   ["GE","HON","RTX","LMT","BA","CAT","DE","UPS","FDX","UNP"],
    "Energy | พลังงาน":           ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","PXD","HAL"],
    "🤖 AI | ปัญญาประดิษฐ์":       ["NVDA","MSFT","GOOGL","META","AMD","PLTR","SMCI","AVGO","ARM","AI"],
    "🚀 Space | อวกาศ":            ["RKLB","LMT","NOC","BA","RTX","ASTS","KTOS","IRDM"],
    "⚡ EV/Battery | ไฟฟ้า":       ["TSLA","RIVN","LCID","NIO","LI","XPEV","ALB","LTHM"],
    "🔒 Crypto/Cyber | คริปโต":    ["COIN","MSTR","MARA","RIOT","CRWD","PANW","ZS","FTNT"],
}

UNIVERSE_OPTIONS = {
    "S&P 500 (503)": fetch_sp500,
    "Nasdaq 100 (101)": fetch_nasdaq100,
    "หุ้นไทย SET/mai": fetch_set,
    "ETF Screener": fetch_etfs,
    "Sector Focus | เลือกตามหมวด": None,
    "Custom Tickers": None,
}


def resolve_tickers(universe, sector_choice, custom_input):
    if universe == "Custom Tickers":
        return [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    elif universe == "Sector Focus | เลือกตามหมวด":
        tickers_all = []
        for s in sector_choice:
            tickers_all += SECTOR_MAP.get(s, [])
        return sorted(set(tickers_all))
    else:
        fn = UNIVERSE_OPTIONS.get(universe)
        return fn() if fn else []


# ════════════════════════════════════════════════════════
# INDICATORS — v4.0 (เพิ่ม ATR, VWAP, BOS, VCP, Weekly MTF)
# ════════════════════════════════════════════════════════

def wilder_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return np.nan
    d = prices.diff().dropna()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.iloc[:period].mean()
    al = l.iloc[:period].mean()
    a = 1.0 / period
    for i in range(period, len(g)):
        ag = a * g.iloc[i] + (1 - a) * ag
        al = a * l.iloc[i] + (1 - a) * al
    return round(100 - 100 / (1 + ag / al), 2) if al > 0 else 100.0


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd(prices: pd.Series):
    ml = ema(prices, 12) - ema(prices, 26)
    sig = ema(ml, 9)
    return round(ml.iloc[-1], 4), round(sig.iloc[-1], 4), round((ml - sig).iloc[-1], 4)


# ── [ใหม่ v4.0] ATR & Support Zone ─────────────────────────────
def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average True Range — วัดความผันผวนเฉลี่ยต่อแท่ง
    ใช้กำหนดระยะห่างของ Stop Loss และตรวจจับว่าราคาอยู่ใกล้แนวรับหรือไม่
    """
    if len(df) < period + 1:
        return np.nan
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 4)


def detect_support_zone(price: float, e50: float, e200: float, atr: float) -> tuple:
    """
    ตรวจจับว่าราคาปัจจุบันอยู่ใกล้แนวรับ (EMA50 / EMA200) หรือไม่
    ใช้ ATR เป็น buffer zone (0.5 ATR = โซน "ใกล้แนวรับ")
    คืนค่า: (zone_label, stop_loss_price)
    - "🟢 ถึงแนวรับ": ราคาแตะ EMA50/200 พอดี (ห่างไม่เกิน 0.5 ATR)
    - "🟡 ใกล้แนวรับ": ห่างแนวรับ 0.5–1.5 ATR
    - "—": ห่างมาก ไม่ถือว่าใกล้แนวรับ
    Stop Loss = แนวรับที่ใกล้ที่สุด - 1 ATR (ใต้แนวรับเสมอ)
    """
    if np.isnan(atr) or atr <= 0:
        return "—", np.nan
    supports = []
    if e50 > 0:
        supports.append(("EMA50", e50))
    if e200 > 0:
        supports.append(("EMA200", e200))
    if not supports:
        return "—", np.nan

    best_label, best_dist, best_level = "—", float("inf"), np.nan
    for lbl, lvl in supports:
        dist = abs(price - lvl)
        if dist < best_dist:
            best_dist = dist
            best_label = lbl
            best_level = lvl

    ratio = best_dist / atr
    stop_loss = round(best_level - atr, 2)
    if ratio <= 0.5:
        return f"🟢 ถึง {best_label}", stop_loss
    elif ratio <= 1.5:
        return f"🟡 ใกล้ {best_label}", stop_loss
    return "—", stop_loss


# ── [ใหม่ v4.0] VWAP Rolling ────────────────────────────────────
def find_vwap(df: pd.DataFrame, period: int = 20) -> float:
    """
    VWAP แบบ rolling window (ไม่ใช่ intraday VWAP จริง เพราะใช้ข้อมูล daily)
    = ต้นทุนเฉลี่ยถ่วงน้ำหนัก Volume ใน N วันล่าสุด
    ใช้แทน VWAP แบบ intraday สำหรับกรอบการวิเคราะห์รายวัน
    ราคาเหนือ VWAP = ผู้ซื้อในช่วงนี้กำไรโดยเฉลี่ย (momentum เป็นบวก)
    """
    if len(df) < period:
        return np.nan
    sub = df.iloc[-period:]
    typical = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    vol = sub["Volume"]
    if vol.sum() == 0:
        return np.nan
    return round(float((typical * vol).sum() / vol.sum()), 2)


# ── [ใหม่ v4.0] Break of Structure ─────────────────────────────
def detect_break_of_structure(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    หา Swing High ล่าสุดใน lookback แท่ง แล้วเช็คว่าราคาปัจจุบัน
    เบรคขึ้นเหนือ Swing High นั้นหรือยัง
    คืนค่า: (label, swing_high_price)
    - "🚨 BOS Breakout": ราคาเพิ่งเบรค Swing High (2 แท่งล่าสุด)
    - "⏳ Near BOS": ราคาอยู่ภายใน 1% ของ Swing High
    - "—": ยังห่างอยู่
    หมายเหตุ: ใช้ close เท่านั้น (ไม่ใช้ high) เพื่อลด false signal
    """
    if len(df) < lookback + 2:
        return "—", np.nan
    cl = df["Close"]
    window = cl.iloc[-(lookback + 1):-1]  # ไม่รวมแท่งปัจจุบัน (no lookahead)
    swing_high = float(window.max())
    current = float(cl.iloc[-1])
    prev = float(cl.iloc[-2])

    if prev < swing_high <= current:
        return "🚨 BOS Breakout", round(swing_high, 2)
    elif current >= swing_high * 0.99:
        return "⏳ Near BOS", round(swing_high, 2)
    return "—", round(swing_high, 2)


# ── [ใหม่ v4.0] VCP Volatility Contraction Pattern ─────────────
def detect_vcp(closes: pd.Series, volumes: pd.Series) -> tuple:
    """
    VCP (Volatility Contraction Pattern) ตาม Mark Minervini:
    ราคาแกว่งแคบลงเป็นขั้นบันได + Volume แห้งลงในช่วง contraction
    ตรวจสอบโดยเปรียบเทียบ range ของ 3 ช่วง (แต่ละช่วง 5 แท่ง)
    และ Volume เฉลี่ยของแต่ละช่วง
    คืนค่า: (label, score 0-3)
    - score 3: VCP ชัดเจน (ทั้ง range และ volume แคบลงทั้งหมด)
    - score 2: VCP บางส่วน
    - score 0-1: ไม่ใช่ VCP
    """
    if len(closes) < 20 or len(volumes) < 20:
        return "—", 0

    def _range(s): return float(s.max() - s.min())
    def _vol(v): return float(v.mean())

    r3 = _range(closes.iloc[-15:-10])
    r2 = _range(closes.iloc[-10:-5])
    r1 = _range(closes.iloc[-5:])
    v3 = _vol(volumes.iloc[-15:-10])
    v2 = _vol(volumes.iloc[-10:-5])
    v1 = _vol(volumes.iloc[-5:])

    range_contracting = (r1 < r2 < r3) if r3 > 0 else False
    vol_contracting = (v1 < v2 < v3) if v3 > 0 else False

    partial_range = (r1 < r3) if r3 > 0 else False
    partial_vol = (v1 < v3) if v3 > 0 else False

    if range_contracting and vol_contracting:
        return "🎯 VCP Clear", 3
    elif range_contracting or (partial_range and vol_contracting):
        return "📐 VCP Partial", 2
    elif partial_range and partial_vol:
        return "🔍 VCP Watch", 1
    return "—", 0


# ── [ใหม่ v4.0] Weekly EMA40 Alignment ─────────────────────────
@retry(times=3, base_delay=0.6)
def _download_weekly(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="2y", interval="1wk", auto_adjust=True)


@st.cache_data(ttl=86400)
def get_weekly_ema40(ticker: str) -> tuple:
    """
    ดึงข้อมูลรายสัปดาห์แล้วคำนวณ EMA40 (สัปดาห์) ≈ EMA200 วัน
    คืน (price_vs_ema40_label, weekly_ema40_value)
    การยืนเหนือ EMA40 สัปดาห์ = เทรนด์ใหญ่ขาขึ้น (เงื่อนไขสำคัญ)
    """
    try:
        df = _download_weekly(ticker)
        if df is None or len(df) < 42:
            return "—", np.nan
        cl = df["Close"]
        e40w = ema(cl, 40).iloc[-1]
        px = cl.iloc[-1]
        if pd.isna(e40w) or e40w <= 0:
            return "—", np.nan
        pct = round((px - e40w) / e40w * 100, 2)
        if pct > 0:
            return f"✅ เหนือ EMA40W (+{pct:.1f}%)", round(float(e40w), 2)
        return f"❌ ต่ำกว่า EMA40W ({pct:.1f}%)", round(float(e40w), 2)
    except Exception as e:
        log_err(f"get_weekly_ema40({ticker})", e)
        return "—", np.nan


# ── Existing indicators (unchanged from v3.x) ───────────────────

def candle_pattern(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "—"
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c.Close - c.Open)
    rng = c.High - c.Low
    if rng == 0:
        return "—"
    lo_sh = min(c.Close, c.Open) - c.Low
    up_sh = c.High - max(c.Close, c.Open)
    if body / rng < 0.10:
        return "🕯 Doji"
    if lo_sh >= 2 * body and up_sh < body * 0.5:
        return "🔨 Hammer"
    if c.Close > c.Open and p.Close < p.Open and c.Open < p.Close and c.Close > p.Open:
        return "🟢 Engulfing"
    return "—"


def ema_pattern(price, e5, e10, e20, e50, e100, e200) -> tuple:
    vals = [price, e5, e10, e20, e50, e100, e200]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return "—", 0
    parts = []
    score = 0
    if price > e5 > e10 > e20 > e50 > e100 > e200:
        parts.append("🏆 Perfect Uptrend"); score = 5
    elif price > e5 > e10 > e20 > e50 > e200:
        parts.append("📈 Strong Uptrend"); score = 4
    elif e20 > e50 > e200 and price > e20:
        parts.append("✨ Golden Align"); score = 3
    sp = (max(e20, e50, e200) - min(e20, e50, e200)) / e200 * 100
    if sp < 2.5 and price > e200:
        parts.append("🔥 Squeeze"); score = max(score, 4)
    elif sp < 4.0 and price > e200:
        parts.append("⚡ Pre-Squeeze"); score = max(score, 2)
    if e200 < price < e50 and price > e20:
        parts.append("🌱 Early Break"); score = max(score, 3)
    fan = (e5 - e200) / e200 * 100 if e200 > 0 else 0
    if fan > 8 and price > e5 and e5 > e50:
        parts.append("🎯 EMA Fan"); score = max(score, 2)
    if not parts:
        return ("❌ Below EMA200", 0) if price < e200 else ("🔄 Mixed", 1)
    return " · ".join(parts), min(score, 5)


def squeeze_direction(closes: pd.Series) -> tuple:
    if len(closes) < 206:
        return "—", np.nan, np.nan
    e20, e50, e200 = ema(closes, 20), ema(closes, 50), ema(closes, 200)
    def bw(i):
        hi = max(e20.iloc[i], e50.iloc[i], e200.iloc[i])
        lo = min(e20.iloc[i], e50.iloc[i], e200.iloc[i])
        return (hi - lo) / e200.iloc[i] * 100 if e200.iloc[i] > 0 else np.nan
    bw0, bw5 = bw(-1), bw(-6)
    if np.isnan(bw0) or np.isnan(bw5):
        return "—", np.nan, np.nan
    delta = round(bw0 - bw5, 3)
    lbl = ("🔥 Squeezing" if delta < -0.4 else "⚡ Tightening" if delta < 0
           else "🌱 Just Broke" if delta < 0.6 else "📈 Expanding")
    return lbl, round(bw0, 2), delta


def signal_age(closes: pd.Series) -> int:
    if len(closes) < 202:
        return -1
    e200 = ema(closes, 200)
    for i in range(1, min(31, len(closes) - 1)):
        if closes.iloc[-i - 1] < e200.iloc[-i - 1] and closes.iloc[-i] > e200.iloc[-i]:
            return i - 1
    return -1


def quiet_accumulation(volumes: pd.Series, closes: pd.Series, rsi: float, n: int = 10) -> tuple:
    if len(volumes) < n or len(closes) < n:
        return 0, "—"
    rv = volumes.iloc[-n:]
    rc = closes.iloc[-n:]
    slope = np.polyfit(range(n), rv.values, 1)[0] > 0
    ranges = [abs(rc.iloc[i] - rc.iloc[i - 1]) / rc.iloc[i - 1] * 100 for i in range(1, n)]
    low_v = np.mean(ranges) < 2.5
    rsi_ok = not np.isnan(rsi) and rsi < 62
    va = volumes.iloc[-30:].mean() if len(volumes) >= 30 else volumes.mean()
    vr = volumes.iloc[-1] / va if va > 0 else 0
    sweet = 1.05 < vr < 2.5
    e20s = np.polyfit(range(5), ema(closes, 20).iloc[-5:].values, 1)[0] > 0 if len(closes) >= 20 else False
    score = sum([slope, low_v, rsi_ok, sweet, e20s])
    lbl = {5: "🔬 Stealth Accum", 4: "📦 Quiet Accum", 3: "🔍 Possible Accum", 2: "👀 Watch", 1: "—", 0: "—"}[score]
    return score, lbl


def relative_strength(closes: pd.Series, bench: pd.Series, period: int = 20) -> float:
    if closes is None or bench is None or len(closes) < 2 or len(bench) < 2:
        return np.nan
    try:
        s = to_date_indexed(closes).rename("s")
        b = to_date_indexed(bench).rename("b")
        aligned = pd.concat([s, b], axis=1).dropna()
        if len(aligned) < period + 1:
            return np.nan
        sr = (aligned["s"].iloc[-1] - aligned["s"].iloc[-period]) / aligned["s"].iloc[-period] * 100
        br = (aligned["b"].iloc[-1] - aligned["b"].iloc[-period]) / aligned["b"].iloc[-period] * 100
        return round(sr - br, 2)
    except Exception as e:
        log_err("relative_strength", e)
        return np.nan


def gem_score(pat_score, acc_score, vol20, rsi, drawdown, mktcap_b) -> tuple:
    s = min(pat_score, 4) + min(acc_score, 3)
    if 1.1 <= vol20 <= 2.0: s += 1
    if 40 <= rsi <= 62: s += 1
    if isinstance(mktcap_b, float) and 0 < mktcap_b < 10: s += 1
    s = min(s, 10)
    lbl = "💎 Hidden Gem" if s >= 8 else "🔭 Emerging Gem" if s >= 6 else "👀 Watch" if s >= 4 else "—"
    return s, lbl


def conservative_stars(price, e200, rsi, vol20, drawdown) -> str:
    s = 0
    if e200 > 0 and abs((price - e200) / e200 * 100) <= 2: s += 1
    if rsi < 35: s += 1
    if vol20 > 2.0: s += 1
    if -15 <= drawdown <= -5: s += 1
    return "⭐" * s if s else "—"


# ── [อัปเกรด v4.0] God-Tier Strategy Signal ─────────────────────
def strategy_signal_v4(
    price: float, e200: float, e50: float,
    rsi: float, vol20: float, macd_h: float, stars: str,
    rs20: float = np.nan,                 # ข้อ 6: RS บังคับ
    zone_label: str = "—",               # ข้อ 1: Support Zone
    bos_label: str = "—",               # ข้อ 3: Break of Structure
    vcp_score: int = 0,                  # ข้อ 4: VCP
    weekly_ok: bool = True,              # ข้อ 5: Weekly EMA40
    stop_loss: float = np.nan,          # ข้อ 1: Stop Loss price
    atr: float = np.nan,                # ข้อ 1: ATR
) -> tuple:
    """
    v4.0: รวมสัญญาณจากทุกระบบ คืน (signal_label, reason, stop_loss)
    ลำดับความสำคัญ (จากสูงสุด):
      🎯 Institutional Breakout — BOS เบรค + Volume + RS บวก + Weekly OK
      🔥 Smart Money Accum     — ถึงแนวรับ + VCP + Volume + RS บวก
      🚀 Breakout              — Volume + เหนือ EMA50/200 + RS บวก
      🔥 Strong Buy            — RSI ต่ำ + Volume + MACD บวก + ใกล้ EMA200
      📈 ขาขึ้น                — EMA เรียงดี
      ⚠️ เฝ้าระวัง             — ใกล้ EMA200 แต่ MACD ลบ
      ⏳ รอ Pullback           — RSI สูงมาก
      ❌ ขาลง / ⚠️ Oversold Bear
    """
    rs_ok = (not np.isnan(rs20)) and rs20 > 0
    p200 = (price - e200) / e200 * 100 if e200 > 0 else 999
    at_support = "🟢" in zone_label
    near_support = "🟡" in zone_label
    bos_active = "BOS" in bos_label or "Near BOS" in bos_label

    stop_str = f" | SL: ${stop_loss:.2f}" if not np.isnan(stop_loss) else ""

    # 🎯 Institutional Breakout — ต้องครบทุกเงื่อนไข
    if ("Breakout" in bos_label and vol20 > 1.5 and rs_ok
            and weekly_ok and price > e50 > e200 and rsi < 80):
        return (
            "🎯 Institutional Breakout",
            f"BOS เบรค Swing High + Volume ({vol20:.1f}x) + RS ชนะตลาด (+{rs20:.1f}%) "
            f"+ Weekly EMA40 ผ่าน{stop_str}",
            stop_loss
        )

    # 🔥 Smart Money Accumulation — ถึงแนวรับ + VCP
    if (at_support and vcp_score >= 2 and vol20 > 1.2 and rs_ok
            and weekly_ok and rsi < 65):
        return (
            "🔥 Smart Money Accum",
            f"ถึงแนวรับ ({zone_label}) + VCP score {vcp_score}/3 + RS +{rs20:.1f}%{stop_str}",
            stop_loss
        )

    # 🚀 Breakout — Volume + เหนือ EMA + RS บวก
    if vol20 > 2.0 and price > e50 > e200 and macd_h > 0 and rs_ok and 50 <= rsi <= 75:
        return (
            "🚀 Breakout",
            f"Volume พุ่ง ({vol20:.1f}x) + ราคา>EMA50>EMA200 + RS +{rs20:.1f}%{stop_str}",
            stop_loss
        )

    # 🔥 Strong Buy — เดิม แต่เพิ่ม RS check
    if (len(stars) >= 3 and rsi < 40 and vol20 > 1.8 and macd_h > 0
            and -5 <= p200 <= 3 and rs_ok):
        return (
            "🔥 Strong Buy",
            f"RSI ต่ำ ({rsi:.0f}) + Volume ({vol20:.1f}x) + MACD บวก + RS +{rs20:.1f}%{stop_str}",
            stop_loss
        )

    # Strong Buy แม้ RS ลบ แต่ให้แจ้งเตือน
    if len(stars) >= 3 and rsi < 40 and vol20 > 1.8 and macd_h > 0 and -5 <= p200 <= 3:
        return (
            "🔥 Strong Buy ⚠️RS-",
            f"RSI ต่ำ ({rsi:.0f}) + Volume ({vol20:.1f}x) แต่ RS แพ้ตลาด ({rs20:.1f}%){stop_str}",
            stop_loss
        )

    # 📈 ขาขึ้น
    if price > e50 > e200 and 40 <= rsi <= 70:
        rs_note = f" (RS {rs20:+.1f}%)" if not np.isnan(rs20) else ""
        return (
            "📈 ขาขึ้น",
            f"ราคา>EMA50>EMA200 + RSI {rsi:.0f}{rs_note}",
            stop_loss
        )

    # ⚠️ เฝ้าระวัง
    if abs(p200) <= 3 and rsi < 50 and macd_h < 0:
        return (
            "⚠️ เฝ้าระวัง",
            f"ใกล้ EMA200 แต่ MACD ลบ RSI {rsi:.0f} — ทิศทางไม่ชัด",
            stop_loss
        )

    # ⏳ รอ Pullback
    if rsi > 75:
        return "⏳ รอ Pullback", f"RSI สูง ({rsi:.0f}) เสี่ยงไล่ราคา", stop_loss

    # ❌ ขาลง
    if price < e200:
        if rsi < 30:
            return "⚠️ Oversold Bear", f"ราคา<EMA200 RSI={rsi:.0f} oversold แต่เทรนด์ลง", stop_loss
        return "❌ ขาลง", "ราคาต่ำกว่า EMA200 — เทรนด์หลักขาลง", stop_loss

    return "🔄 Neutral", "ไม่เข้าเงื่อนไขชัดเจน", stop_loss


# ════════════════════════════════════════════════════════
# ANALYZER — v4.0
# ════════════════════════════════════════════════════════

@retry(times=3, base_delay=0.6)
def _download_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)


@st.cache_data(ttl=3600)
def _cached_history(ticker, period, interval):
    try:
        df = _download_history(ticker, period, interval)
        return None if (df is None or df.empty) else df
    except Exception as e:
        log_err(f"history({ticker})", e)
        return None


def _normalize_dividend_yield(raw):
    if raw is None:
        return np.nan
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return np.nan
    if v <= 0:
        return 0.0
    return round(v * 100, 2) if v < 1 else round(v, 2)


@retry(times=3, base_delay=0.6)
def _download_info(ticker):
    return yf.Ticker(ticker).info or {}


def _safe_num(val, decimals=2):
    if val is None:
        return np.nan
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return np.nan


@st.cache_data(ttl=21600)
def _cached_fundamentals(ticker):
    try:
        info = _download_info(ticker)
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        mktcap = info.get("marketCap")
        mktcap_b = (mktcap / 1e9) if isinstance(mktcap, (int, float)) else np.nan

        # ── [ใหม่ v4.0] Fundamental Confluence (ข้อ 11) ──────────
        eps_growth = info.get("earningsGrowth")       # yfinance field (YoY)
        rev_growth = info.get("revenueGrowth")        # yfinance field (YoY)
        eps_g = _safe_num(eps_growth, 4)
        rev_g = _safe_num(rev_growth, 4)
        growth_badge = ""
        if (not np.isnan(eps_g) and eps_g > 0) or (not np.isnan(rev_g) and rev_g > 0):
            growth_badge = "🌟 Growth Stock"

        return {
            "pe": _safe_num(pe),
            "pb": _safe_num(pb),
            "div": _normalize_dividend_yield(info.get("dividendYield")),
            "mktcap_b": _safe_num(mktcap_b),
            "eps_growth_pct": round(eps_g * 100, 1) if not np.isnan(eps_g) else np.nan,
            "rev_growth_pct": round(rev_g * 100, 1) if not np.isnan(rev_g) else np.nan,
            "growth_badge": growth_badge,
        }
    except Exception as e:
        log_err(f"fundamentals({ticker})", e)
        return {"pe": np.nan, "pb": np.nan, "div": np.nan, "mktcap_b": np.nan,
                "eps_growth_pct": np.nan, "rev_growth_pct": np.nan, "growth_badge": ""}


@st.cache_data(ttl=3600)
def analyze(ticker, period="1y", interval="1d", bench_tuple=None,
            include_weekly=False) -> Optional[dict]:
    """
    v4.0: เพิ่ม ATR, VWAP, BOS, VCP, Weekly MTF, God-Tier Signal, Growth Badge
    include_weekly=True → ดึง weekly data เพิ่ม (ช้าขึ้น ~0.5s/หุ้น)
    """
    try:
        df = _cached_history(ticker, period, interval)
        if df is None or len(df) < 30:
            return None
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        cl = df["Close"]
        vl = df["Volume"]
        px = cl.iloc[-1]

        if pd.isna(px) or px <= 0:
            log_err(f"analyze({ticker})", ValueError(f"ราคาผิดปกติ: {px}"))
            return None

        ep = {n: ema(cl, n).iloc[-1] for n in [5, 10, 20, 50, 100, 200]}
        ed = {n: round((px - v) / v * 100, 2) if v > 0 else np.nan for n, v in ep.items()}

        rsi_val = wilder_rsi(cl)
        ml, ms, mh = macd(cl)
        v20a = vl.iloc[-20:].mean() if len(vl) >= 20 else vl.mean()
        v3ma = vl.iloc[-63:].mean() if len(vl) >= 63 else vl.mean()
        v6ma = vl.iloc[-126:].mean() if len(vl) >= 126 else vl.mean()
        vc = vl.iloc[-1]
        vm20 = round(vc / v20a, 2) if v20a > 0 else np.nan
        vm3m = round(vc / v3ma, 2) if v3ma > 0 else np.nan
        vm6m = round(vc / v6ma, 2) if v6ma > 0 else np.nan

        hi52 = cl.rolling(min(252, len(cl))).max().iloc[-1]
        draw = round((px - hi52) / hi52 * 100, 2) if hi52 > 0 else np.nan
        prev_c = round(cl.iloc[-2], 2) if len(cl) >= 2 else px
        ytd_start = cl[cl.index.year == datetime.date.today().year]
        base0 = ytd_start.iloc[0] if len(ytd_start) > 1 else cl.iloc[0]
        ytd_ret = round((px - base0) / base0 * 100, 2)

        # ── v4.0 new indicators ──────────────────────────
        atr_val = calculate_atr(df)
        zone_lbl, stop_loss = detect_support_zone(px, ep[50], ep[200], atr_val)
        vwap_val = find_vwap(df)
        vwap_diff = round((px - vwap_val) / vwap_val * 100, 2) if vwap_val and vwap_val > 0 else np.nan
        bos_lbl, swing_high = detect_break_of_structure(df)
        vcp_lbl, vcp_sc = detect_vcp(cl, vl)

        # Weekly EMA40 (ข้อ 5)
        weekly_ok = True  # default ถ้าไม่ได้ดึง
        weekly_lbl = "—"
        weekly_e40 = np.nan
        if include_weekly:
            weekly_lbl, weekly_e40 = get_weekly_ema40(ticker)
            weekly_ok = "✅" in weekly_lbl

        rs20 = rs50 = np.nan
        if bench_tuple:
            dates, vals = zip(*bench_tuple)
            bench = pd.Series(vals, index=pd.to_datetime(dates))
            rs20 = relative_strength(cl, bench, 20)
            rs50 = relative_strength(cl, bench, 50)

        trend = "🟢 Bull" if px > ep[200] else "🔴 Bear"
        patt = candle_pattern(df)
        stars = conservative_stars(px, ep[200], rsi_val, vm20 or 0, draw or 0)
        ep_lbl, ep_sc = ema_pattern(px, ep[5], ep[10], ep[20], ep[50], ep[100], ep[200])
        acc_sc, acc_lb = quiet_accumulation(vl, cl, rsi_val)
        sq_lbl, bw_now, bw_delta = squeeze_direction(cl)
        age = signal_age(cl)

        # God-Tier Signal (ข้อ 7)
        sig, sig_reason, sig_stop = strategy_signal_v4(
            price=px, e200=ep[200], e50=ep[50],
            rsi=rsi_val, vol20=vm20 or 0, macd_h=mh, stars=stars,
            rs20=rs20, zone_label=zone_lbl, bos_label=bos_lbl,
            vcp_score=vcp_sc, weekly_ok=weekly_ok, stop_loss=stop_loss, atr=atr_val,
        )

        fnd = _cached_fundamentals(ticker)
        gs, gl = gem_score(ep_sc, acc_sc, vm20 or 0, rsi_val, draw or 0, fnd["mktcap_b"])

        return {
            "Ticker": ticker, "Price": round(px, 2), "ราคาปิด": prev_c,
            "Trend": trend, "Signal": sig, "Signal Reason": sig_reason,
            "Stop Loss": round(sig_stop, 2) if not np.isnan(sig_stop or np.nan) else np.nan,
            "ATR": round(atr_val, 2) if not np.isnan(atr_val) else np.nan,
            "Phase": ep_lbl, "Stars": stars,
            "EMA5": round(ep[5], 2), "EMA10": round(ep[10], 2), "EMA20": round(ep[20], 2),
            "EMA50": round(ep[50], 2), "EMA100": round(ep[100], 2), "EMA200": round(ep[200], 2),
            "vs EMA5%": ed[5], "vs EMA10%": ed[10], "vs EMA20%": ed[20],
            "vs EMA50%": ed[50], "vs EMA100%": ed[100], "vs EMA200%": ed[200],
            "VWAP": vwap_val, "vs VWAP%": vwap_diff,
            "Support Zone": zone_lbl, "BOS": bos_lbl, "Swing High": round(swing_high, 2) if not np.isnan(swing_high) else np.nan,
            "VCP": vcp_lbl, "VCP Score": vcp_sc,
            "Weekly EMA40": weekly_lbl,
            "RSI": rsi_val, "MACD": ml, "Signal_L": ms, "MACD_H": mh,
            "Vol×20D": vm20, "Vol×3M": vm3m, "Vol×6M": vm6m,
            "YTD%": ytd_ret, "Drawdown%": draw, "High52W": round(hi52, 2),
            "vs52W%": round((px - hi52) / hi52 * 100, 2) if hi52 > 0 else np.nan,
            "Candle": patt, "EMA Pattern": ep_lbl, "Pat Score": ep_sc,
            "Accum": acc_lb, "Accum Score": acc_sc, "Gem Score": gs, "💎 Gem": gl,
            "Squeeze": sq_lbl, "BW%": bw_now, "BW Δ5d": bw_delta, "Signal Age": age,
            "RS 20D": rs20, "RS 50D": rs50,
            "P/E": fnd["pe"], "P/BV": fnd["pb"], "Div%": fnd["div"], "MktCap$B": fnd["mktcap_b"],
            "EPS Growth%": fnd["eps_growth_pct"], "Rev Growth%": fnd["rev_growth_pct"],
            "🌟 Growth": fnd["growth_badge"],
        }
    except Exception as e:
        log_err(f"analyze({ticker})", e)
        return None


def make_bench_tuple(bench_df):
    idx = pd.to_datetime(bench_df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return tuple(zip(idx.strftime("%Y-%m-%d"), bench_df["Close"].values.tolist()))


def batch_scan(tickers, period="1y", interval="1d", bench_tuple=None,
               max_workers=6, include_weekly=False,
               progress_cb: Optional[Callable] = None):
    results = []
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(analyze, tk, period, interval, bench_tuple, include_weekly): tk
            for tk in tickers
        }
        for fut in as_completed(futures):
            done += 1
            try:
                d = fut.result()
                if d:
                    results.append(d)
            except Exception as e:
                log_err(f"batch_scan({futures[fut]})", e)
            if progress_cb:
                progress_cb(done, total)
    return pd.DataFrame(results) if results else pd.DataFrame()


def fetch_live(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        px = getattr(fi, "last_price", None)
        pc = getattr(fi, "previous_close", None)
        chg = round((px - pc) / pc * 100, 2) if px and pc else None
        return {
            "price": round(px, 2) if px else "N/A",
            "change": chg,
            "high": round(getattr(fi, "day_high", 0) or 0, 2),
            "low": round(getattr(fi, "day_low", 0) or 0, 2),
            "vol": f"{int(getattr(fi, 'last_volume', 0) or 0):,}",
            "cap": f"${(getattr(fi, 'market_cap', 0) or 0) / 1e9:.1f}B",
        }
    except Exception as e:
        log_err(f"fetch_live({ticker})", e)
        return {}


# ════════════════════════════════════════════════════════
# [ใหม่ v4.0] MARKET REGIME FILTER (ข้อ 10)
# ════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_market_regime(benchmark_ticker: str = "SPY") -> dict:
    """
    ตรวจสอบ Market Regime โดยดูว่า benchmark (SPY หรือ SET50.BK)
    อยู่เหนือ EMA200 หรือไม่ คืนค่า dict พร้อม label และสี
    ใช้แสดงคำเตือนบน Dashboard ก่อนสัญญาณหุ้นทุกตัว
    """
    try:
        df = yf.Ticker(benchmark_ticker).history(period="1y", interval="1d", auto_adjust=True)
        if df is None or len(df) < 201:
            return {"label": "—", "color": "#8b949e", "warning": False, "ticker": benchmark_ticker}
        cl = df["Close"]
        e200 = ema(cl, 200).iloc[-1]
        px = cl.iloc[-1]
        pct = round((px - e200) / e200 * 100, 2) if e200 > 0 else 0
        above = px > e200
        return {
            "label": f"{'✅ Bull' if above else '⚠️ Bear'} — {benchmark_ticker} {'เหนือ' if above else 'ต่ำกว่า'} EMA200 ({pct:+.1f}%)",
            "color": "#3fb950" if above else "#f85149",
            "warning": not above,
            "ticker": benchmark_ticker,
            "pct": pct,
        }
    except Exception as e:
        log_err(f"get_market_regime({benchmark_ticker})", e)
        return {"label": "—", "color": "#8b949e", "warning": False, "ticker": benchmark_ticker}


# ════════════════════════════════════════════════════════
# BACKTESTER (unchanged from v3.x)
# ════════════════════════════════════════════════════════

BACKTEST_NOTES = (
    "ไม่หักค่าคอมมิชชั่น/สเปรด · survivorship bias · Sharpe ประมาณจาก trade returns "
    "· ผลย้อนหลังไม่ใช่การันตีอนาคต ไม่ใช่คำแนะนำการลงทุน"
)


@retry(times=3, base_delay=0.6)
def _download_2y(ticker):
    return yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)


@st.cache_data(ttl=86400)
def backtest(ticker, hold_days=20):
    try:
        df = _download_2y(ticker)
        if df is None or len(df) < 220:
            return {"error": "ข้อมูลไม่พอ (ต้องการ 2 ปี)"}
        cl = df["Close"]
        op = df["Open"]
        e20, e50, e200 = ema(cl, 20), ema(cl, 50), ema(cl, 200)
        trades = []
        in_trade = False
        entry_price = 0.0
        entry_i = 0
        upper = len(cl) - hold_days - 2
        for i in range(200, max(200, upper)):
            hi = max(e20.iloc[i], e50.iloc[i], e200.iloc[i])
            lo = min(e20.iloc[i], e50.iloc[i], e200.iloc[i])
            bw = (hi - lo) / e200.iloc[i] * 100 if e200.iloc[i] > 0 else np.nan
            if not in_trade and bw < 3.0 and cl.iloc[i] > e200.iloc[i]:
                entry_price = op.iloc[i + 1]
                entry_i = i + 1
                in_trade = True
            elif in_trade and (i - entry_i) >= hold_days:
                exit_price = cl.iloc[i]
                trades.append({
                    "ret": round((exit_price - entry_price) / entry_price * 100, 2),
                    "entry_date": str(cl.index[entry_i].date()),
                    "exit_date": str(cl.index[i].date()),
                })
                in_trade = False
        bh_ret = round((cl.iloc[-1] - cl.iloc[200]) / cl.iloc[200] * 100, 2)
        if not trades:
            return {"n": 0, "win_rate": 0, "avg": 0, "best": 0, "worst": 0, "trades": [],
                    "buy_hold_ret": bh_ret, "max_drawdown": 0, "sharpe": None, "notes": BACKTEST_NOTES}
        rets = [t["ret"] for t in trades]
        wins = [r for r in rets if r > 0]
        equity = np.array([1.0])
        for r in rets:
            equity = np.append(equity, equity[-1] * (1 + r / 100))
        running_max = np.maximum.accumulate(equity)
        max_dd = round(float(((equity - running_max) / running_max * 100).min()), 2)
        ann_factor = 252 / hold_days if hold_days > 0 else 1
        mean_r, std_r = float(np.mean(rets)), float(np.std(rets))
        sharpe = round((mean_r / std_r) * np.sqrt(ann_factor), 2) if std_r > 0 else None
        return {
            "n": len(trades), "win_rate": round(len(wins) / len(trades) * 100, 1),
            "avg": round(mean_r, 2), "median": round(float(np.median(rets)), 2),
            "best": round(max(rets), 2), "worst": round(min(rets), 2),
            "trades": rets, "trade_details": trades,
            "buy_hold_ret": bh_ret,
            "strategy_compound_ret": round((equity[-1] - 1) * 100, 2),
            "max_drawdown": max_dd, "sharpe": sharpe, "notes": BACKTEST_NOTES,
        }
    except Exception as e:
        log_err(f"backtest({ticker})", e)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════
# PREFETCH (GitHub Release pattern — unchanged from v3.5)
# ════════════════════════════════════════════════════════

GITHUB_REPO = "bigpk2002/BANNVICH01"
RELEASE_TAG = "latest-data"
PREFETCH_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/latest_scan.json"
ALERTS_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/alerts.json"
PREFETCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "latest_scan.json")
ALERTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alerts.json")


@st.cache_data(ttl=300)
def load_prefetched_bundle():
    if os.path.exists(PREFETCH_PATH):
        try:
            with open(PREFETCH_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("generated_at"), pd.DataFrame(payload.get("data", []))
        except Exception as e:
            log_err("load_prefetched_bundle(local)", e)
    try:
        import requests
        resp = requests.get(PREFETCH_URL, timeout=15)
        if resp.ok:
            payload = resp.json()
            return payload.get("generated_at"), pd.DataFrame(payload.get("data", []))
    except Exception as e:
        log_err("load_prefetched_bundle(release)", e)
    return None, pd.DataFrame()


@st.cache_data(ttl=300)
def load_prefetch_alerts():
    if os.path.exists(ALERTS_PATH):
        try:
            with open(ALERTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("new_signals", [])
        except Exception as e:
            log_err("load_prefetch_alerts(local)", e)
    try:
        import requests
        resp = requests.get(ALERTS_URL, timeout=15)
        if resp.ok:
            return resp.json().get("new_signals", [])
    except Exception as e:
        log_err("load_prefetch_alerts(release)", e)
    return []


def get_with_bundle_fallback(tickers, bundle_df, max_live_fallback=15):
    if bundle_df is None or bundle_df.empty or "Ticker" not in bundle_df.columns:
        have = pd.DataFrame()
        missing = list(tickers)
    else:
        have = bundle_df[bundle_df["Ticker"].isin(tickers)].copy()
        found = set(have["Ticker"].tolist())
        missing = [t for t in tickers if t not in found]
    if missing and len(missing) <= max_live_fallback:
        extra_rows = [analyze(tk) for tk in missing]
        extra_rows = [r for r in extra_rows if r]
        if extra_rows:
            have = pd.concat([have, pd.DataFrame(extra_rows)], ignore_index=True) if not have.empty else pd.DataFrame(extra_rows)
    return have


# ════════════════════════════════════════════════════════
# ALERTS
# ════════════════════════════════════════════════════════

NOTABLE_SIGNALS = ("🎯 Institutional Breakout", "🔥 Smart Money Accum",
                   "🔥 Strong Buy", "🚀 Breakout")


def detect_new_signals(current_df, last_signals):
    if current_df is None or current_df.empty or "Signal" not in current_df.columns:
        return []
    new_hits = []
    for _, row in current_df.iterrows():
        tk, sig = row.get("Ticker"), row.get("Signal")
        if any(s in str(sig) for s in NOTABLE_SIGNALS) and last_signals.get(tk) != sig:
            new_hits.append({"ticker": tk, "signal": sig})
    return new_hits


def signals_snapshot(df):
    if df is None or df.empty or "Signal" not in df.columns:
        return {}
    return dict(zip(df["Ticker"], df["Signal"]))


def maybe_notify_telegram(message):
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        return False
    if not token or not chat_id:
        return False
    try:
        import requests
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             data={"chat_id": chat_id, "text": message}, timeout=8)
        return resp.ok
    except Exception as e:
        log_err("maybe_notify_telegram", e)
        return False


# ════════════════════════════════════════════════════════
# STYLES & UI HELPERS
# ════════════════════════════════════════════════════════

CSS_BLOCK = """
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background:#0d1117 !important; }
.main .block-container { padding: 1rem 1.2rem 2rem 1.2rem !important; max-width:100% !important; }

p, span, div, label, li, td, th { color:#e6edf3 !important; }
h1,h2,h3,h4,h5,h6 { color:#ffffff !important; font-weight:700 !important; }
strong, b { color:#ffffff !important; }
small, .stCaption p { color:#8b949e !important; font-size:0.78rem !important; }
code { color:#79c0ff !important; background:#161b22 !important; padding:1px 5px !important; border-radius:4px !important; }

/* ── METRIC CARDS ── */
div[data-testid="metric-container"] {
    background:#161b22 !important; border:1px solid #30363d !important;
    border-radius:10px !important; padding:12px 16px !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] span {
    color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;
    font-size:1.45rem !important; font-weight:800 !important;
}
[data-testid="stMetricLabel"] p { color:#8b949e !important; font-size:0.72rem !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:#161b22 !important; border-radius:8px !important; padding:4px !important;
}
.stTabs [data-baseweb="tab"] {
    color:#8b949e !important; font-weight:600 !important; font-size:0.82rem !important;
    border-radius:6px !important; padding:6px 12px !important; background:transparent !important;
}
.stTabs [aria-selected="true"] { background:#238636 !important; color:#ffffff !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background:#161b22 !important; border-right:1px solid #21262d !important;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div { color:#e6edf3 !important; }

/* ── INPUTS ── */
.stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div {
    background:#21262d !important; border-color:#30363d !important;
}
.stTextArea textarea, .stTextInput input {
    background:#21262d !important; color:#e6edf3 !important; border-color:#30363d !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background:linear-gradient(135deg,#238636,#2ea043) !important;
    color:#ffffff !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important; font-size:0.88rem !important; padding:9px 18px !important;
    width:100% !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#2ea043,#3fb950) !important;
    box-shadow:0 4px 14px rgba(46,160,67,0.35) !important;
}

/* ── CARD (mobile-friendly) ── */
.stock-card {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:12px 14px; margin:6px 0; cursor:pointer;
}
.stock-card:hover { border-color:#30363d; }
.signal-badge {
    display:inline-block; background:#1c2128; border:1px solid #30363d;
    border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:700;
}

/* ── EXPANDER ── */
details { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; }
details summary { color:#c9d1d9 !important; font-weight:600 !important; padding:10px 14px !important; }

/* ── PROGRESS / SPINNER ── */
.stProgress > div > div { background:#238636 !important; }
.stSpinner > div { border-top-color:#2ea043 !important; }

/* ── REGIME BANNER ── */
.regime-bull {
    background:#0d2818; border:1px solid #238636; border-radius:8px;
    padding:8px 14px; margin-bottom:10px;
}
.regime-bear {
    background:#2d0f0f; border:1px solid #f85149; border-radius:8px;
    padding:8px 14px; margin-bottom:10px;
}

/* ── MOBILE responsive ── */
@media (max-width: 768px) {
    .main .block-container { padding: 0.5rem 0.6rem 1rem 0.6rem !important; }
    .stTabs [data-baseweb="tab"] { font-size:0.72rem !important; padding:5px 8px !important; }
    [data-testid="stMetricValue"] span { font-size:1.1rem !important; }
}

#MainMenu, footer, .stDeployButton { display:none !important; }
</style>
"""


def inject_css():
    st.markdown(CSS_BLOCK, unsafe_allow_html=True)


def _sty_signal(v):
    v = str(v)
    if "Institutional" in v: return "color:#ffd700;font-weight:800;"
    if "Smart Money" in v: return "color:#ab7df8;font-weight:800;"
    if "Strong Buy" in v: return "color:#3fb950;font-weight:800;"
    if "Breakout" in v or "เบรคเอาท์" in v: return "color:#f7b731;font-weight:700;"
    if "Uptrend" in v or "ขาขึ้น" in v: return "color:#3fb950;font-weight:600;"
    if "ขาลง" in v or "Bear" in v: return "color:#f85149;font-weight:700;"
    if "เฝ้าระวัง" in v or "Watch" in v: return "color:#d29922;font-weight:600;"
    if "Squeeze" in v or "Accum" in v: return "color:#26c6da;font-weight:700;"
    return "color:#e6edf3;"


def _sty_rsi(v):
    try:
        f = float(v)
        if f < 35: return "color:#3fb950;font-weight:700;"
        if f > 70: return "color:#f85149;font-weight:700;"
        if f < 45: return "color:#79c0ff;"
    except Exception:
        pass
    return "color:#e6edf3;"


def _sty_pct(v):
    try:
        f = float(str(v).replace("%", "").replace("+", ""))
        if f > 2: return "color:#3fb950;font-weight:600;"
        if f < -2: return "color:#f85149;font-weight:600;"
    except Exception:
        pass
    return "color:#8b949e;"


def _sty_gem(v):
    v = str(v)
    if "Hidden Gem" in v: return "color:#ffd700;font-weight:800;"
    if "Emerging" in v: return "color:#3fb950;font-weight:700;"
    if "Watch" in v: return "color:#d29922;font-weight:600;"
    return "color:#8b949e;"


def _sty_squeeze(v):
    v = str(v)
    if "Squeezing" in v: return "color:#ab7df8;font-weight:800;"
    if "Tightening" in v: return "color:#79c0ff;font-weight:700;"
    if "Just Broke" in v: return "color:#3fb950;font-weight:700;"
    if "Expanding" in v: return "color:#f7b731;font-weight:600;"
    return "color:#8b949e;"


def _sty_rs(v):
    try:
        f = float(v)
        if f > 5: return "color:#3fb950;font-weight:700;"
        if f > 0: return "color:#79c0ff;"
        if f < -5: return "color:#f85149;font-weight:700;"
        return "color:#d29922;"
    except Exception:
        return "color:#8b949e;"


def _sty_gs(v):
    try:
        n = int(v)
        if n >= 8: return "color:#ffd700;font-weight:800;"
        if n >= 6: return "color:#3fb950;font-weight:700;"
        if n >= 4: return "color:#26c6da;"
    except Exception:
        pass
    return "color:#8b949e;"


def _sty_zone(v):
    v = str(v)
    if "🟢" in v: return "color:#3fb950;font-weight:700;"
    if "🟡" in v: return "color:#f7b731;font-weight:600;"
    return "color:#8b949e;"


def _sty_bos(v):
    v = str(v)
    if "Breakout" in v: return "color:#f7b731;font-weight:800;"
    if "Near" in v: return "color:#d29922;font-weight:700;"
    return "color:#8b949e;"


def _sty_vcp(v):
    v = str(v)
    if "Clear" in v: return "color:#ab7df8;font-weight:800;"
    if "Partial" in v: return "color:#79c0ff;font-weight:700;"
    if "Watch" in v: return "color:#d29922;"
    return "color:#8b949e;"


def _sty_growth(v):
    v = str(v)
    if "Growth" in v: return "color:#ffd700;font-weight:700;"
    return "color:#8b949e;"


BASE_TBL = {
    "background-color": "#161b22", "color": "#e6edf3",
    "border": "1px solid #21262d", "font-size": "13px", "padding": "5px 10px",
}
HDR_TBL = [{"selector": "th", "props": [
    ("background-color", "#21262d"), ("color", "#ffffff"),
    ("font-weight", "700"), ("font-size", "11px"), ("padding", "8px 10px"),
    ("text-transform", "uppercase"), ("letter-spacing", "0.05em"),
]}]


def make_table(df, style_map=None):
    s = df.style.set_properties(**BASE_TBL).set_table_styles(HDR_TBL).hide(axis="index")
    if style_map:
        for col, fn in style_map.items():
            if col in df.columns:
                s = s.map(fn, subset=[col])
    return s


def info_card(label, value, color="#ffffff", sub=""):
    sub_html = f'<div style="color:#8b949e;font-size:0.75rem;margin-top:3px;">{sub}</div>' if sub else ""
    return (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
            f'padding:12px 14px;min-width:100px;">'
            f'<div style="color:#8b949e;font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:5px;">{label}</div>'
            f'<div style="color:{color};font-size:1.35rem;font-weight:800;line-height:1.2;">{value}</div>'
            f'{sub_html}</div>')


# ════════════════════════════════════════════════════════
# [ใหม่ v4.0] CARD VIEW — Mobile-Friendly (ข้อ 8)
# ════════════════════════════════════════════════════════

def _signal_color(sig: str) -> str:
    if "Institutional" in sig: return "#ffd700"
    if "Smart Money" in sig: return "#ab7df8"
    if "Strong Buy" in sig: return "#3fb950"
    if "Breakout" in sig: return "#f7b731"
    if "ขาขึ้น" in sig: return "#3fb950"
    if "ขาลง" in sig or "Bear" in sig: return "#f85149"
    if "เฝ้าระวัง" in sig: return "#d29922"
    return "#8b949e"


def render_card_view(df: pd.DataFrame, max_cards: int = 50):
    """
    Card View — หน้าแรกแสดงแค่ "ชื่อหุ้น | สัญญาณ | ราคา"
    กดขยาย st.expander เพื่อดูข้อมูลเชิงลึก
    Mobile-friendly: ไม่ต้องสไลด์ตาราง
    """
    shown = 0
    for _, row in df.iterrows():
        if shown >= max_cards:
            st.caption(f"แสดง {max_cards} หุ้นแรก — ใช้ฟิลเตอร์เพื่อแคบลง")
            break
        shown += 1

        ticker = str(row.get("Ticker", ""))
        price = row.get("Price", 0)
        prev_c = row.get("ราคาปิด", price)
        sig = str(row.get("Signal", "—"))
        sig_reason = str(row.get("Signal Reason", ""))
        trend = str(row.get("Trend", "—"))
        rsi = row.get("RSI", np.nan)
        rs20 = row.get("RS 20D", np.nan)
        stop = row.get("Stop Loss", np.nan)
        vcp = str(row.get("VCP", "—"))
        bos = str(row.get("BOS", "—"))
        zone = str(row.get("Support Zone", "—"))
        weekly = str(row.get("Weekly EMA40", "—"))
        growth = str(row.get("🌟 Growth", ""))
        gem = str(row.get("💎 Gem", "—"))
        accum = str(row.get("Accum", "—"))
        squeeze = str(row.get("Squeeze", "—"))
        vm20 = row.get("Vol×20D", np.nan)

        chg = round((price - prev_c) / prev_c * 100, 2) if prev_c and prev_c != 0 else 0
        chg_col = "#3fb950" if chg >= 0 else "#f85149"
        chg_arr = "▲" if chg >= 0 else "▼"
        sig_col = _signal_color(sig)

        # Header line (always visible)
        label_html = (
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
            f'<b style="color:#ffffff;font-size:1rem;">{ticker}</b>'
            f'<span style="color:{sig_col};font-size:0.85rem;font-weight:700;">{sig}</span>'
            f'<span style="color:#ffffff;font-size:0.9rem;">${price:,.2f}</span>'
            f'<span style="color:{chg_col};font-size:0.82rem;">{chg_arr}{chg}%</span>'
            + (f'<span style="color:#ffd700;font-size:0.78rem;">{growth}</span>' if growth else "")
            + f'</div>'
        )

        with st.expander(label_html, expanded=False):
            # เหตุผล
            if sig_reason and sig_reason != "—":
                st.markdown(f'<p style="color:#8b949e;font-size:0.82rem;margin:0 0 8px 0;">📋 {sig_reason}</p>',
                            unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("RSI", f"{rsi:.0f}" if not np.isnan(rsi) else "—")
                st.metric("Vol×20D", f"{vm20:.1f}x" if not np.isnan(vm20) else "—")
            with c2:
                st.metric("RS 20D", f"{rs20:+.1f}%" if not np.isnan(rs20) else "—")
                st.metric("Stop Loss", f"${stop:.2f}" if not np.isnan(stop) else "—")
            with c3:
                st.metric("💎 Gem", gem)
                st.metric("Accum", accum)

            # Badge row
            badges = []
            if "🟢" in zone or "🟡" in zone:
                badges.append(f'<span style="background:#1a2e1a;border:1px solid #238636;border-radius:5px;padding:2px 8px;font-size:0.75rem;color:#3fb950;">{zone}</span>')
            if "BOS" in bos:
                badges.append(f'<span style="background:#2d2a14;border:1px solid #f7b731;border-radius:5px;padding:2px 8px;font-size:0.75rem;color:#f7b731;">{bos}</span>')
            if "VCP" in vcp:
                badges.append(f'<span style="background:#1a1a2e;border:1px solid #ab7df8;border-radius:5px;padding:2px 8px;font-size:0.75rem;color:#ab7df8;">{vcp}</span>')
            if "✅" in weekly:
                badges.append(f'<span style="background:#0d2818;border:1px solid #238636;border-radius:5px;padding:2px 8px;font-size:0.75rem;color:#3fb950;">{weekly[:20]}</span>')
            if "Squeeze" in squeeze or "Tighten" in squeeze:
                badges.append(f'<span style="background:#1a0e2e;border:1px solid #ab7df8;border-radius:5px;padding:2px 8px;font-size:0.75rem;color:#ab7df8;">{squeeze}</span>')

            if badges:
                st.markdown('<div style="display:flex;flex-wrap:wrap;gap:4px;margin:6px 0;">' +
                            "".join(badges) + '</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TV CHART
# ════════════════════════════════════════════════════════

def tv_chart(ticker, height=620, interval="D"):
    import streamlit.components.v1 as components
    nyse = {"JPM","JNJ","V","PG","UNH","HD","MA","DIS","BAC","XOM","CVX","WMT","KO","T","VZ","GS","MS"}
    is_thai = ticker.endswith(".BK")
    sym = ticker.replace(".BK", "") if is_thai else ticker
    prefix = "SET" if is_thai else ("NYSE" if ticker in nyse else "NASDAQ")
    html = f"""
    <div style="border-radius:10px;overflow:hidden;border:1px solid #21262d;">
    <div class="tradingview-widget-container" style="height:{height}px;width:100%;">
    <div class="tradingview-widget-container__widget" style="height:{height}px;width:100%;"></div>
    <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
    {{
        "autosize":true,"symbol":"{prefix}:{sym}","interval":"{interval}",
        "timezone":"Asia/Bangkok","theme":"dark","style":"1","locale":"th",
        "backgroundColor":"#0d1117","gridColor":"rgba(48,54,61,0.3)",
        "hide_top_toolbar":false,"hide_legend":false,"save_image":false,
        "studies":[
            {{"id":"MAExp@tv-basicstudies","inputs":{{"length":20}},"styles":{{"plot_0":{{"color":"#f7b731","linewidth":1}}}}}},
            {{"id":"MAExp@tv-basicstudies","inputs":{{"length":50}},"styles":{{"plot_0":{{"color":"#26c6da","linewidth":1}}}}}},
            {{"id":"MAExp@tv-basicstudies","inputs":{{"length":200}},"styles":{{"plot_0":{{"color":"#ef5350","linewidth":2}}}}}},
            "RSI@tv-basicstudies","MACD@tv-basicstudies","ATR@tv-basicstudies"
        ]
    }}
    </script></div></div>"""
    components.html(html, height=height + 10, scrolling=False)


# ════════════════════════════════════════════════════════
# [ใหม่ v4.0] POSITION SIZING CALCULATOR (ข้อ 9)
# ════════════════════════════════════════════════════════

def render_position_sizer(price: float, stop_loss: float, atr: float):
    """
    คำนวณขนาด Position จาก:
    - ขนาดพอร์ต (Port Size)
    - ความเสี่ยงที่รับได้ (% ของพอร์ต)
    - จุด Stop Loss (คำนวณจาก ATR อัตโนมัติ หรือกรอกเอง)
    สูตร: จำนวนหุ้น = (Port × Risk%) / (ราคา − Stop Loss)
    """
    st.markdown("#### 📐 Position Sizing Calculator")
    st.caption("คำนวณจำนวนหุ้นที่ควรซื้อตามระดับความเสี่ยงที่ตั้งไว้")

    ps1, ps2, ps3 = st.columns(3)
    with ps1:
        port_size = st.number_input("ขนาดพอร์ต ($)", min_value=100.0, value=10000.0,
                                     step=1000.0, key="ps_port")
    with ps2:
        risk_pct = st.slider("ความเสี่ยงต่อ Trade (%)", 0.5, 5.0, 1.0, 0.5, key="ps_risk")
    with ps3:
        # ถ้ามี stop loss จาก ATR ให้ pre-fill ไว้
        default_sl = stop_loss if (not np.isnan(stop_loss) and stop_loss > 0 and stop_loss < price) else max(price * 0.95, 0.01)
        user_sl = st.number_input("Stop Loss ($)", min_value=0.01,
                                   value=round(float(default_sl), 2), step=0.5, key="ps_sl")

    risk_per_share = price - user_sl
    if risk_per_share <= 0:
        st.error("⚠️ Stop Loss ต้องต่ำกว่าราคาปัจจุบัน")
        return

    risk_amount = port_size * (risk_pct / 100)
    shares = int(risk_amount / risk_per_share)
    position_value = shares * price
    position_pct = position_value / port_size * 100

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("จำนวนหุ้น", f"{shares:,} หุ้น")
    r2.metric("มูลค่า Position", f"${position_value:,.0f}")
    r3.metric("% ของพอร์ต", f"{position_pct:.1f}%")
    r4.metric("ความเสี่ยง (Max Loss)", f"${risk_amount:,.0f}")

    atr_note = f" · ATR={atr:.2f} (ใช้ในการคำนวณ SL อัตโนมัติ)" if not np.isnan(atr) else ""
    st.caption(f"⚠️ ตัวเลขนี้เป็นแค่จุดอ้างอิง ไม่ใช่คำแนะนำการลงทุน{atr_note}")


# ════════════════════════════════════════════════════════
# [ใหม่ v4.0] MINI USER GUIDE (ข้อ 12)
# ════════════════════════════════════════════════════════

def render_user_guide():
    with st.sidebar.expander("📖 คู่มือการใช้งาน", expanded=False):
        st.markdown("""
**🎯 สัญญาณระดับท็อป (God-Tier)**

| สัญญาณ | ความหมาย | เงื่อนไขหลัก |
|---|---|---|
| 🎯 Institutional Breakout | เบรคแนวต้าน+สถาบันเข้า | BOS+Volume+RS+Weekly |
| 🔥 Smart Money Accum | สถาบันสะสมในแนวรับ | VCP+แนวรับ+RS บวก |
| 🚀 Breakout | วิ่งแรงเหนือ EMA | Volume×2+EMA50>200 |
| 🔥 Strong Buy | โอกาสดีใกล้ฐาน | RSI<40+Volume สูง |

---

**📐 วิธีใช้ Position Sizing**
1. กรอก **ขนาดพอร์ต** (เงินทั้งหมดที่มี)
2. ตั้ง **ความเสี่ยง** ต่อ trade (แนะนำ 1-2%)
3. ระบบคำนวณ **Stop Loss** อัตโนมัติจาก ATR
4. ดูผลลัพธ์: "ควรซื้อกี่หุ้น" และ "มูลค่า position"

**ตัวอย่าง:** พอร์ต $10,000 · ความเสี่ยง 1% · SL ห่าง $2
→ ซื้อได้ 50 หุ้น (ขาดทุนสูงสุด $100 = 1% ของพอร์ต)

---

**🔴 กฎเหล็ก Stop Loss**
- ตัดขาดทุน **ทันที** เมื่อราคาหลุด Stop Loss
- อย่าเลื่อน SL ลงต่ำกว่าเดิม (averaging down)
- SL ที่ดี = ต่ำกว่าแนวรับ 1 ATR เสมอ

---

**📊 อ่านค่า Support Zone**
- 🟢 ถึงแนวรับ — ราคาแตะ EMA50/200 (ห่าง ≤0.5 ATR) โอกาสดีที่สุด
- 🟡 ใกล้แนวรับ — ห่าง 0.5-1.5 ATR รอยืนยันก่อน
- 🚨 BOS Breakout — เบรค Swing High ล่าสุด
- 🎯 VCP Clear — ราคาแกว่งแคบลง+Volume แห้ง (สัญญาณ Minervini)

---

**⚠️ ข้อควรระวัง**
- สัญญาณทุกอย่างเป็น heuristic ไม่ใช่การันตี
- Weekly EMA40 เปิดปิดได้ใน sidebar (ช้าลงแต่แม่นขึ้น)
- ตลาดขาลง (Regime ❌) ควรลดขนาด position
""", unsafe_allow_html=False)


# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Screener Pro v4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


def main():
    st.markdown("""
    <div style="text-align:center;padding:6px 0 12px 0;">
        <h1 style="font-size:1.6rem;margin:0;">
            📊 Institutional Stock Screener
            <span style="font-size:0.85rem;color:#3fb950;"> v4.0</span>
        </h1>
        <p style="color:#8b949e;font-size:0.82rem;margin:3px 0 0 0;">
            ATR · BOS · VCP · Multi-Timeframe · God-Tier Signals · Position Sizing
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ────────────────────────────────────
    for k, v in [("df", pd.DataFrame()), ("watchlist", None), ("ran", False)]:
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.watchlist is None:
        st.session_state.watchlist = load_watchlist()

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ ตั้งค่า")
        universe = st.selectbox("🌍 Universe | กลุ่มหุ้น", list(UNIVERSE_OPTIONS.keys()))

        sector_choice = []
        if universe == "Sector Focus | เลือกตามหมวด":
            sector_choice = st.multiselect("เลือก Sector", list(SECTOR_MAP.keys()),
                                           default=["Technology | เทคโนโลยี"])

        custom_input = ""
        if universe == "Custom Tickers":
            custom_input = st.text_area("Tickers (คั่นด้วย ,)", "AAPL,MSFT,NVDA", height=70)

        st.markdown("---")
        st.markdown("**🔬 Filters**")
        min_gem = st.slider("💎 Min Gem Score", 0, 10, 0)
        min_accum = st.slider("📦 Min Accum Score", 0, 5, 0)
        sig_type_filter = st.multiselect("Signal Type",
            ["🎯 Institutional Breakout", "🔥 Smart Money Accum",
             "🚀 Breakout", "🔥 Strong Buy", "📈 ขาขึ้น"],
            default=[], placeholder="ทั้งหมด")

        st.markdown("---")
        with st.expander("⚙️ Advanced Settings"):
            period = st.selectbox("Period", ["1y", "2y", "6mo", "3mo"], index=0)
            interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
            use_rs = st.checkbox("คำนวณ RS vs SPY", value=True)
            # ข้อ 5: Weekly MTF toggle
            use_weekly = st.checkbox("✅ Weekly EMA40 MTF", value=True,
                                     help="ดึงข้อมูลรายสัปดาห์เพิ่ม (+~0.5s/หุ้น) เพื่อยืนยันเทรนด์ใหญ่")
            # ข้อ 8: View mode toggle
            view_mode = st.radio("View Mode", ["📱 Card View (มือถือ)", "📊 Table View (ตาราง)"],
                                 index=0)
            max_tk = st.slider("Max Tickers", 10, 300, 50, step=10)
            # ข้อ 10: Market regime benchmark
            regime_bench = st.selectbox("Regime Benchmark",
                ["SPY", "QQQ", "SET50.BK"], index=0)

        st.markdown("---")
        run_btn = st.button("🚀 Run Screener (สแกนสด)", use_container_width=True)

        with st.expander("💾 Export"):
            if not st.session_state.df.empty:
                csv = st.session_state.df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv,
                    f"screener_v4_{datetime.date.today()}.csv", "text/csv",
                    use_container_width=True)
            else:
                st.caption("รัน Screener ก่อน")

        with st.expander("🗑️ ล้าง Cache"):
            if st.button("ล้าง Cache", use_container_width=True):
                tickers_for_clear = resolve_tickers(universe, sector_choice, custom_input)[:max_tk]
                if clear_cache_for(universe, tuple(tickers_for_clear), period, interval):
                    st.success("ล้างแล้ว")
                else:
                    st.info("ไม่มี cache")

        # ข้อ 12: Mini User Guide
        render_user_guide()

        st.markdown(
            f"<p style='color:#7d8590;font-size:0.7rem;margin-top:8px;'>"
            f"Watchlist: {len(st.session_state.watchlist)} หุ้น · v4.0</p>",
            unsafe_allow_html=True)

    # ── Resolve tickers ──────────────────────────────────
    tickers_all = resolve_tickers(universe, sector_choice, custom_input)
    tickers_use = tickers_all[:max_tk]

    auto_loaded = False
    bundle_gen_at = None
    new_signal_hits = []

    # ── [ข้อ 10] Market Regime Banner (ดึงก่อนแสดงผลเสมอ) ────
    regime = get_market_regime(regime_bench)
    regime_class = "regime-bear" if regime["warning"] else "regime-bull"
    st.markdown(
        f'<div class="{regime_class}">'
        f'<span style="color:{regime["color"]};font-weight:700;font-size:0.88rem;">'
        f'🌐 Market Regime: {regime["label"]}</span>'
        + (' <span style="color:#f85149;font-size:0.82rem;">— ⚠️ ตลาดรวมเป็นขาลง การเทรดมีความเสี่ยงสูง ลดขนาด position</span>'
           if regime["warning"] else "")
        + '</div>', unsafe_allow_html=True)

    # ── Run screener ─────────────────────────────────────
    if run_btn:
        bench_tuple = None
        if use_rs:
            with st.spinner("ดึง SPY benchmark…"):
                try:
                    spy_df = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
                    bench_tuple = make_bench_tuple(spy_df)
                except Exception as e:
                    log_err("fetch SPY benchmark", e)
                    st.warning("ดึง SPY ไม่สำเร็จ — สแกนต่อโดยไม่มี RS")

        prog = st.progress(0.0, text=f"⚡ กำลังสแกน 0/{len(tickers_use)} หุ้น…")

        def _on_progress(done, total):
            prog.progress(done / total if total else 1.0,
                          text=f"⚡ สแกน {done}/{total} หุ้น{'  (Weekly MTF เปิด — ช้าขึ้นปกติ)' if use_weekly else ''}…")

        df = batch_scan(tuple(tickers_use), period, interval, bench_tuple,
                        include_weekly=use_weekly, progress_cb=_on_progress)
        prog.empty()
        st.session_state.df = df
        st.session_state.ran = True
        save_disk_cache(universe, tuple(tickers_use), period, interval, df)

        last_sig = load_last_signals(universe)
        new_signal_hits = detect_new_signals(df, last_sig)
        save_last_signals(universe, signals_snapshot(df))
        if new_signal_hits:
            msg = "🔔 สัญญาณใหม่: " + ", ".join(
                f"{h['ticker']} {h['signal']}" for h in new_signal_hits[:20])
            maybe_notify_telegram(msg)
    else:
        bundle_gen_at, bundle_df = load_prefetched_bundle()
        if bundle_gen_at:
            have = get_with_bundle_fallback(tickers_use, bundle_df)
            st.session_state.df = have
            st.session_state.ran = True
            auto_loaded = True
            new_signal_hits = load_prefetch_alerts()
        elif not st.session_state.ran:
            st.session_state.df = pd.DataFrame()

    df = st.session_state.df

    # ── Status bar ────────────────────────────────────────
    if st.session_state.ran and not df.empty:
        if auto_loaded:
            try:
                gen_dt = datetime.datetime.fromisoformat(str(bundle_gen_at).replace("Z", "+00:00"))
                gen_lbl = gen_dt.astimezone(ZoneInfo("Asia/Bangkok")).strftime("%d/%m %H:%M น.")
            except Exception:
                gen_lbl = str(bundle_gen_at) or "—"
            st.markdown(
                f'<div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;'
                f'padding:7px 12px;margin-bottom:8px;font-size:0.82rem;">'
                f'<span style="color:#3fb950;">⚡ ข้อมูลล่วงหน้า</span>'
                f'<span style="color:#8b949e;"> · {gen_lbl} · {len(df)} หุ้น</span>'
                f'</div>', unsafe_allow_html=True)
        else:
            age_lbl = cache_age_label(universe, tuple(tickers_use), period, interval)
            st.markdown(
                f'<div style="background:#1c2128;border:1px solid #238636;border-radius:8px;'
                f'padding:7px 12px;margin-bottom:8px;font-size:0.82rem;">'
                f'<span style="color:#3fb950;">✅ สแกนสดเสร็จ</span>'
                f'<span style="color:#8b949e;"> · {age_lbl} · {len(df)} หุ้น</span>'
                f'</div>', unsafe_allow_html=True)

        # New signals alert
        if new_signal_hits:
            chips = " ".join(
                f'<span style="background:#132a1a;border:1px solid #3fb950;border-radius:5px;'
                f'padding:2px 8px;font-size:0.76rem;margin-right:3px;">'
                f'<b style="color:#3fb950;">{h["ticker"]}</b> {h["signal"]}</span>'
                for h in new_signal_hits[:20])
            st.markdown(
                f'<div style="background:#132a1a;border:1px solid #3fb950;border-radius:8px;'
                f'padding:8px 12px;margin-bottom:8px;">'
                f'<span style="color:#3fb950;font-weight:700;font-size:0.83rem;">🔔 สัญญาณใหม่ {len(new_signal_hits)} หุ้น</span>'
                f' {chips}</div>', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "💎 Hidden Gems",
        "🔍 Deep Dive",
        "📈 Backtester",
        "⭐ Watchlist",
    ])

    # ════════════════════════════════════════════════════
    # TAB 1: DASHBOARD
    # ════════════════════════════════════════════════════
    with tab1:
        if not st.session_state.ran:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#8b949e;">
                <div style="font-size:2.5rem;">📊</div>
                <h3 style="color:#c9d1d9;">ยังไม่มีข้อมูล</h3>
                <p>กด 🚀 Run Screener หรือรอ prefetch อัตโนมัติ</p>
            </div>""", unsafe_allow_html=True)
        elif df.empty:
            st.error("⚠️ ไม่พบข้อมูล — ลองกด Run Screener หรือตรวจ Ticker/อินเทอร์เน็ต")
        else:
            # Summary cards
            total = len(df)
            bulls = len(df[df["Trend"].str.contains("Bull", na=False)])
            institutional = len(df[df["Signal"].str.contains("Institutional", na=False)]) if "Signal" in df else 0
            smart_money = len(df[df["Signal"].str.contains("Smart Money", na=False)]) if "Signal" in df else 0
            breaks = len(df[df["Signal"].str.contains("Breakout", na=False)]) if "Signal" in df else 0
            gems = len(df[df["💎 Gem"].str.contains("Gem", na=False)]) if "💎 Gem" in df else 0
            growth_cnt = len(df[df["🌟 Growth"].str.contains("Growth", na=False)]) if "🌟 Growth" in df else 0
            avg_rsi = df["RSI"].mean() if "RSI" in df else 0

            cards_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">'
            cards_html += info_card("สแกน", str(total))
            cards_html += info_card("Bull", str(bulls), "#3fb950")
            cards_html += info_card("🎯 Institutional", str(institutional), "#ffd700")
            cards_html += info_card("🔥 Smart Money", str(smart_money), "#ab7df8")
            cards_html += info_card("🚀 Breakout", str(breaks), "#f7b731")
            cards_html += info_card("💎 Gem", str(gems), "#ffd700")
            cards_html += info_card("🌟 Growth", str(growth_cnt), "#3fb950")
            cards_html += info_card("Avg RSI", f"{avg_rsi:.0f}", "#79c0ff")
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            st.caption("⚠️ Signal เป็น heuristic จาก ATR/RSI/Volume/BOS/VCP — ไม่ใช่คำแนะนำการลงทุน "
                       "ดูคอลัมน์ 'เหตุผล' และ Stop Loss ประกอบการตัดสินใจเสมอ")

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                sig_filter = st.multiselect("Signal", df["Signal"].unique().tolist() if "Signal" in df else [],
                                            default=sig_type_filter or [], key="d_sig", placeholder="ทั้งหมด")
            with fc2:
                trend_filter = st.multiselect("Trend", ["🟢 Bull", "🔴 Bear"],
                                              default=[], key="d_tr", placeholder="ทั้งหมด")
            with fc3:
                zone_filter = st.multiselect("Support Zone",
                    [x for x in df["Support Zone"].unique().tolist() if "🟢" in str(x) or "🟡" in str(x)] if "Support Zone" in df else [],
                    default=[], key="d_zone", placeholder="ทั้งหมด")

            # Apply filters
            mask = pd.Series(True, index=df.index)
            if sig_filter: mask &= df["Signal"].isin(sig_filter)
            if trend_filter: mask &= df["Trend"].apply(lambda x: any(t in str(x) for t in trend_filter))
            if zone_filter: mask &= df["Support Zone"].isin(zone_filter)
            if min_gem > 0 and "Gem Score" in df.columns: mask &= df["Gem Score"] >= min_gem
            if min_accum > 0 and "Accum Score" in df.columns: mask &= df["Accum Score"] >= min_accum
            dfv = df[mask].copy()

            # Sort by signal priority
            prio = {"🎯 Institutional Breakout": 0, "🔥 Smart Money Accum": 1,
                    "🚀 Breakout": 2, "🔥 Strong Buy": 3, "🔥 Strong Buy ⚠️RS-": 4,
                    "📈 ขาขึ้น": 5, "⚠️ เฝ้าระวัง": 6, "🔄 Neutral": 7,
                    "⏳ รอ Pullback": 8, "❌ ขาลง": 9}
            if "Signal" in dfv.columns:
                dfv["_p"] = dfv["Signal"].map(prio).fillna(10)
                dfv = dfv.sort_values("_p").drop(columns=["_p"])

            st.markdown(f"**{len(dfv)} หุ้นที่ตรงเงื่อนไข**")

            # ── View Mode Switch (ข้อ 8) ──────────────────
            if "Card View" in view_mode:
                render_card_view(dfv, max_cards=60)
            else:
                show_cols = [c for c in ["Ticker", "Price", "Trend", "Signal", "Signal Reason",
                                         "Stop Loss", "Support Zone", "BOS", "VCP",
                                         "Weekly EMA40", "RSI", "Vol×20D", "RS 20D",
                                         "💎 Gem", "Accum", "🌟 Growth", "EMA Pattern"]
                             if c in dfv.columns]
                smap = {"Signal": _sty_signal, "💎 Gem": _sty_gem, "RSI": _sty_rsi,
                        "Support Zone": _sty_zone, "BOS": _sty_bos, "VCP": _sty_vcp,
                        "RS 20D": _sty_rs, "🌟 Growth": _sty_growth}
                st.dataframe(make_table(dfv[show_cols], smap), use_container_width=True, height=520)

            st.markdown("---")
            wl_col1, wl_col2 = st.columns([3, 1])
            with wl_col1:
                add_tk = st.text_input("➕ เพิ่ม Watchlist", placeholder="AAPL", key="wl_add")
            with wl_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("เพิ่ม", key="wl_add_btn") and add_tk.strip():
                    tk = add_tk.strip().upper()
                    if tk not in st.session_state.watchlist:
                        st.session_state.watchlist.append(tk)
                        save_watchlist(st.session_state.watchlist)
                        st.success(f"เพิ่ม {tk} แล้ว")

    # ════════════════════════════════════════════════════
    # TAB 2: HIDDEN GEMS
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 💎 Hidden Gem Finder")
        st.caption("EMA สวย + Volume สะสม + VCP + ตลาดยังไม่สนใจ")

        if df.empty:
            st.info("รัน Screener ก่อน")
        else:
            g_cols = st.columns(4)
            keywords = [("💎 Hidden Gem", "Hidden", "#ffd700"),
                        ("🔭 Emerging Gem", "Emerging", "#3fb950"),
                        ("🎯 VCP Clear", "VCP Clear", "#ab7df8"),
                        ("🟢 ถึงแนวรับ", "ถึง", "#26c6da")]
            for i, (lbl, kw, clr) in enumerate(keywords):
                cnt = df.apply(lambda r, kw=kw: kw in str(r.get("💎 Gem", "")) or
                               kw in str(r.get("VCP", "")) or kw in str(r.get("Support Zone", "")), axis=1).sum()
                g_cols[i].metric(lbl, int(cnt))

            st.markdown("---")

            gf1, gf2, gf3 = st.columns(3)
            with gf1:
                gem_f = st.multiselect("💎 Gem Level", ["💎 Hidden Gem", "🔭 Emerging Gem"],
                                       default=[], key="gf1", placeholder="ทั้งหมด")
            with gf2:
                vcp_f = st.multiselect("VCP Pattern", ["🎯 VCP Clear", "📐 VCP Partial"],
                                       default=[], key="gf2", placeholder="ทั้งหมด")
            with gf3:
                zone_f2 = st.multiselect("Support Zone", ["🟢", "🟡"],
                                         default=[], key="gf3", placeholder="ทั้งหมด")

            gem_show = [c for c in ["Ticker", "Price", "💎 Gem", "Gem Score",
                                    "VCP", "Support Zone", "BOS", "EMA Pattern",
                                    "Accum", "RSI", "Vol×20D", "RS 20D", "Signal",
                                    "Stop Loss", "🌟 Growth", "MktCap$B"] if c in df.columns]
            dfg = df[gem_show].copy()
            gm = pd.Series(True, index=dfg.index)
            if gem_f: gm &= df["💎 Gem"].isin(gem_f)
            if vcp_f: gm &= df["VCP"].isin(vcp_f)
            if zone_f2: gm &= df["Support Zone"].apply(lambda x: any(z in str(x) for z in zone_f2))
            if min_gem > 0: gm &= df["Gem Score"] >= min_gem
            dfg = dfg[gm]
            if "Gem Score" in dfg.columns:
                dfg = dfg.sort_values("Gem Score", ascending=False)

            gsmap = {"💎 Gem": _sty_gem, "VCP": _sty_vcp, "Support Zone": _sty_zone,
                     "BOS": _sty_bos, "Gem Score": _sty_gs, "Signal": _sty_signal,
                     "RSI": _sty_rsi, "RS 20D": _sty_rs, "🌟 Growth": _sty_growth}
            st.dataframe(make_table(dfg, gsmap), use_container_width=True, height=500)

    # ════════════════════════════════════════════════════
    # TAB 3: DEEP DIVE
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔍 วิเคราะห์รายตัว")
        pick_list = df["Ticker"].tolist() if not df.empty else tickers_use[:50]
        d1, d2, d3 = st.columns([3, 1, 1])
        with d1:
            sel = st.selectbox("เลือกหุ้น", pick_list, key="dd_sel")
        with d2:
            ch_h = st.selectbox("ความสูงกราฟ", [620, 700, 800, 500], index=0, key="dd_h")
        with d3:
            ch_iv = st.selectbox("Timeframe", ["D", "W", "60", "15"], index=0, key="dd_iv",
                                 format_func=lambda x: {"D": "รายวัน", "W": "สัปดาห์", "60": "1H", "15": "15M"}[x])

        if sel:
            row = None
            if not df.empty and sel in df["Ticker"].values:
                row = df[df["Ticker"] == sel].iloc[0].to_dict()

            if row:
                px_now = row.get("Price", 0)
                pc_now = row.get("ราคาปิด", px_now)
                chg_pct = round((px_now - pc_now) / pc_now * 100, 2) if pc_now else 0
                chg_col = "#3fb950" if chg_pct >= 0 else "#f85149"
                chg_arr = "▲" if chg_pct >= 0 else "▼"
                sig_now = row.get("Signal", "—")
                sig_reason_now = row.get("Signal Reason", "")
                stop_now = row.get("Stop Loss", np.nan)
                atr_now = row.get("ATR", np.nan)
                vwap_now = row.get("VWAP", np.nan)
                vwap_diff_now = row.get("vs VWAP%", np.nan)
                zone_now = row.get("Support Zone", "—")
                bos_now = row.get("BOS", "—")
                vcp_now = row.get("VCP", "—")
                weekly_now = row.get("Weekly EMA40", "—")
                rs20_now = row.get("RS 20D", np.nan)
                growth_now = row.get("🌟 Growth", "")

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;flex-wrap:wrap;">'
                    f'<span style="font-size:1.9rem;font-weight:800;color:#ffffff;">${px_now:,.2f}</span>'
                    f'<span style="color:{chg_col};font-size:1rem;font-weight:700;">{chg_arr} {chg_pct}%</span>'
                    f'<span style="background:#21262d;border:1px solid #30363d;border-radius:6px;'
                    f'padding:4px 10px;font-size:0.85rem;font-weight:700;">{sig_now}</span>'
                    + (f'<span style="color:#ffd700;font-size:0.85rem;">{growth_now}</span>' if growth_now else "")
                    + f'</div>', unsafe_allow_html=True)

                if sig_reason_now:
                    st.caption(f"📋 {sig_reason_now}")

                # v4.0 indicator badges
                new_badge_html = '<div style="display:flex;flex-wrap:wrap;gap:5px;margin:8px 0;">'
                badge_items = [
                    (zone_now, "#238636" if "🟢" in zone_now else "#d29922" if "🟡" in zone_now else None),
                    (bos_now, "#f7b731" if "BOS" in bos_now else None),
                    (vcp_now, "#ab7df8" if "VCP" in vcp_now else None),
                    (weekly_now[:25] if weekly_now != "—" else "—",
                     "#3fb950" if "✅" in weekly_now else "#f85149" if "❌" in weekly_now else None),
                ]
                for lbl, col in badge_items:
                    if col and lbl != "—":
                        new_badge_html += (f'<span style="background:#1c2128;border:1px solid {col};'
                                           f'border-radius:6px;padding:3px 10px;font-size:0.78rem;'
                                           f'font-weight:600;color:{col};">{lbl}</span>')
                new_badge_html += "</div>"
                st.markdown(new_badge_html, unsafe_allow_html=True)

                # EMA badges
                ema_info = [(5, "#a8b3c5"), (20, "#f7b731"), (50, "#26c6da"), (200, "#ef5350")]
                bdg = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px 0;">'
                for n, col in ema_info:
                    ev = row.get(f"EMA{n}", None)
                    dev = row.get(f"vs EMA{n}%", None)
                    if ev and dev is not None:
                        dc = "#3fb950" if dev > 0 else "#f85149"
                        sgn = "+" if dev > 0 else ""
                        bdg += (f'<div style="background:#1c2128;border:1px solid {col}40;'
                                f'border-radius:8px;padding:8px 10px;min-width:80px;">'
                                f'<div style="color:{col};font-size:0.65rem;font-weight:700;">EMA {n}</div>'
                                f'<div style="color:#ffffff;font-size:0.88rem;font-weight:700;">${ev:,.2f}</div>'
                                f'<div style="color:{dc};font-size:0.72rem;">{sgn}{dev:.2f}%</div></div>')
                bdg += '</div>'
                st.markdown(bdg, unsafe_allow_html=True)

            st.caption("📈 กราฟจาก TradingView · EMA20/50/200 · RSI · MACD · ATR")
            tv_chart(sel, height=ch_h, interval=ch_iv)

            st.markdown("---")
            fetch_live_btn = st.button("⚡ ดึงข้อมูลสด", key="dd_live")
            if fetch_live_btn:
                with st.spinner("กำลังดึงข้อมูลสด…"):
                    rt = fetch_live(sel)
                if rt:
                    chg = rt.get("change") or 0
                    arr = "▲" if chg >= 0 else "▼"
                    cols_rt = st.columns(6)
                    cols_rt[0].metric("💰 ราคาสด", str(rt["price"]))
                    cols_rt[1].metric("📈 เปลี่ยน", f"{arr} {chg}%")
                    cols_rt[2].metric("🔼 High", str(rt["high"]))
                    cols_rt[3].metric("🔽 Low", str(rt["low"]))
                    cols_rt[4].metric("📊 Volume", rt["vol"])
                    cols_rt[5].metric("🏢 Mkt Cap", rt["cap"])

            if row:
                st.markdown("---")
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.markdown("**MOMENTUM**")
                    st.metric("RSI (14)", row.get("RSI", "—"))
                    st.metric("MACD_H", row.get("MACD_H", "—"))
                    st.metric("ATR", row.get("ATR", "—"))
                    st.metric("VWAP", f'${row.get("VWAP", "—")}')
                    st.metric("vs VWAP%", f'{row.get("vs VWAP%", "—")}%')
                with tc2:
                    st.markdown("**VOLUME & STRENGTH**")
                    st.metric("Vol ×20D", f'{row.get("Vol×20D", "—")}×')
                    st.metric("Accum", row.get("Accum", "—"))
                    st.metric("RS 20D", f'{row.get("RS 20D", "—")}%')
                    st.metric("RS 50D", f'{row.get("RS 50D", "—")}%')
                    st.metric("Gem Score", row.get("Gem Score", "—"))
                with tc3:
                    st.markdown("**PERFORMANCE & FUND**")
                    st.metric("YTD%", f'{row.get("YTD%", "—")}%')
                    st.metric("Drawdown%", f'{row.get("Drawdown%", "—")}%')
                    st.metric("P/E", row.get("P/E", "—"))
                    st.metric("EPS Growth%", f'{row.get("EPS Growth%", "—")}%')
                    st.metric("Rev Growth%", f'{row.get("Rev Growth%", "—")}%')

                st.markdown("---")
                # ── [ข้อ 9] Position Sizing Calculator ──────────
                render_position_sizer(
                    price=float(row.get("Price", 100)),
                    stop_loss=float(row.get("Stop Loss", np.nan) or np.nan),
                    atr=float(row.get("ATR", np.nan) or np.nan),
                )

    # ════════════════════════════════════════════════════
    # TAB 4: BACKTESTER
    # ════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📈 Backtester — EMA Squeeze Strategy")
        st.caption("เข้าซื้อที่ open แท่งถัดไป · เปรียบ Buy&Hold · Max Drawdown · Sharpe")

        b1, b2, b3 = st.columns([3, 1, 1])
        with b1:
            bt_ticker = st.text_input("Ticker", value="AAPL", key="bt_tk").upper()
        with b2:
            hold_d = st.selectbox("ถือกี่วัน", [10, 15, 20, 30], index=2, key="bt_hold")
        with b3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button("▶️ Run Backtest", key="bt_run")

        if run_bt and bt_ticker:
            with st.spinner(f"กำลัง Backtest {bt_ticker}…"):
                res = backtest(bt_ticker, hold_d)
            if "error" in res:
                st.error(f"❌ {res['error']}")
            elif res.get("n", 0) == 0:
                st.warning("ไม่พบ signal ใน 2 ปี")
            else:
                wc = "#3fb950" if res["win_rate"] >= 55 else "#d29922" if res["win_rate"] >= 45 else "#f85149"
                ac = "#3fb950" if res["avg"] > 0 else "#f85149"
                cards = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0;">'
                cards += info_card("Trades", str(res["n"]))
                cards += info_card("Win Rate", f'{res["win_rate"]}%', wc)
                cards += info_card("Avg Return", f'{res["avg"]}%', ac)
                cards += info_card("Best", f'+{res["best"]}%', "#3fb950")
                cards += info_card("Worst", f'{res["worst"]}%', "#f85149")
                cards += '</div>'
                st.markdown(cards, unsafe_allow_html=True)

                strat_ret = res.get("strategy_compound_ret", 0)
                bh_ret = res.get("buy_hold_ret", 0)
                beat = strat_ret > bh_ret
                cards2 = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 14px 0;">'
                cards2 += info_card("กลยุทธ์ Compound", f'{strat_ret:+.1f}%',
                                    "#3fb950" if beat else "#f85149", "ทบต้นทุก trade")
                cards2 += info_card("Buy & Hold", f'{bh_ret:+.1f}%', "#79c0ff")
                cards2 += info_card("Max Drawdown", f'{res.get("max_drawdown", 0)}%', "#f85149")
                sharpe_v = res.get("sharpe")
                cards2 += info_card("Sharpe (ประมาณ)", f'{sharpe_v}' if sharpe_v is not None else "—", "#ab7df8")
                cards2 += '</div>'
                st.markdown(cards2, unsafe_allow_html=True)

                st.info("✅ กลยุทธ์ดีกว่า Buy&Hold" if beat else "⚠️ Buy&Hold ดีกว่ากลยุทธ์นี้ในช่วงทดสอบ")
                with st.expander("⚠️ ข้อจำกัดของ Backtest"):
                    st.caption(res.get("notes", ""))

                with st.expander("ดู trades ทั้งหมด"):
                    details = res.get("trade_details", [])
                    if details:
                        tdf = pd.DataFrame(details)
                        tdf.insert(0, "Trade #", range(1, len(tdf) + 1))
                        tdf["Result"] = tdf["ret"].apply(lambda x: "✅ Win" if x > 0 else "❌ Loss")
                        tdf = tdf.rename(columns={"ret": "Return %", "entry_date": "Entry", "exit_date": "Exit"})
                        st.dataframe(make_table(tdf), use_container_width=True)

    # ════════════════════════════════════════════════════
    # TAB 5: WATCHLIST
    # ════════════════════════════════════════════════════
    with tab5:
        st.markdown("### ⭐ Watchlist")
        st.caption("persist บน disk · ล้างถ้า redeploy")

        wc1, wc2, wc3 = st.columns([3, 1, 1])
        with wc1:
            new_tk = st.text_input("ชื่อหุ้น", placeholder="AAPL หรือ PTT.BK", key="wl_new")
        with wc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ เพิ่ม", key="wl_add2") and new_tk.strip():
                tk = new_tk.strip().upper()
                if tk not in st.session_state.watchlist:
                    st.session_state.watchlist.append(tk)
                    save_watchlist(st.session_state.watchlist)
        with wc3:
            st.markdown("<br>", unsafe_allow_html=True)
            rem_tk = st.selectbox("ลบออก", ["—"] + st.session_state.watchlist, key="wl_rem")
            if rem_tk != "—":
                if st.button("🗑️ ลบ", key="wl_del"):
                    st.session_state.watchlist.remove(rem_tk)
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()

        if not st.session_state.watchlist:
            st.info("ยังไม่มีหุ้น — เพิ่มจาก Dashboard หรือพิมพ์ข้างบน")
        else:
            st.markdown(f"**{len(st.session_state.watchlist)} หุ้น:** {', '.join(st.session_state.watchlist)}")
            st.markdown("---")
            scan_wl = st.button("🔄 Scan Watchlist ทั้งหมด", key="wl_scan")
            if scan_wl:
                with st.spinner("กำลังวิเคราะห์ Watchlist…"):
                    _, bundle_df_wl = load_prefetched_bundle()
                    wl_df_result = get_with_bundle_fallback(
                        st.session_state.watchlist, bundle_df_wl, max_live_fallback=50)
                    st.session_state["wl_df"] = wl_df_result

            if "wl_df" in st.session_state and not st.session_state["wl_df"].empty:
                wdf = st.session_state["wl_df"]
                wl_show = [c for c in ["Ticker", "Price", "Trend", "Signal", "Signal Reason",
                                       "Stop Loss", "Support Zone", "BOS", "VCP",
                                       "RSI", "Vol×20D", "RS 20D", "💎 Gem",
                                       "🌟 Growth", "YTD%", "Drawdown%"]
                           if c in wdf.columns]
                wsmap = {"Signal": _sty_signal, "💎 Gem": _sty_gem, "RSI": _sty_rsi,
                         "Support Zone": _sty_zone, "BOS": _sty_bos, "VCP": _sty_vcp,
                         "RS 20D": _sty_rs, "🌟 Growth": _sty_growth}
                st.dataframe(make_table(wdf[wl_show], wsmap), use_container_width=True, height=400)

                with st.expander("📈 Backtest ทุกตัวใน Watchlist"):
                    bt_rows = []
                    for tk in st.session_state.watchlist:
                        with st.spinner(f"Backtest {tk}…"):
                            r = backtest(tk)
                        if "error" not in r and r.get("n", 0) > 0:
                            bt_rows.append({"Ticker": tk, "Trades": r["n"],
                                "Win%": r["win_rate"], "Avg Ret%": r["avg"],
                                "Best%": r["best"], "Worst%": r["worst"],
                                "vs Buy&Hold%": round(r.get("strategy_compound_ret", 0) - r.get("buy_hold_ret", 0), 2)})
                    if bt_rows:
                        bt_df = pd.DataFrame(bt_rows)
                        st.dataframe(make_table(bt_df, {"Win%": lambda v: _sty_rs(v),
                                                         "Avg Ret%": _sty_rs,
                                                         "vs Buy&Hold%": _sty_rs}),
                                     use_container_width=True)


if __name__ == "__main__":
    main()
