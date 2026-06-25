# ╔══════════════════════════════════════════════════════════════╗
# ║   INSTITUTIONAL STOCK SCREENER  —  v3.1 (single-file)        ║
# ║   เหมือน v3.0 ทุกจุด (แก้ความแม่นยำ/ความเร็ว/backtest/         ║
# ║   alert+watchlist persist) แต่รวมกลับมาเป็นไฟล์เดียว           ║
# ║   เพื่อให้ deploy ง่ายแบบเดิม — แทนที่ app.py ไฟล์เดียว จบ       ║
# ╚══════════════════════════════════════════════════════════════╝
# สรุปการเปลี่ยนแปลงจาก v2.0 เดิม (รายละเอียดเต็มอยู่ใน docstring/comment
# ของแต่ละฟังก์ชันด้านล่าง):
#   1. ความแม่นยำ: แก้บั๊ก relative_strength เทียบ "ตำแหน่ง" ข้ามตลาดที่ปฏิทิน
#      วันเทรดต่างกัน (หุ้นไทย .BK vs SPY) + guard format ของ dividendYield
#   2. ความเร็ว/เสถียร: ลด network call ต่อ ticker, แยก cache fundamentals
#      ออกจาก cache ราคา, เพิ่ม retry+backoff, สแกนแบบ concurrent
#   3. Backtest: เข้าซื้อที่ open แท่งถัดไป (ไม่ lookahead), เทียบ Buy&Hold,
#      เพิ่ม Max Drawdown และ Sharpe โดยประมาณ
#   4. ฟีเจอร์ใหม่: watchlist persist ข้าม session จริง (เซฟลง disk) +
#      แจ้งเตือนสัญญาณใหม่ (in-app + Telegram แบบออปชัน)
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
# [merged from lib/utils.py]
# ════════════════════════════════════════════════════════
# UTILITIES — ใช้ร่วมกันทุกโมดูล
#   • logging (เหมือน v2.0 เดิม)
#   • retry decorator พร้อม exponential backoff — (ใหม่ใน v3.0)
#     เดิม v2.0 ไม่มี retry เลย ถ้า Yahoo ตอบ rate-limit/timeout ครั้งเดียว
#     หุ้นตัวนั้นจะหายไปจากผลสแกนทันทีโดยไม่มีการลองใหม่
#   • to_date_indexed() — (ใหม่ใน v3.0) ใช้ normalize index ของราคาให้เป็น
#     "วันที่" ล้วน (ไม่มี time/timezone) สำหรับเทียบ 2 ซีรีส์ที่มาจาก
#     ตลาดคนละ timezone/ปฏิทินวันเทรด (เช่นหุ้นไทย .BK เทียบกับ SPY สหรัฐฯ)
import logging
import random
import time
from functools import wraps

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("screener")


def log_err(context: str, e: Exception) -> None:
    """Log error แบบสั้น ไม่ทำให้ UI พัง แค่ไม่ให้ error หายไปเงียบๆ"""
    logger.warning("%s -> %s: %s", context, type(e).__name__, e)


def retry(times: int = 3, base_delay: float = 0.6, exceptions=(Exception,)):
    """
    Decorator: ลองใหม่แบบ exponential backoff + jitter เมื่อ network call ล่ม
    ชั่วคราว (เช่น Yahoo ตอบ 429 / timeout)

    v3.3: เพิ่ม backoff แบบยาวเป็นพิเศษเฉพาะ error ที่เป็น rate-limit จริงๆ
    (เห็นจาก log การรันจริงบน GitHub Actions ว่า "Too Many Requests" เกิดขึ้น
    เป็นชุดต่อเนื่องหลังยิง request รัวๆ — backoff สั้นแบบเดิม (<2 วินาทีรวม)
    ไม่พอให้ Yahoo คลายการบล็อก ลองใหม่กี่ครั้งก็ยังโดนซ้ำ) ตอนนี้ถ้า error
    message มีคำว่า rate limit ชัดๆ จะรอยาวขึ้นมาก (8s, 16s, 32s...) แทน
    """
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
                        if is_rate_limit:
                            delay = 8 * (2 ** attempt) + random.uniform(0, 2)
                        else:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return deco


def to_date_indexed(s: pd.Series) -> pd.Series:
    """
    Normalize index ของ Series ราคาให้เป็นวันที่ล้วน ตัด time + timezone ออก
    จำเป็นก่อนเทียบ 2 ซีรีส์ที่มี trading calendar ต่างกัน (เช่น SET ไทย vs
    NYSE สหรัฐฯ มีวันหยุดไม่ตรงกัน) ด้วย "วันที่จริง" แทนตำแหน่ง index
    """
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out = s.copy()
    out.index = idx.normalize()
    return out


# ════════════════════════════════════════════════════════
# [merged from lib/cache_store.py]
# ════════════════════════════════════════════════════════
# DISK CACHE & PERSISTENCE
#   • Scan-result cache ต่อ universe (เหมือน v2.0 เดิม ย้ายมาไว้ที่นี่)
#   • Watchlist persistence — (ใหม่ใน v3.0)
#     เดิม v2.0 watchlist อยู่ใน st.session_state ล้วนๆ → ปิดเบราว์เซอร์/รีโหลด
#     หน้าเว็บแล้วหายทันที ตอนนี้บันทึกลง disk เหมือน scan cache
#   • Last-signal snapshot — (ใหม่ใน v3.0) ใช้เทียบว่ามีหุ้นไหน "เพิ่งเปลี่ยน
#     เป็น Strong Buy/Breakout ตั้งแต่สแกนล่าสุด" เพื่อทำแถบแจ้งเตือนในแดชบอร์ด
# 
# ข้อจำกัดที่ควรรู้ (บอกตรงๆ ไม่ได้โฆษณาเกินจริง):
# Streamlit Community Cloud ใช้ container แบบ ephemeral — ไฟล์พวกนี้จะอยู่
# ข้าม "restart/sleep-wake" ตามปกติ แต่จะถูกล้างถ้า redeploy ใหม่จาก git push
# (filesystem ของ container ถูกสร้างใหม่ทั้งหมด) ถ้าต้องการ persistence แบบ
# ถาวร 100% ข้าม deploy ต้องต่อ external storage (Google Sheets/Supabase/S3)
# ซึ่งเป็นข้อจำกัดของแพลตฟอร์ม ไม่ใช่ของโค้ดส่วนนี้
import datetime
import hashlib
import json
import os
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".scan_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)

WATCHLIST_PATH = os.path.join(CACHE_DIR, "watchlist.json")
SIGNALS_DIR = os.path.join(CACHE_DIR, "last_signals")
os.makedirs(SIGNALS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SCAN-RESULT CACHE (เหมือน v2.0)
# ─────────────────────────────────────────────────────────────
def _next_refresh_time(now: datetime.datetime) -> datetime.datetime:
    bkk = ZoneInfo("Asia/Bangkok")
    now_bkk = now.astimezone(bkk)
    cutoff_today = now_bkk.replace(hour=4, minute=0, second=0, microsecond=0)
    if now_bkk >= cutoff_today:
        return cutoff_today
    return cutoff_today - datetime.timedelta(days=1)


def cache_key(universe: str, tickers: tuple, period: str, interval: str) -> str:
    raw = f"{universe}|{period}|{interval}|{','.join(sorted(tickers))}"
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe_name = "".join(c for c in universe if c.isalnum())[:20]
    return f"{safe_name}_{h}"


def load_disk_cache(universe: str, tickers: tuple, period: str, interval: str) -> Optional[pd.DataFrame]:
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


def save_disk_cache(universe: str, tickers: tuple, period: str, interval: str, df: pd.DataFrame) -> None:
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


def cache_age_label(universe: str, tickers: tuple, period: str, interval: str) -> str:
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


def clear_cache_for(universe: str, tickers: tuple, period: str, interval: str) -> bool:
    """ลบ cache ของ universe นี้ — คืนค่า True ถ้ามีไฟล์ให้ลบจริง"""
    key = cache_key(universe, tickers, period, interval)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ─────────────────────────────────────────────────────────────
# WATCHLIST PERSISTENCE (ใหม่ v3.0)
# ─────────────────────────────────────────────────────────────
def load_watchlist() -> list:
    if not os.path.exists(WATCHLIST_PATH):
        return []
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_err("load_watchlist", e)
        return []


def save_watchlist(items: list) -> None:
    try:
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
    except Exception as e:
        log_err("save_watchlist", e)


# ─────────────────────────────────────────────────────────────
# LAST-SIGNAL SNAPSHOT — สำหรับแจ้งเตือน "สัญญาณใหม่ตั้งแต่สแกนล่าสุด" (ใหม่ v3.0)
# ─────────────────────────────────────────────────────────────
def _signals_path(universe: str) -> str:
    safe = "".join(c for c in universe if c.isalnum())[:30] or "default"
    return os.path.join(SIGNALS_DIR, f"{safe}.json")


def load_last_signals(universe: str) -> dict:
    path = _signals_path(universe)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_err("load_last_signals", e)
        return {}


def save_last_signals(universe: str, mapping: dict) -> None:
    path = _signals_path(universe)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
    except Exception as e:
        log_err("save_last_signals", e)


# ════════════════════════════════════════════════════════
# [merged from lib/universes.py]
# ════════════════════════════════════════════════════════
# MODULE — UNIVERSE FETCHERS
# ย้ายมาจาก v2.0 ตรงๆ ไม่มีบั๊กในส่วนนี้ที่ต้องแก้ไข เปลี่ยนแค่ตำแหน่งไฟล์
# เพื่อให้ app.py หลักไม่ต้องยาว 1,500+ บรรทัดในไฟล์เดียว
import streamlit as st
import pandas as pd



@st.cache_data(ttl=86400)
def fetch_sp500():
    """
    v3.3: Wikipedia บล็อก request จาก IP ของ cloud/datacenter (รวม GitHub
    Actions runner) ด้วย 403 Forbidden แบบไม่สนใจ User-Agent — ยืนยันจาก log
    การรันจริง ตอนนี้ใช้ CSV ที่ดูแลโดยชุมชน (datasets/s-and-p-500-companies
    บน GitHub ซึ่งโฮสต์ผ่าน raw.githubusercontent.com ไม่ถูกบล็อกแบบเดียวกัน)
    เป็นแหล่งหลัก แล้วค่อย fallback ไป Wikipedia (เผื่อรันจาก IP ที่ไม่ถูกบล็อก
    เช่น เครื่องคุณเอง) แล้ว fallback สุดท้ายเป็น list สั้นๆกันพังทั้งหมด
    """
    try:
        import requests
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        from io import StringIO
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
                "UNH", "JNJ", "XOM", "WMT", "MA", "HD", "CVX", "MRK", "ABBV", "KO",
                "PEP", "BAC", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "ABT", "DHR",
                "LIN", "ADBE", "CRM", "NFLX", "TXN", "NEE", "PM", "WFC", "RTX", "ORCL",
                "AMD", "QCOM", "UPS", "INTC", "HON", "UNP", "LOW", "IBM", "AMGN", "SBUX"]


@st.cache_data(ttl=86400)
def fetch_nasdaq100():
    """v3.3: Wikipedia 403 บล็อกจาก cloud IP เหมือนกับ fetch_sp500 — ยังไม่เจอ
    CSV ทางเลือกที่ verified ว่าเสถียรพอสำหรับ index นี้โดยเฉพาะ จึงพยายาม
    ดึงจาก Wikipedia ก่อน (อาจสำเร็จถ้ารันจาก IP ที่ไม่ถูกบล็อก) แล้ว fallback
    เป็น list ที่ใหญ่ขึ้นมาก (~95 ตัว เทียบจาก 10 ตัวเดิม) ถ้าดึงไม่ได้จริงๆ"""
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
    return sorted(["AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","AVGO","COST",
        "NFLX","AMD","PEP","ADBE","CSCO","TMUS","INTC","CMCSA","QCOM","TXN",
        "AMAT","INTU","ISRG","HON","AMGN","BKNG","VRTX","SBUX","MDLZ","GILD",
        "ADI","REGN","PANW","LRCX","MU","PYPL","SNPS","CDNS","KLAC","MAR",
        "ORLY","CTAS","ASML","ABNB","MRVL","FTNT","CRWD","ADSK","NXPI","MNST",
        "PCAR","ROST","PAYX","KDP","ODFL","AEP","EXC","IDXX","FAST","EA",
        "CSGP","CPRT","DXCM","BIIB","GEHC","ON","MCHP","WBD","ANSS","TTD",
        "CCEP","DASH","MDB","TEAM","ZS","GFS","ILMN","WDAY","VRSK","CTSH",
        "BKR","XEL","DDOG","CDW","FANG","CHTR","LULU","MELI","EBAY","KHC",
        "TTWO","ALGN","ARM","APP","AXON","DECK","PLTR","CSX","GEN","LIN"])



@st.cache_data(ttl=86400)
def fetch_russell2000():
    return sorted(["ACVA","ALKT","ARCB","BJRI","CALX","CATO","CBRL","CLFD","COKE","CPSS",
        "CRAI","CRGY","CSWI","CVCO","DCOM","DFIN","DKNG","DNOW","DXPE","ECPG",
        "EFSC","EGHT","EPIX","ESCA","ETON","EVRI","EXPI","FBMS","FBNC","FCPT",
        "FFBC","FFIN","FISI","FIZZ","FLGT","FLNC","FMAO","FMNB","GDEN","GIII",
        "GNTY","GPOR","HAFC","HALO","HCAT","HCKT","HCSG","HIFS","HMST","HNVR",
        "HOPE","HTBK","HTLD","HURN","HWKN","HZO","IART","IBCP","IBP","IBTX",
        "ICAD","ICFI","JACK","JAMF","KALU","KLIC","KNSL","KTOS","LBRT","LCII",
        "LDOS","LECO","LEVI","LGND","LMAT","LMND","LNTH","LOCO","LUNA","LYFT",
        "MATX","MBLY","MEDP","MGNI","MLKN","MMSI","MORN","MRTN","MTSI","NABL",
        "NARI","NATI","NMIH","NOVT","NSIT","NTNX","NVST","OCGN","OMCL","ONTO",
        "OPCH","OSIS","PACK","PAHC","PCOR","PCRX","PDCO","PENN","PGNY","PLXS",
        "PODD","POWL","PRDO","PRGS","PRIM","PRLD","PSMT","PSTG","PTCT","PUMP",
        "QDEL","QTWO","RAMP","RARE","RCKT","RDNT","RGEN","RIOT","RNST","ROCK",
        "RPRX","RYTM","SAFE","SAGE","SAIA","SATS","SBCF","SFNC","SHLS","SHOO",
        "SILK","SITM","SKYW","SMCI","SMPL","SNOW","SNPS","SOUN","SPSC","STAA",
        "STNE","STRL","SUMO","SUPN","SWAV","SWKS","TASK","TDOC","TMDX","TORC",
        "TRMK","TRNO","TROW","TRST","TTGT","TTMI","TWST","UBCP","UCTT","UDMY",
        "ULCC","UNFI","UPST","USNA","USTR","VBTX","VERA","VIAV","VIRT","VLCN",
        "VNDA","VRNS","VRNT","VSEC","VSTO","WAFD","WERN","WEYS","WINA","WKME",
        "WOLF","WOOF","WSFS","WTFC","XPEL","XPOF","YELP","ZEUS","ZLAB","ZYXI"])


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
        "QQQ","QQQM","SOXX","SMH","HACK","IGV","WCLD","IWM","IWO","MDY","IJR",
        "EEM","EWJ","EWZ","FXI","VEA","VWO","INDA","TUR","EWY","EWT",
        "ARKK","ARKQ","ARKG","ARKF","ARKW","BOTZ","ROBO","AIQ",
        "GLD","SLV","GDX","GDXJ","USO","COPX",
        "TQQQ","SOXL","SPXL","TLT","HYG","LQD","EMB",
        "VYM","SCHD","VIG","NOBL"])


@st.cache_data(ttl=86400)
def fetch_broad_us():
    sp = fetch_sp500()
    nd = fetch_nasdaq100()
    extra = ["AEHR","ALEC","AMBA","AMKR","APPF","ARWR","ATRC","AZEK","BILL",
             "BIRK","BLKB","BURL","CACC","CAKE","CALM","CARG","CELH","CENTA",
             "CHDN","CHEF","CHUY","CIVI","CLFD","COMP","COOP","CRDO","CROX",
             "CWST","DAKT","DDOG","DFIN","DKNG","DLTH","DOCN","DOCS","DOOR",
             "DRVN","DXCM","EDIT","EGHT","ENVA","EPAM","ESAB","EVGO","EWBC",
             "EXAS","EXEL","EXPI","FELE","FIGS","FIZZ","FOUR","FROG","GRND",
             "HIMS","HLIT","HUBS","HWKN","IART","IIPR","IMVT","INDB","INFA",
             "INST","IONS","IRTC","ITCI","JACK","JAMF","JOBY","KLIC","KNSL",
             "KTOS","KVYO","LBRT","LEVI","LGND","LMND","LOCO","LYFT","MATX",
             "MBLY","MEDP","MGNI","MLAB","MLKN","MMSI","MORN","MPWR","MRTN",
             "MTSI","NARI","NATI","NKTR","NOVT","NSIT","NTNX","OCGN","OMCL",
             "ONTO","OPCH","PACK","PCOR","PENN","PGNY","PLXS","PODD","POWL",
             "PRDO","PRGS","PRIM","PSMT","PSTG","PUMP","QDEL","QTWO","RAMP",
             "RARE","RCKT","RDNT","RGEN","RIOT","ROCK","RPRX","RYTM","SAGE",
             "SAIA","SATS","SHLS","SHOO","SILK","SITM","SKYW","SMCI","SMPL",
             "SOUN","SPSC","STAA","STNE","STRL","SUMO","SWAV","SWKS","TASK",
             "TDOC","TMDX","TRMK","TROW","TTGT","TTMI","TWST","UCTT","UDMY",
             "ULCC","UPST","USNA","VERA","VIAV","VIRT","VRNS","VRNT","VSEC",
             "WAFD","WERN","WEYS","WINA","WOLF","WSFS","WTFC","XPEL","YELP"]
    return sorted(set(sp + nd + extra))


SECTOR_MAP = {
    "Technology | เทคโนโลยี":     ["AAPL","MSFT","NVDA","GOOGL","META","AVGO","ORCL","AMD","QCOM","TXN","AMAT","MU","LRCX","KLAC","CDNS","SNPS","NXPI","MCHP","ADI","FTNT"],
    "Healthcare | สุขภาพ":        ["UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN","ISRG","VRTX","REGN","GILD","CVS","CI","ELV","HCA","IDXX","DXCM"],
    "Financials | การเงิน":       ["JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","USB","PNC","COF","TFC","MCO","SPGI","ICE","CME","AON","MMC","CB","PGR"],
    "Consumer | สินค้าอุปโภค":    ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","BKNG","MAR","HLT","YUM","DRI","ROST","TJX","ULTA","LULU","DKNG","WYNN","CZR"],
    "Industrials | อุตสาหกรรม":   ["GE","HON","RTX","LMT","BA","CAT","DE","UPS","FDX","UNP","CSX","NSC","EMR","ETN","PH","ROK","IR","XYL","CARR","OTIS"],
    "Energy | พลังงาน":           ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","PXD","FANG","HAL","BKR","DVN","HES","APA","CTRA","MRO","OXY","WMB","KMI"],
    "Comm Svcs | สื่อสาร":        ["NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","PARA","FOX","FOXA","WBD","EA","TTWO","RBLX","MTCH","IAC","ZG","ANGI","LYFT","UBER"],
    "Real Estate | อสังหาริมทรัพย์": ["AMT","PLD","CCI","EQIX","PSA","DLR","O","WELL","AVB","EQR","SPG","VTR","ARE","BXP","KIM","REG","NNN","WPC","COLD","IIPR"],
    "Utilities | สาธารณูปโภค":    ["NEE","DUK","SO","D","SRE","AEP","XEL","PCG","EIX","WEC","ES","ETR","FE","PPL","CMS","AES","NI","EVRG","CNP","LNT"],
    "Materials | วัสดุ":          ["LIN","APD","ECL","DD","PPG","NEM","FCX","NUE","VMC","MLM","ALB","BALL","IP","CF","MOS","FMC","CE","RPM","ATI","CMC"],
    "ETFs | กองทุน ETF":          ["SPY","QQQ","IWM","XLK","XLF","XLE","XLV","XLI","XLP","XLU","GLD","TLT","HYG","EEM","EWJ","ARKK","SOXL","TQQQ","VYM","SCHD"],
    "🚀 Space | อวกาศ":              ["RKLB","LMT","NOC","BA","RTX","ASTS","SPCE","LUNR","RDW","KTOS","IRDM","VSAT","MAXR","ASTR","PL","TDY"],
    "🤖 AI | ปัญญาประดิษฐ์":         ["NVDA","MSFT","GOOGL","META","AMD","PLTR","SMCI","AVGO","ARM","AI","SNOW","PATH","BBAI","SOUN","UPST","CRM"],
    "💊 Biotech/Pharma | ยา/ไบโอเทค": ["LLY","UNH","JNJ","MRK","ABBV","VRTX","REGN","GILD","AMGN","MRNA","BNTX","ISRG","BIIB","ALNY","SRPT","RARE"],
    "🏦 Banking | ธนาคาร":           ["JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","COF","SCHW","BK","STT","FITB","RF","KEY"],
    "⚡ EV/Battery | ไฟฟ้า/แบตเตอรี่": ["TSLA","RIVN","LCID","NIO","LI","XPEV","ALB","LTHM","ENVX","QS","FREY","CHPT","BLNK","PLUG","FCEL","STEM"],
    "🎮 Gaming/Streaming | เกม/สตรีมมิ่ง": ["NFLX","DIS","RBLX","EA","TTWO","NTDOY","SONY","SPOT","PARA","WBD","ATVI","U","RICK","GME","HUYA","DOYU"],
    "🔒 Crypto/Cyber | คริปโต/ไซเบอร์": ["COIN","MSTR","MARA","RIOT","HUT","CLSK","BITF","CRWD","PANW","ZS","FTNT","OKTA","S","NET","CYBR","TENB"],
    "🏠 REIT | กองทุนอสังหา":         ["O","PLD","AMT","EQIX","PSA","DLR","SPG","AVB","EQR","WELL","VTR","ARE","BXP","KIM","REG","IIPR"],
}

UNIVERSE_OPTIONS = {
    "S&P 500 (503)": fetch_sp500,
    "Nasdaq 100 (101)": fetch_nasdaq100,
    "Russell 2000 Small Cap": fetch_russell2000,
    "US Broad Market (~700)": fetch_broad_us,
    "หุ้นไทย SET/mai": fetch_set,
    "ETF Screener (70)": fetch_etfs,
    "Sector Focus | เลือกตามหมวด": None,
    "Custom Tickers": None,
}


def resolve_tickers(universe: str, sector_choice: list, custom_input: str) -> list:
    """single source of truth สำหรับ resolve รายชื่อ ticker (เหมือน v2.0)"""
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
# [merged from lib/indicators.py]
# ════════════════════════════════════════════════════════
# MODULE — MATH ENGINE
# ทุกฟังก์ชันเหมือน v2.0 เดิม ยกเว้น relative_strength() ที่แก้บั๊กการเทียบวันที่
# (ดู docstring ของฟังก์ชันนั้นสำหรับรายละเอียด)
import numpy as np
import pandas as pd



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
    if delta < -0.4:
        lbl = "🔥 Squeezing"
    elif delta < 0:
        lbl = "⚡ Tightening"
    elif delta < 0.6:
        lbl = "🌱 Just Broke"
    else:
        lbl = "📈 Expanding"
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
    """
    เทียบ % การเปลี่ยนแปลงของหุ้นกับ benchmark (เช่น SPY) ใน N แท่งล่าสุด

    FIX (v3.0) — เดิม v2.0 เทียบโดยใช้ "ตำแหน่ง" (closes.iloc[-period] vs
    spy.iloc[-period]) ตรงๆ ระหว่าง 2 ซีรีส์ ซึ่งถูกต้องเฉพาะกรณีทั้งคู่มี
    ปฏิทินวันเทรดเหมือนกันทุกวันเท่านั้น (เช่น หุ้นสหรัฐฯ เทียบกับ SPY ซึ่งใช้
    ปฏิทิน NYSE เหมือนกัน) แต่ผิดทันทีถ้าเทียบ "หุ้นไทย .BK" กับ SPY เพราะ
    วันหยุดตลาดไทยกับสหรัฐฯ ไม่ตรงกัน ทำให้ "20 แท่งที่แล้ว" ของหุ้นไทยกับของ
    SPY ไม่ใช่วันเดียวกันจริง — ค่า RS ที่ได้คลาดเคลื่อนโดยไม่มี error ใดๆ
    ขึ้นเตือนเลย (silent bug)

    ตอนนี้ join ทั้งสองซีรีส์ด้วย "วันที่จริง" ก่อนคำนวณ (ผ่าน
    utils.to_date_indexed) เพื่อให้แน่ใจว่าเทียบช่วงเวลาเดียวกันเสมอ ไม่ว่า
    หุ้นจะมาจากตลาดไหน
    """
    if closes is None or bench is None:
        return np.nan
    if len(closes) < 2 or len(bench) < 2:
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
    s = min(pat_score, 4)
    s += min(acc_score, 3)
    if 1.1 <= vol20 <= 2.0:
        s += 1
    if 40 <= rsi <= 62:
        s += 1
    if isinstance(mktcap_b, float) and 0 < mktcap_b < 10:
        s += 1
    s = min(s, 10)
    lbl = "💎 Hidden Gem" if s >= 8 else "🔭 Emerging Gem" if s >= 6 else "👀 Watch" if s >= 4 else "—"
    return s, lbl


def strategy_signal(price, e200, e50, rsi, vol20, macd_h, stars) -> str:
    p200 = (price - e200) / e200 * 100 if e200 > 0 else 999
    if len(stars) >= 3 and rsi < 40 and vol20 > 1.8 and macd_h > 0 and -5 <= p200 <= 3:
        return "🔥 Strong Buy"
    if vol20 > 2.0 and price > e50 > e200 and macd_h > 0 and 50 <= rsi <= 75:
        return "🚀 Breakout"
    if price > e50 > e200 and 40 <= rsi <= 70:
        return "📈 ขาขึ้น"
    if abs(p200) <= 3 and rsi < 50 and macd_h < 0:
        return "⚠️ เฝ้าระวัง"
    if rsi > 75:
        return "⏳ รอ Pullback"
    if price < e200:
        return "⚠️ Oversold Bear" if rsi < 30 else "❌ ขาลง"
    return "🔄 Neutral"


def conservative_stars(price, e200, rsi, vol20, drawdown) -> str:
    s = 0
    if e200 > 0 and abs((price - e200) / e200 * 100) <= 2:
        s += 1
    if rsi < 35:
        s += 1
    if vol20 > 2.0:
        s += 1
    if -15 <= drawdown <= -5:
        s += 1
    return "⭐" * s if s else "—"


# ════════════════════════════════════════════════════════
# [merged from lib/analyzer.py]
# ════════════════════════════════════════════════════════
# MODULE — SINGLE TICKER PIPELINE + BATCH PROCESSOR
# 
# เปลี่ยนจาก v2.0 (รายละเอียดอยู่ในแต่ละ docstring):
#   1. ดึง fundamentals (.info) แยก cache จากราคา/เทคนิคัล + ดึงรอบเดียว
#      (เดิมยิงทั้ง .fast_info และ .info แยกกัน = 2 network call ต่อ ticker
#      ต่อสแกน ทั้งที่ fundamentals ไม่ได้เปลี่ยนรายวัน)
#   2. dividendYield ใช้ guard ตาม magnitude แทนการ assume format คงที่
#      (Yahoo เคยเปลี่ยน format ของ field นี้มาแล้ว — เห็นได้จาก GitHub issues
#      หลายอันใน ranaroussi/yfinance — โค้ดเดิมคูณ 100 เสมอ ถ้า field เปลี่ยน
#      มาเป็น % อยู่แล้วจะได้ yield ผิดเพี้ยนไปมาก)
#   3. retry + exponential backoff ทุก network call (เดิมไม่มี retry เลย)
#   4. batch_scan ใช้ ThreadPoolExecutor ยิง concurrent (เดิม sequential
#      ทีละตัว + sleep คงที่ — ช้าและไม่จำเป็น เพราะงานนี้เป็น I/O-bound)
#   5. relative_strength เรียกด้วยซีรีส์ที่มี date index จริง (ดู indicators.py)
#      แทนการส่ง tuple ของค่าดิบที่ไม่มีวันที่กำกับ
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf



@retry(times=3, base_delay=0.6)
def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)


@st.cache_data(ttl=3600)
def _cached_history(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = _download_history(ticker, period, interval)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        log_err(f"history({ticker})", e)
        return None


def _normalize_dividend_yield(raw) -> float:
    """
    Yahoo เคยเปลี่ยน format ของ dividendYield ไปมา (ทศนิยมเช่น 0.024 บางช่วง
    เทียบเท่า 2.4% แต่บางเวอร์ชันคืนค่าเป็น % ตรงๆ คือ 2.4 อยู่แล้ว) เดิม
    v2.0 คูณ 100 เสมอ — ถ้า field เปลี่ยนมาเป็น % แล้วจะได้ yield ผิดเป็น
    240% ทันทีแบบไม่มี error เตือน

    Guard ตรงนี้ใช้ magnitude เป็นตัวตัดสิน: ถ้าค่าที่ได้น้อยกว่า 1 ถือว่า
    เป็นทศนิยม (คูณ 100) ถ้ามากกว่า 1 ถือว่าเป็น % อยู่แล้ว — robust กว่า
    การ assume format คงที่ ไม่ว่า yfinance/Yahoo จะเปลี่ยน field นี้อีกกี่ครั้ง
    """
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
def _download_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}


def _safe_num(val, decimals=2):
    """แปลงค่าเป็น float อย่างปลอดภัย — เคยพบว่า field บางตัวจาก Yahoo (เช่น P/E
    ของ BILL) คืนมาเป็น string แทนตัวเลข ทำให้ round() พังทั้งฟังก์ชันและ field
    อื่นที่ดีอยู่แล้วก็พลอยหายไปด้วย (v3.3 แก้ — เช็คทีละ field แทน)"""
    if val is None:
        return np.nan
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return np.nan


@st.cache_data(ttl=21600)  # 6 ชม. — fundamentals เปลี่ยนช้ากว่าราคามาก ไม่ต้องดึงซ้ำทุกสแกน
def _cached_fundamentals(ticker: str) -> dict:
    try:
        info = _download_info(ticker)
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        mktcap = info.get("marketCap")
        mktcap_b = (mktcap / 1e9) if isinstance(mktcap, (int, float)) else np.nan
        return {
            "pe": _safe_num(pe),
            "pb": _safe_num(pb),
            "div": _normalize_dividend_yield(info.get("dividendYield")),
            "mktcap_b": _safe_num(mktcap_b),
        }
    except Exception as e:
        log_err(f"fundamentals({ticker})", e)
        return {"pe": np.nan, "pb": np.nan, "div": np.nan, "mktcap_b": np.nan}


@st.cache_data(ttl=3600)
def analyze(ticker: str, period: str = "1y", interval: str = "1d", bench_tuple=None) -> Optional[dict]:
    """
    bench_tuple: tuple ของ (date_iso_string, close) ของ benchmark (เช่น SPY)
    เปลี่ยนจาก v2.0 ที่ส่งเป็น tuple ค่าดิบไม่มีวันที่กำกับ — จำเป็นสำหรับ
    relative_strength() เวอร์ชันใหม่ที่ join ด้วยวันที่จริง
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

        trend = "🟢 Bull" if px > ep[200] else "🔴 Bear"
        patt = candle_pattern(df)
        stars = conservative_stars(px, ep[200], rsi_val, vm20 or 0, draw or 0)
        sig = strategy_signal(px, ep[200], ep[50], rsi_val, vm20 or 0, mh, stars)

        ep_lbl, ep_sc = ema_pattern(px, ep[5], ep[10], ep[20], ep[50], ep[100], ep[200])
        acc_sc, acc_lb = quiet_accumulation(vl, cl, rsi_val)
        sq_lbl, bw_now, bw_delta = squeeze_direction(cl)
        age = signal_age(cl)

        rs20 = rs50 = np.nan
        if bench_tuple:
            dates, vals = zip(*bench_tuple)
            bench = pd.Series(vals, index=pd.to_datetime(dates))
            rs20 = relative_strength(cl, bench, 20)
            rs50 = relative_strength(cl, bench, 50)

        fnd = _cached_fundamentals(ticker)
        gs, gl = gem_score(ep_sc, acc_sc, vm20 or 0, rsi_val, draw or 0, fnd["mktcap_b"])

        return {
            "Ticker": ticker, "Price": round(px, 2), "ราคาปิด": prev_c,
            "Trend": trend, "Signal": sig, "Phase": ep_lbl, "Stars": stars,
            "EMA5": round(ep[5], 2), "EMA10": round(ep[10], 2), "EMA20": round(ep[20], 2),
            "EMA50": round(ep[50], 2), "EMA100": round(ep[100], 2), "EMA200": round(ep[200], 2),
            "vs EMA5%": ed[5], "vs EMA10%": ed[10], "vs EMA20%": ed[20],
            "vs EMA50%": ed[50], "vs EMA100%": ed[100], "vs EMA200%": ed[200],
            "RSI": rsi_val, "MACD": ml, "Signal_L": ms, "MACD_H": mh,
            "Vol×20D": vm20, "Vol×3M": vm3m, "Vol×6M": vm6m,
            "YTD%": ytd_ret, "Drawdown%": draw, "High52W": round(hi52, 2),
            "vs52W%": round((px - hi52) / hi52 * 100, 2) if hi52 > 0 else np.nan,
            "Candle": patt, "EMA Pattern": ep_lbl, "Pat Score": ep_sc,
            "Accum": acc_lb, "Accum Score": acc_sc, "Gem Score": gs, "💎 Gem": gl,
            "Squeeze": sq_lbl, "BW%": bw_now, "BW Δ5d": bw_delta, "Signal Age": age,
            "RS 20D": rs20, "RS 50D": rs50,
            "P/E": fnd["pe"], "P/BV": fnd["pb"], "Div%": fnd["div"], "MktCap$B": fnd["mktcap_b"],
        }
    except Exception as e:
        log_err(f"analyze({ticker})", e)
        return None


def make_bench_tuple(bench_df: pd.DataFrame) -> tuple:
    """แปลง DataFrame ราคาของ benchmark (เช่น SPY) เป็น tuple ของ (date_iso, close)
    เพื่อให้ผ่าน st.cache_data ได้ (ต้อง hashable) พร้อมคงวันที่ไว้สำหรับ
    relative_strength() เวอร์ชันใหม่ — เดิม v2.0 ส่งแค่ tuple(values) ทำให้
    วันที่หายไปตั้งแต่จุดนี้"""
    idx = pd.to_datetime(bench_df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return tuple(zip(idx.strftime("%Y-%m-%d"), bench_df["Close"].values.tolist()))


def batch_scan(
    tickers: tuple,
    period: str = "1y",
    interval: str = "1d",
    bench_tuple=None,
    max_workers: int = 6,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    เดิม v2.0 สแกนทีละตัว sequential (sleep 0.4 วินาทีทุกๆ 25 ตัว) — สแกน
    300 ตัวต้องรอ network round-trip ของตัวก่อนหน้าจบก่อนถึงจะเริ่มตัวต่อไป
    ตอนนี้ใช้ ThreadPoolExecutor ยิง concurrent เพราะงานนี้เป็น I/O-bound
    (รอ network) ไม่ใช่ CPU-bound — max_workers ถูกจำกัดไว้ไม่สูงเกินไป
    เพื่อลดความเสี่ยงโดน Yahoo rate-limit จาก request ที่ถี่เกินไป
    """
    results = []
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze, tk, period, interval, bench_tuple): tk for tk in tickers}
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


def fetch_live(ticker: str) -> dict:
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
# [merged from lib/backtest.py]
# ════════════════════════════════════════════════════════
# MODULE — BACKTESTER
# 
# เปลี่ยนจาก v2.0:
#   1. เข้าซื้อที่ "ราคาเปิดของแท่งถัดไป" (i+1) ไม่ใช่ "ราคาปิดของแท่งที่เกิด
#      สัญญาณ" (i) — เดิมใช้ close ของแท่งเดียวกับที่คำนวณสัญญาณ ซึ่งในทาง
#      ปฏิบัติเทรดจริงทำไม่ได้ (รู้ว่าสัญญาณเกิดก็ต่อเมื่อแท่งนั้นปิดแล้ว)
#   2. เพิ่ม Buy & Hold ของหุ้นตัวเดียวกัน ช่วงเวลาเดียวกัน เป็น benchmark
#      เทียบ — เดิมดู win rate ลอยๆ ไม่รู้ว่ากลยุทธ์ดีกว่า "ถือเฉยๆ" จริงไหม
#   3. เพิ่ม Max Drawdown (จาก equity curve ของ trade ที่ compound ต่อกัน)
#      และ Sharpe ratio แบบประมาณการจาก distribution ของ trade returns
#   4. ระบุข้อจำกัดของ backtest นี้ตรงๆ ในผลลัพธ์ (ดู key "notes")
# 
# ข้อจำกัดที่ยังมีอยู่ (ไม่ได้ทำให้ backtest นี้สมบูรณ์แบบ บอกตรงๆ):
#   • ไม่หักค่าคอมมิชชั่น/สเปรด/สลิปเพจ
#   • ทดสอบบนหุ้นที่ "ยังอยู่ใน index วันนี้" เท่านั้น → survivorship bias
#   • Sharpe คำนวณจาก distribution ของ trade returns ไม่ใช่ daily returns
#     แบบเข้มงวด ถือเป็นค่าประมาณ ไม่ใช่ Sharpe ที่ใช้เทียบกับกองทุนจริงได้
#   • กลยุทธ์เดียว ผลย้อนหลังไม่ใช่การันตีผลในอนาคต ไม่ใช่คำแนะนำการลงทุน
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


BACKTEST_NOTES = (
    "ไม่หักค่าคอมมิชชั่น/สเปรด · ทดสอบบนหุ้นที่ยังอยู่ใน index วันนี้เท่านั้น "
    "(survivorship bias) · Sharpe เป็นค่าประมาณจาก trade returns ไม่ใช่ "
    "daily returns แบบเข้มงวด · ผลย้อนหลังไม่ใช่การันตีอนาคต ไม่ใช่คำแนะนำการลงทุน"
)


@retry(times=3, base_delay=0.6)
def _download_2y(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)


@st.cache_data(ttl=86400)
def backtest(ticker: str, hold_days: int = 20) -> dict:
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
                entry_price = op.iloc[i + 1]  # เข้าซื้อที่ open ของแท่งถัดไป ไม่ใช่ close วันนี้
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

        bh_start = cl.iloc[200]
        bh_end = cl.iloc[-1]
        bh_ret = round((bh_end - bh_start) / bh_start * 100, 2)

        if not trades:
            return {
                "n": 0, "win_rate": 0, "avg": 0, "best": 0, "worst": 0, "trades": [],
                "buy_hold_ret": bh_ret, "max_drawdown": 0, "sharpe": None, "notes": BACKTEST_NOTES,
            }

        rets = [t["ret"] for t in trades]
        wins = [r for r in rets if r > 0]

        equity = [1.0]
        for r in rets:
            equity.append(equity[-1] * (1 + r / 100))
        equity = np.array(equity)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        max_dd = round(float(drawdowns.min()), 2)

        ann_factor = 252 / hold_days if hold_days > 0 else 1
        mean_r, std_r = float(np.mean(rets)), float(np.std(rets))
        sharpe = round((mean_r / std_r) * np.sqrt(ann_factor), 2) if std_r > 0 else None
        total_compound_ret = round((equity[-1] - 1) * 100, 2)

        return {
            "n": len(trades), "win_rate": round(len(wins) / len(trades) * 100, 1),
            "avg": round(mean_r, 2), "median": round(float(np.median(rets)), 2),
            "best": round(max(rets), 2), "worst": round(min(rets), 2),
            "trades": rets, "trade_details": trades,
            "buy_hold_ret": bh_ret, "strategy_compound_ret": total_compound_ret,
            "max_drawdown": max_dd, "sharpe": sharpe, "notes": BACKTEST_NOTES,
        }
    except Exception as e:
        log_err(f"backtest({ticker})", e)
        return {"error": str(e)}


# ════════════════════════════════════════════════════════
# [merged from lib/styles.py]
# ════════════════════════════════════════════════════════
# MODULE — STYLES & UI HELPERS
# ย้ายมาจาก v2.0 ตรงๆ (CSS theme, dataframe style functions, info_card)
import streamlit as st

CSS_BLOCK = """
<style>
/* ── BASE ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background:#0d1117 !important; }
.main .block-container { padding: 1.2rem 2rem 2rem 2rem !important; max-width:100% !important; }

/* ── ALL TEXT defaults ── */
p, span, div, label, li, td, th { color:#e6edf3 !important; }
h1,h2,h3,h4,h5,h6 { color:#ffffff !important; font-weight:700 !important; line-height:1.3 !important; }
strong, b { color:#ffffff !important; }
small, .stCaption p { color:#8b949e !important; font-size:0.78rem !important; }
code { color:#79c0ff !important; background:#161b22 !important; padding:1px 5px !important; border-radius:4px !important; }
hr { border-color:#21262d !important; margin:1rem 0 !important; }

/* ── METRIC CARDS — CRITICAL: force bright values ── */
div[data-testid="metric-container"] {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    border-radius:10px !important;
    padding:14px 18px !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span,
[data-testid="stMetricLabel"] div {
    color:#8b949e !important;
    font-size:0.72rem !important;
    font-weight:600 !important;
    text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] span {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    font-size:1.6rem !important;
    font-weight:800 !important;
    line-height:1.25 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:#161b22 !important;
    border-radius:8px !important;
    padding:4px !important;
    gap:2px !important;
}
.stTabs [data-baseweb="tab"] {
    color:#8b949e !important;
    font-weight:600 !important;
    font-size:0.85rem !important;
    border-radius:6px !important;
    padding:7px 16px !important;
    background:transparent !important;
}
.stTabs [aria-selected="true"] {
    background:#238636 !important;
    color:#ffffff !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#e6edf3 !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background:#161b22 !important;
    border-right:1px solid #21262d !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color:#e6edf3 !important; }

/* ── INPUTS ── */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background:#21262d !important;
    border-color:#30363d !important;
}
.stSelectbox span, .stMultiSelect span { color:#e6edf3 !important; }
.stTextArea textarea, .stTextInput input {
    background:#21262d !important;
    color:#e6edf3 !important;
    border-color:#30363d !important;
}
.stSlider [data-testid="stThumbValue"] span { color:#ffffff !important; }

/* ── BUTTONS ── */
.stButton > button {
    background:linear-gradient(135deg,#238636,#2ea043) !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:8px !important;
    font-weight:700 !important;
    font-size:0.88rem !important;
    padding:9px 18px !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#2ea043,#3fb950) !important;
    box-shadow:0 4px 14px rgba(46,160,67,0.35) !important;
}

/* ── EXPANDER ── */
details { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; }
details summary { color:#c9d1d9 !important; font-weight:600 !important; padding:10px 14px !important; }
details summary:hover { color:#ffffff !important; }

/* ── DATAFRAME ── */
.stDataFrame { border-radius:8px !important; overflow:hidden !important; }

/* ── ALERTS ── */
.stAlert, [data-testid="stNotification"] {
    background:#1c2128 !important;
    border-color:#30363d !important;
}
.stAlert p { color:#e6edf3 !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color:#2ea043 !important; }

/* ── PROGRESS BAR ── */
.stProgress > div > div { background:#238636 !important; }

/* ── HIDE CHROME ── */
#MainMenu, footer, .stDeployButton { display:none !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS_BLOCK, unsafe_allow_html=True)


def _sty_signal(v):
    v = str(v)
    if "Strong Buy" in v or "Hidden Gem" in v: return "color:#3fb950;font-weight:800;"
    if "Breakout" in v or "เบรคเอาท์" in v:   return "color:#f7b731;font-weight:700;"
    if "Uptrend"  in v or "ขาขึ้น" in v:       return "color:#3fb950;font-weight:600;"
    if "Avoid" in v or "ขาลง" in v:             return "color:#f85149;font-weight:700;"
    if "Watch" in v or "เฝ้าระวัง" in v:       return "color:#d29922;font-weight:600;"
    if "Squeeze" in v:                           return "color:#ab7df8;font-weight:700;"
    if "Accum" in v or "Stealth" in v:           return "color:#26c6da;font-weight:700;"
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
    if "Emerging" in v:   return "color:#3fb950;font-weight:700;"
    if "Watch" in v:      return "color:#d29922;font-weight:600;"
    return "color:#8b949e;"


def _sty_squeeze(v):
    v = str(v)
    if "Squeezing" in v:  return "color:#ab7df8;font-weight:800;"
    if "Tightening" in v: return "color:#79c0ff;font-weight:700;"
    if "Just Broke" in v: return "color:#3fb950;font-weight:700;"
    if "Expanding" in v:  return "color:#f7b731;font-weight:600;"
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


def _sty_wr(v):
    try:
        f = float(v)
        if f >= 60: return "color:#3fb950;font-weight:700;"
        if f >= 50: return "color:#79c0ff;"
        return "color:#f85149;"
    except Exception:
        return ""


BASE_TBL = {
    "background-color": "#161b22",
    "color": "#e6edf3",
    "border": "1px solid #21262d",
    "font-size": "13px",
    "padding": "5px 10px",
}
HDR_TBL = [{"selector": "th", "props": [
    ("background-color", "#21262d"), ("color", "#ffffff"),
    ("font-weight", "700"), ("font-size", "11px"),
    ("padding", "8px 10px"), ("text-transform", "uppercase"),
    ("letter-spacing", "0.05em"),
]}]


def make_table(df, style_map: dict = None) -> object:
    """Apply consistent dark styling + optional column-level styling."""
    s = df.style.set_properties(**BASE_TBL).set_table_styles(HDR_TBL).hide(axis="index")
    if style_map:
        for col, fn in style_map.items():
            if col in df.columns:
                s = s.map(fn, subset=[col])
    return s


def info_card(label: str, value: str, color="#ffffff", sub="") -> str:
    """Compact HTML metric card — guaranteed readable."""
    sub_html = f'<div style="color:#8b949e;font-size:0.75rem;margin-top:3px;">{sub}</div>' if sub else ""
    return (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
            f'padding:14px 16px;min-width:110px;">'
            f'<div style="color:#8b949e;font-size:0.7rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:6px;">{label}</div>'
            f'<div style="color:{color};font-size:1.45rem;font-weight:800;line-height:1.2;">{value}</div>'
            f'{sub_html}'
            f'</div>')


# ════════════════════════════════════════════════════════
# [merged from lib/tv_chart.py]
# ════════════════════════════════════════════════════════
# MODULE — TRADINGVIEW WIDGET (relocated unchanged from v2.0)


def tv_chart(ticker: str, height: int = 620, interval: str = "D") -> None:
    import streamlit.components.v1 as components

    nyse = {"JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "CVX", "WMT",
            "KO", "PFE", "MRK", "T", "VZ", "IBM", "GE", "GM", "F", "GS", "MS", "C", "WFC"}
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
            "RSI@tv-basicstudies","MACD@tv-basicstudies"
        ]
    }}
    </script></div></div>"""
    components.html(html, height=height + 10, scrolling=False)


# ════════════════════════════════════════════════════════
# [merged from lib/sector_view.py]
# ════════════════════════════════════════════════════════
# MODULE — SECTOR HEATMAP (เหมือน v2.0 logic เดิม ย้ายมาไว้แยกไฟล์)
import numpy as np
import pandas as pd
import streamlit as st



@st.cache_data(ttl=3600)
def sector_heatmap_data() -> pd.DataFrame:
    """สรุปคะแนนเฉลี่ยต่อ Sector — ใหม่ v3.2: ใช้ข้อมูลจาก bundle ที่ดึงไว้
    ล่วงหน้าก่อน (เพราะ SECTOR_MAP tickers ถูกรวมอยู่ใน fetch_data.py แล้ว)
    เรียก analyze() สดเฉพาะตอนไม่มี bundle เท่านั้น (กันยิง Yahoo ซ้ำ)"""
    _, bundle_df = load_prefetched_bundle()
    use_bundle = bundle_df is not None and not bundle_df.empty and "Ticker" in bundle_df.columns

    rows = []
    for sector, tickers in SECTOR_MAP.items():
        sample = tickers[:5]
        scores = []
        if use_bundle:
            sub = bundle_df[bundle_df["Ticker"].isin(sample)]
            for _, d in sub.iterrows():
                scores.append({
                    "gem": d.get("Gem Score", 0) or 0,
                    "accum": d.get("Accum Score", 0) or 0,
                    "rs20": d.get("RS 20D", 0) or 0,
                    "bull": 1 if "Bull" in str(d.get("Trend", "")) else 0,
                })
        else:
            for tk in sample:
                d = analyze(tk)
                if d:
                    scores.append({
                        "gem": d.get("Gem Score", 0) or 0,
                        "accum": d.get("Accum Score", 0) or 0,
                        "rs20": d.get("RS 20D", 0) or 0,
                        "bull": 1 if "Bull" in str(d.get("Trend", "")) else 0,
                    })
        if scores:
            rows.append({
                "Sector": sector,
                "Avg Gem Score": round(np.mean([s["gem"] for s in scores]), 1),
                "Avg Accum": round(np.mean([s["accum"] for s in scores]), 1),
                "Avg RS 20D": round(np.mean([s["rs20"] for s in scores]), 1),
                "Bull %": round(np.mean([s["bull"] for s in scores]) * 100, 0),
                "Sample": ", ".join(sample),
            })
    return pd.DataFrame(rows).sort_values("Avg Gem Score", ascending=False)


# ════════════════════════════════════════════════════════
# [merged from lib/alerts.py]
# ════════════════════════════════════════════════════════
# MODULE — ALERTS (ใหม่ใน v3.0)
# 
# ฟีเจอร์ที่ขอเพิ่ม "แจ้งเตือน" ทำเป็น 2 ชั้น:
#   1. ในแอปเอง (ไม่ต้องตั้งค่าอะไรเพิ่ม) — เทียบสัญญาณของสแกนรอบนี้กับ
#      สแกนรอบล่าสุดที่บันทึกไว้ (cache_store.load_last_signals) แล้วโชว์ว่า
#      มีหุ้นไหนเพิ่ง "กลายเป็น Strong Buy / Breakout" ตั้งแต่รอบก่อน
#   2. Telegram push (ออปชันแล้วแต่ผู้ใช้) — ถ้าตั้งค่า secrets
#      TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ไว้ใน .streamlit/secrets.toml
#      ระบบจะส่งข้อความแจ้งเตือนออกไปด้วย ถ้าไม่ตั้งค่าไว้ ฟังก์ชันจะ no-op
#      เงียบๆ ไม่ error และไม่บังคับให้ต้องมี Bot
from typing import Optional

import pandas as pd
import streamlit as st


NOTABLE_SIGNALS = ("🔥 Strong Buy", "🚀 Breakout")


def detect_new_signals(current_df: pd.DataFrame, last_signals: dict) -> list:
    """คืนรายการ dict {ticker, signal} ที่เพิ่งเปลี่ยนเป็นสัญญาณเด่น
    (Strong Buy / Breakout) ตั้งแต่สแกนรอบล่าสุด"""
    if current_df is None or current_df.empty or "Signal" not in current_df.columns:
        return []
    new_hits = []
    for _, row in current_df.iterrows():
        tk, sig = row.get("Ticker"), row.get("Signal")
        if sig in NOTABLE_SIGNALS and last_signals.get(tk) != sig:
            new_hits.append({"ticker": tk, "signal": sig})
    return new_hits


def signals_snapshot(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Signal" not in df.columns:
        return {}
    return dict(zip(df["Ticker"], df["Signal"]))


def maybe_notify_telegram(message: str) -> bool:
    """ส่งข้อความผ่าน Telegram ถ้ามี secrets ตั้งไว้ — ไม่มีก็ไม่ทำอะไร (no-op)"""
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        return False
    if not token or not chat_id:
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=8)
        return resp.ok
    except Exception as e:
        log_err("maybe_notify_telegram", e)
        return False


PREFETCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "latest_scan.json")
ALERTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alerts.json")


@st.cache_data(ttl=300)
def load_prefetched_bundle():
    """
    อ่านไฟล์ data/latest_scan.json ที่ GitHub Actions ดึงไว้ล่วงหน้าทุก 4 ชม.
    (ใหม่ v3.2) — เปลี่ยนจาก v3.0/3.1 ที่ต้องรอให้มีคนกด Run Screener ก่อน
    ถึงจะมีข้อมูล ตอนนี้ "การดึงข้อมูล" กับ "การดู" แยกกันคนละจุดสมบูรณ์ ไฟล์นี้
    ถูกเขียนโดย fetch_data.py (รันจาก GitHub Action) ไม่ใช่จากแอปตัวนี้เอง

    คืนค่า (generated_at: str|None, df: pd.DataFrame) — ถ้ายังไม่มีไฟล์
    (เช่น ก่อน Action รันรอบแรก) จะคืน (None, DataFrame ว่าง)
    """
    if not os.path.exists(PREFETCH_PATH):
        return None, pd.DataFrame()
    try:
        with open(PREFETCH_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("generated_at"), pd.DataFrame(payload.get("data", []))
    except Exception as e:
        log_err("load_prefetched_bundle", e)
        return None, pd.DataFrame()


@st.cache_data(ttl=300)
def load_prefetch_alerts():
    """อ่าน data/alerts.json (สัญญาณใหม่ระหว่างรอบล่าสุดกับรอบก่อนหน้า) ที่
    fetch_data.py คำนวณไว้แล้วครั้งเดียวตอนดึงข้อมูล (ใหม่ v3.2) — ไม่คำนวณซ้ำ
    ทุกครั้งที่มีคนเข้าเว็บ เพื่อไม่ให้ผลลัพธ์ขึ้นกับว่าใครเข้ามาดูก่อน-หลัง"""
    if not os.path.exists(ALERTS_PATH):
        return []
    try:
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("new_signals", [])
    except Exception as e:
        log_err("load_prefetch_alerts", e)
        return []


def get_with_bundle_fallback(tickers: list, bundle_df: pd.DataFrame, max_live_fallback: int = 15) -> pd.DataFrame:
    """ดึงข้อมูลของ tickers ที่ต้องการจาก bundle ที่ดึงไว้ล่วงหน้าก่อน ถ้ามีบาง
    ticker ไม่อยู่ใน bundle (เช่น พิมพ์ ticker แปลกๆใน Custom) ค่อย live fallback
    ทีละตัวสำหรับส่วนที่ขาดเท่านั้น (ใหม่ v3.2)"""
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


st.set_page_config(
    page_title="Stock Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

def main():
    st.markdown("""
    <div style="text-align:center;padding:8px 0 16px 0;">
        <h1 style="font-size:1.8rem;margin:0;">📊 Institutional Stock Screener <span style="font-size:0.9rem;color:#3fb950;">v3.0</span></h1>
        <p style="color:#8b949e;font-size:0.85rem;margin:4px 0 0 0;">
            Precision Math · Multi-Market · Hidden Gem Engine · Backtester
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ──────────────────────────────────
    if "df" not in st.session_state: st.session_state.df = pd.DataFrame()
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()  # โหลดจาก disk แทนเริ่มเป็น [] เสมอ
    if "ran" not in st.session_state: st.session_state.ran = False

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ ตั้งค่า")

        universe = st.selectbox("🌍 Universe | กลุ่มหุ้น", list(UNIVERSE_OPTIONS.keys()))

        sector_choice = []
        if universe == "Sector Focus | เลือกตามหมวด":
            sector_choice = st.multiselect("เลือก Sector | หมวดหุ้น", list(SECTOR_MAP.keys()),
                                           default=["Technology | เทคโนโลยี"])

        custom_input = ""
        if universe == "Custom Tickers":
            custom_input = st.text_area("Tickers (คั่นด้วย ,)", "AAPL,MSFT,NVDA,GOOGL", height=80)

        st.markdown("---")
        st.markdown("**🔬 Filters**")

        min_gem = st.slider("💎 Min Gem Score", 0, 10, 0)
        min_accum = st.slider("📦 Min Accum Score", 0, 5, 0)
        pat_filter = st.multiselect("EMA Pattern | รูปแบบเส้น EMA",
            ["🏆 Perfect Uptrend", "📈 Strong Uptrend", "✨ Golden Align",
             "🔥 Squeeze", "⚡ Pre-Squeeze", "🌱 Early Break", "🎯 EMA Fan"],
            default=[], placeholder="ทั้งหมด")

        st.markdown("---")
        with st.expander("📅 Timeframe"):
            period = st.selectbox("ช่วงเวลา | Period", ["1y", "2y", "6mo", "3mo"], index=0)
            interval = st.selectbox("Interval | ช่วงแท่งเทียน", ["1d", "1wk"], index=0)
            use_rs = st.checkbox("คำนวณ RS vs SPY", value=True,
                                  help="ช้าขึ้นเล็กน้อย แต่ได้ข้อมูลสำคัญ")

        max_tk = st.slider("Max Tickers | จำนวนหุ้นสูงสุด", 10, 300, 50, step=10)

        st.markdown("---")
        notify_tg = st.checkbox("📲 แจ้งเตือนผ่าน Telegram ถ้าตั้งค่าไว้", value=True,
                                help="เฉพาะตอนกด Run สแกนสดด้วยตัวเอง — ต้องตั้ง "
                                     "TELEGRAM_BOT_TOKEN และ TELEGRAM_CHAT_ID ใน "
                                     ".streamlit/secrets.toml ก่อน ถ้าไม่ตั้งจะไม่มีผลอะไร "
                                     "(ส่วนการแจ้งเตือนของรอบ prefetch อัตโนมัติทุก 4 ชม. "
                                     "ตั้งค่าแยกที่ GitHub Action ไม่เกี่ยวกับติ๊กนี้)")

        st.markdown("---")
        run_btn = st.button("🚀 Run Screener | สแกนสดเดี๋ยวนี้", use_container_width=True,
                            help="ปกติไม่ต้องกดเลย — ข้อมูลมาจากรอบดึงอัตโนมัติทุก 4 ชม. อยู่แล้ว "
                                 "กดปุ่มนี้เฉพาะตอนอยากได้ข้อมูลสดเดี๋ยวนี้ ไม่รอรอบถัดไป")

        with st.expander("💾 Export | ส่งออกข้อมูล"):
            if not st.session_state.df.empty:
                csv = st.session_state.df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv,
                    f"screener_{datetime.date.today()}.csv", "text/csv",
                    use_container_width=True)
            else:
                st.caption("รัน Screener ก่อน")

        with st.expander("🗑️ ล้าง Cache (เฉพาะของสแกนสด/manual)"):
            st.caption("ใช้ลบเฉพาะ cache ของการกด 'Run Screener' สแกนสดเอง "
                      "ไม่กระทบข้อมูล prefetch อัตโนมัติทุก 4 ชม. (อันนั้นอัปเดตเองจาก GitHub Action)")
            if st.button("ล้าง Cache ของ Universe นี้", use_container_width=True):
                tickers_for_clear = resolve_tickers(universe, sector_choice, custom_input)[:max_tk]
                if clear_cache_for(universe, tuple(tickers_for_clear), period, interval):
                    st.success("ล้างแล้ว — กด Run Screener เพื่อสแกนสดใหม่")
                else:
                    st.info("ยังไม่มี Cache สแกนสดสำหรับ Universe นี้")

        st.markdown("---")
        st.markdown(f"<p style='color:#7d8590;font-size:0.72rem;'>Data: Yahoo Finance<br>"
                    f"ข้อมูลหลัก: ดึงอัตโนมัติทุก 4 ชม. ผ่าน GitHub Action<br>"
                    f"Watchlist: {len(st.session_state.watchlist)} หุ้น (persist ข้าม session)</p>",
                    unsafe_allow_html=True)

    # ── Resolve tickers ──────────────────────────────────────
    tickers_all = resolve_tickers(universe, sector_choice, custom_input)
    tickers_use = tickers_all[:max_tk]

    auto_loaded = False
    bundle_gen_at = None
    new_signal_hits = []

    # ── Run screener (กดเอง = สแกนสดตอนนี้เลย ไม่รอรอบ prefetch ทุก 4 ชม.) ──
    if run_btn:
        bench_tuple = None
        if use_rs:
            with st.spinner("ดึงข้อมูล SPY เป็น benchmark…"):
                try:
                    spy_df = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
                    bench_tuple = make_bench_tuple(spy_df)
                except Exception as e:
                    log_err("fetch SPY benchmark", e)
                    st.warning("ดึงข้อมูล SPY ไม่สำเร็จ — จะสแกนต่อโดยไม่มี Relative Strength")

        prog = st.progress(0.0, text=f"⚡ กำลังสแกน 0/{len(tickers_use)} หุ้น…")

        def _on_progress(done, total):
            prog.progress(done / total if total else 1.0, text=f"⚡ กำลังสแกน {done}/{total} หุ้น…")

        df = batch_scan(tuple(tickers_use), period, interval, bench_tuple, progress_cb=_on_progress)
        prog.empty()
        st.session_state.df = df
        st.session_state.ran = True
        save_disk_cache(universe, tuple(tickers_use), period, interval, df)

        # ── แจ้งเตือนสัญญาณใหม่ (เทียบกับสแกนสดของตัวเองรอบก่อน — แยกจากของ prefetch) ──
        last_sig = load_last_signals(universe)
        new_signal_hits = detect_new_signals(df, last_sig)
        save_last_signals(universe, signals_snapshot(df))
        if new_signal_hits and notify_tg:
            msg = "🔔 สัญญาณใหม่ (" + universe + "): " + ", ".join(
                f"{h['ticker']} {h['signal']}" for h in new_signal_hits[:20])
            maybe_notify_telegram(msg)

    # ── ดีฟอลต์ (ไม่กด Run): อ่านจากข้อมูลที่ดึงไว้ล่วงหน้าทุก 4 ชม. (v3.2 ใหม่) ──
    # เปลี่ยนจาก v3.0/3.1 ที่ต้องรอให้มีคนกด Run ก่อนถึงจะมีข้อมูล — ตอนนี้แอป
    # ไม่ได้ไปคุยกับ Yahoo ตอนคนเข้าดูเลย แค่อ่านไฟล์ที่ fetch_data.py
    # (รันจาก GitHub Action ทุก 4 ชม.) เตรียมไว้ให้แล้ว
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

    # ── แสดงสถานะ ──────────────────────────────────────
    if st.session_state.ran and not df.empty:
        if auto_loaded:
            try:
                gen_dt = datetime.datetime.fromisoformat(str(bundle_gen_at).replace("Z", "+00:00"))
                gen_lbl = gen_dt.astimezone(ZoneInfo("Asia/Bangkok")).strftime("%d/%m %H:%M น.")
            except Exception:
                gen_lbl = str(bundle_gen_at) or "—"
            st.markdown(
                f'<div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;'
                f'padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                f'<span style="color:#3fb950;font-size:0.85rem;">⚡ ข้อมูลล่วงหน้า — อัปเดตอัตโนมัติทุก 4 ชม.</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;">ดึงล่าสุด {gen_lbl} · {universe} · '
                f'{len(df)} หุ้น</span>'
                f'<span style="color:#7d8590;font-size:0.75rem;">— ไม่ต้องรอ ไม่ต้องกด Run</span>'
                f'</div>', unsafe_allow_html=True)
        else:
            age_lbl = cache_age_label(universe, tuple(tickers_use), period, interval)
            st.markdown(
                f'<div style="background:#1c2128;border:1px solid #238636;border-radius:8px;'
                f'padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                f'<span style="color:#3fb950;font-size:0.85rem;">✅ สแกนสดเสร็จแล้ว (manual)</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;">{age_lbl} · {universe} · '
                f'{len(df)} หุ้น · บันทึกแล้ว</span>'
                f'</div>', unsafe_allow_html=True)

        # ── แถบแจ้งเตือนสัญญาณใหม่ ──
        if new_signal_hits:
            chips = " ".join(
                f'<span style="background:#21262d;border:1px solid #3fb950;border-radius:6px;'
                f'padding:3px 10px;font-size:0.78rem;margin-right:4px;">'
                f'<b style="color:#3fb950;">{h["ticker"]}</b> {h["signal"]}</span>'
                for h in new_signal_hits[:25]
            )
            st.markdown(
                f'<div style="background:#132a1a;border:1px solid #3fb950;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:10px;">'
                f'<div style="color:#3fb950;font-weight:700;font-size:0.85rem;margin-bottom:6px;">'
                f'🔔 สัญญาณใหม่ตั้งแต่สแกนล่าสุด ({len(new_signal_hits)} หุ้น)</div>'
                f'<div>{chips}</div></div>', unsafe_allow_html=True)
    elif st.session_state.ran and df.empty and bundle_gen_at:
        st.warning("⚠️ Universe นี้ยังไม่อยู่ในข้อมูลที่ดึงไว้ล่วงหน้า — กด 🚀 Run Screener "
                  "เพื่อดึงสดสำหรับ Universe นี้แทน")


    # ── TABS ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard | แดชบอร์ด",
        "💎 Hidden Gems | หุ้นซ่อนเร้น",
        "🔍 Deep Dive | เจาะลึกหุ้น",
        "📈 Backtester | ทดสอบย้อนหลัง",
        "🗺️ Sector Map | แผนผังกลุ่มหุ้น",
        "⭐ Watchlist | รายการเฝ้าดู",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1: DASHBOARD
    # ════════════════════════════════════════════════════════
    with tab1:
        if not st.session_state.ran:
            st.markdown("""
            <div style="text-align:center;padding:80px 0;color:#8b949e;">
                <div style="font-size:3rem;">📊</div>
                <h3 style="color:#c9d1d9;">ยังไม่มีข้อมูลล่วงหน้าสำหรับ Universe นี้</h3>
                <p>ปกติข้อมูลจะโผล่ขึ้นอัตโนมัติ (ดึงทุก 4 ชม.) — ถ้ายังไม่เห็น ลองกด
                🚀 Run Screener เพื่อดึงสดเองครั้งนี้</p>
            </div>""", unsafe_allow_html=True)
        elif df.empty:
            st.error("⚠️ ไม่พบข้อมูล — ลองกด 🚀 Run Screener เพื่อดึงสด หรือตรวจสอบ Ticker/อินเทอร์เน็ต")
        else:
            total = len(df)
            bulls = len(df[df["Trend"].str.contains("Bull", na=False)])
            gems = len(df[df["💎 Gem"].str.contains("Gem", na=False)]) if "💎 Gem" in df else 0
            breaks = len(df[df["Signal"].str.contains("Breakout|เบรคเอาท์", na=False)]) if "Signal" in df else 0
            strong = len(df[df["Signal"].str.contains("Strong Buy", na=False)]) if "Signal" in df else 0
            avg_rsi = df["RSI"].mean() if "RSI" in df else 0

            cards_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">'
            cards_html += info_card("สแกน", str(total))
            cards_html += info_card("Bull Trend", str(bulls), "#3fb950")
            cards_html += info_card("Strong Buy", str(strong), "#3fb950")
            cards_html += info_card("Breakout", str(breaks), "#f7b731")
            cards_html += info_card("Hidden Gem", str(gems), "#ffd700")
            cards_html += info_card("Avg RSI", f"{avg_rsi:.1f}", "#79c0ff")
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                sig_filter = st.multiselect("Signal | สัญญาณ", df["Signal"].unique().tolist() if "Signal" in df else [],
                                            default=[], key="d_sig", placeholder="ทั้งหมด")
            with fc2:
                trend_filter = st.multiselect("Trend | แนวโน้ม", ["🟢 Bull", "🔴 Bear"],
                                              default=[], key="d_tr", placeholder="ทั้งหมด")
            with fc3:
                sq_filter = st.multiselect("Squeeze | การหดตัว", df["Squeeze"].unique().tolist() if "Squeeze" in df else [],
                                           default=[], key="d_sq", placeholder="ทั้งหมด")

            show_cols = [c for c in ["Ticker", "Price", "ราคาปิด", "Trend", "RSI", "EMA Pattern",
                                     "Squeeze", "Signal Age", "💎 Gem", "Accum", "RS 20D", "Signal", "Stars"]
                         if c in df.columns]
            dfv = df[show_cols].copy()

            if "Signal Age" in dfv.columns:
                dfv["Signal Age"] = dfv["Signal Age"].apply(
                    lambda x: f"{int(x)}d ago" if isinstance(x, (int, float)) and x >= 0 else "—")

            mask = pd.Series(True, index=dfv.index)
            if sig_filter: mask &= df["Signal"].isin(sig_filter)
            if trend_filter: mask &= df["Trend"].apply(lambda x: any(t in str(x) for t in trend_filter))
            if sq_filter: mask &= df["Squeeze"].isin(sq_filter)
            if min_gem > 0 and "Gem Score" in df.columns: mask &= df["Gem Score"] >= min_gem
            if min_accum > 0 and "Accum Score" in df.columns: mask &= df["Accum Score"] >= min_accum
            if pat_filter and "EMA Pattern" in df.columns:
                mask &= df["EMA Pattern"].apply(lambda x: any(p in str(x) for p in pat_filter))
            dfv = dfv[mask]

            prio = {"🔥 Strong Buy": 0, "🚀 Breakout": 1, "📈 ขาขึ้น": 2,
                    "⚠️ เฝ้าระวัง": 3, "🔄 Neutral": 4, "⏳ รอ Pullback": 5, "❌ ขาลง": 6}
            if "Signal" in dfv.columns:
                dfv["_p"] = dfv["Signal"].map(prio).fillna(7)
                dfv = dfv.sort_values("_p").drop(columns=["_p"])

            smap = {"Signal": _sty_signal, "💎 Gem": _sty_gem, "RSI": _sty_rsi,
                    "Squeeze": _sty_squeeze, "RS 20D": _sty_rs, "Accum": _sty_signal,
                    "EMA Pattern": _sty_signal}
            st.markdown(f"**{len(dfv)} หุ้นที่ตรงเงื่อนไข**")
            st.dataframe(make_table(dfv, smap), use_container_width=True, height=520)

            st.markdown("---")
            wl_col1, wl_col2 = st.columns([3, 1])
            with wl_col1:
                add_tk = st.text_input("➕ เพิ่มในรายการเฝ้าดู", placeholder="AAPL", key="wl_add")
            with wl_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("เพิ่ม Watchlist | เพิ่มรายการเฝ้าดู") and add_tk.strip():
                    tk = add_tk.strip().upper()
                    if tk not in st.session_state.watchlist:
                        st.session_state.watchlist.append(tk)
                        save_watchlist(st.session_state.watchlist)  # persist ทันที (ใหม่ v3.0)
                        st.success(f"เพิ่ม {tk} แล้ว")

    # ════════════════════════════════════════════════════════
    # TAB 2: HIDDEN GEMS
    # ════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 💎 Hidden Gem Finder")
        st.caption("หุ้นที่ EMA สวย + Volume สะสมเงียบๆ + ตลาดยังไม่สนใจ")

        if df.empty:
            st.info("รัน Screener ก่อนครับ")
        else:
            g_cols = st.columns(4)
            keywords = [("💎 Hidden Gem", "Hidden", "#ffd700"),
                        ("🔭 Emerging Gem", "Emerging", "#3fb950"),
                        ("🔬 Stealth Accum", "Stealth", "#ab7df8"),
                        ("🔥 Squeeze", "Squeeze", "#ef5350")]
            for i, (lbl, kw, clr) in enumerate(keywords):
                cnt = df.apply(lambda r, kw=kw: kw in str(r.get("💎 Gem", "")) or
                               kw in str(r.get("EMA Pattern", "")) or
                               kw in str(r.get("Accum", "")), axis=1).sum()
                g_cols[i].metric(lbl, int(cnt))

            st.markdown("---")

            if "EMA Pattern" in df.columns:
                pat_vc = df["EMA Pattern"].value_counts().head(8)
                with st.expander("📊 EMA Pattern ที่พบ", expanded=True):
                    pc = st.columns(4)
                    for i, (pat, cnt) in enumerate(pat_vc.items()):
                        pc[i % 4].markdown(
                            f'<div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;'
                            f'padding:10px 14px;margin:3px 0;">'
                            f'<div style="font-size:0.85rem;font-weight:700;color:#e6edf3;">{pat}</div>'
                            f'<div style="color:#8b949e;font-size:0.75rem;">{cnt} หุ้น</div></div>',
                            unsafe_allow_html=True)

            st.markdown("---")

            gf1, gf2 = st.columns(2)
            with gf1:
                gem_f = st.multiselect("💎 Gem Label | ระดับหุ้นซ่อนเร้น",
                    ["💎 Hidden Gem", "🔭 Emerging Gem", "👀 Watch"],
                    default=[], key="gf1", placeholder="ทั้งหมด")
            with gf2:
                acc_f = st.multiselect("📦 Accumulation | การสะสมหุ้น",
                    ["🔬 Stealth Accum", "📦 Quiet Accum", "🔍 Possible Accum", "👀 Watch"],
                    default=[], key="gf2", placeholder="ทั้งหมด")

            gem_show = [c for c in ["Ticker", "Price", "ราคาปิด", "💎 Gem", "Gem Score",
                                    "EMA Pattern", "Squeeze", "Accum", "Accum Score",
                                    "RSI", "Vol×20D", "RS 20D", "Signal", "MktCap$B"] if c in df.columns]
            dfg = df[gem_show].copy()

            gm = pd.Series(True, index=dfg.index)
            if gem_f: gm &= df["💎 Gem"].isin(gem_f)
            if acc_f: gm &= df["Accum"].isin(acc_f)
            if min_gem > 0: gm &= df["Gem Score"] >= min_gem
            if min_accum > 0: gm &= df["Accum Score"] >= min_accum
            if pat_filter: gm &= df["EMA Pattern"].apply(lambda x: any(p in str(x) for p in pat_filter))
            dfg = dfg[gm]
            if "Gem Score" in dfg.columns:
                dfg = dfg.sort_values("Gem Score", ascending=False)

            gsmap = {"💎 Gem": _sty_gem, "Accum": _sty_signal, "EMA Pattern": _sty_signal,
                     "Gem Score": _sty_gs, "Signal": _sty_signal, "RSI": _sty_rsi, "Squeeze": _sty_squeeze}
            st.markdown(f"**{len(dfg)} หุ้น**")
            st.dataframe(make_table(dfg, gsmap), use_container_width=True, height=540)

            with st.expander("📖 อ่านค่า"):
                st.markdown("""
**💎 Gem Score (0–10)**
- **8–10** `💎 Hidden Gem` — EMA สวย + สะสมเงียบ + cap เล็ก
- **6–7** `🔭 Emerging Gem` — สัญญาณดี ยังไม่ครบ
- **4–5** `👀 Watch` — ควรติดตาม

**EMA Pattern**
- `🏆 Perfect Uptrend` — price > EMA5>10>20>50>100>200
- `🔥 Squeeze` — EMA 20/50/200 ชิดกัน < 2.5% → กำลังจะเบรค
- `🌱 Early Break` — เพิ่งข้าม EMA200 ขึ้นมา

**Squeeze Direction**
- `🔥 Squeezing` — bandwidth แคบลง → **ยังไม่สาย**
- `🌱 Just Broke` — เพิ่งเบรค → **รีบตัดสินใจ**
- `📈 Expanding` — กางออกแล้ว → อาจช้าไปแล้ว
                """)

    # ════════════════════════════════════════════════════════
    # TAB 3: DEEP DIVE
    # ════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔍 วิเคราะห์รายตัว")

        pick_list = df["Ticker"].tolist() if not df.empty else tickers_use[:50]
        d1, d2, d3 = st.columns([3, 1, 1])
        with d1:
            sel = st.selectbox("เลือกหุ้น | Select Ticker", pick_list, key="dd_sel")
        with d2:
            ch_h = st.selectbox("ความสูงกราฟ | Chart Height", [620, 700, 800, 500], index=0, key="dd_h")
        with d3:
            ch_iv = st.selectbox("Timeframe | ช่วงเวลากราฟ", ["D", "W", "60", "15"], index=0, key="dd_iv",
                                 format_func=lambda x: {"D": "รายวัน", "W": "สัปดาห์", "60": "1H", "15": "15M"}[x])

        if sel:
            row = None
            if not df.empty and sel in df["Ticker"].values:
                row = df[df["Ticker"] == sel].iloc[0].to_dict()

            if row:
                px_now = row.get("Price", 0)
                pc_now = row.get("ราคาปิด", 0)
                chg_pct = round((px_now - pc_now) / pc_now * 100, 2) if pc_now else 0
                chg_col = "#3fb950" if chg_pct >= 0 else "#f85149"
                chg_arr = "▲" if chg_pct >= 0 else "▼"
                sq_now = row.get("Squeeze", "—")
                age_now = row.get("Signal Age", -1)
                age_str = f"{age_now}d ago" if isinstance(age_now, (int, float)) and age_now >= 0 else "—"
                sig_now = row.get("Signal", "—")
                rs20_now = row.get("RS 20D", np.nan)

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:16px;'
                    f'padding:10px 0 6px 0;flex-wrap:wrap;">'
                    f'<span style="font-size:2rem;font-weight:800;color:#ffffff;">'
                    f'${px_now:,.2f}</span>'
                    f'<span style="color:{chg_col};font-size:1.1rem;font-weight:700;">'
                    f'{chg_arr} {chg_pct}%</span>'
                    f'<span style="color:#8b949e;font-size:0.82rem;">ปิด: '
                    f'<b style="color:#c9d1d9;">${pc_now:,.2f}</b></span>'
                    f'<span style="color:#8b949e;font-size:0.82rem;">Signal Age: '
                    f'<b style="color:#f7b731;">{age_str}</b></span>'
                    f'<span style="color:#8b949e;font-size:0.82rem;">Squeeze: '
                    f'<b style="color:#ab7df8;">{sq_now}</b></span>'
                    f'<span style="color:#8b949e;font-size:0.82rem;">RS 20D: '
                    f'<b style="color:{"#3fb950" if (rs20_now or 0) > 0 else "#f85149"};">'
                    f'{rs20_now:.1f}%</b></span>'
                    f'<span style="background:#21262d;border:1px solid #30363d;'
                    f'border-radius:6px;padding:4px 12px;font-size:0.85rem;font-weight:700;">'
                    f'{sig_now}</span>'
                    f'</div>', unsafe_allow_html=True)

                ema_info = [(5, "#a8b3c5"), (10, "#a8b3c5"), (20, "#f7b731"),
                            (50, "#26c6da"), (100, "#ab7df8"), (200, "#ef5350")]
                bdg = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px 0;">'
                for n, col in ema_info:
                    ev = row.get(f"EMA{n}", None)
                    dev = row.get(f"vs EMA{n}%", None)
                    if ev and dev is not None:
                        dc = "#3fb950" if dev > 0 else "#f85149"
                        sgn = "+" if dev > 0 else ""
                        bdg += (f'<div style="background:#1c2128;border:1px solid {col}40;'
                                f'border-radius:8px;padding:8px 12px;min-width:88px;">'
                                f'<div style="color:{col};font-size:0.68rem;font-weight:700;'
                                f'letter-spacing:0.05em;">EMA {n}</div>'
                                f'<div style="color:#ffffff;font-size:0.95rem;font-weight:700;">'
                                f'${ev:,.2f}</div>'
                                f'<div style="color:{dc};font-size:0.75rem;font-weight:600;">'
                                f'{sgn}{dev:.2f}%</div></div>')
                bdg += '</div>'
                st.markdown(bdg, unsafe_allow_html=True)

            st.caption("📈 กราฟจาก TradingView · 🟡 EMA20 · 🔵 EMA50 · 🔴 EMA200 · RSI · MACD")
            tv_chart(sel, height=ch_h, interval=ch_iv)

            st.markdown("---")

            fetch_live_btn = st.button("⚡ ดึงข้อมูลสด (Real-Time)", key="dd_live")
            if fetch_live_btn:
                with st.spinner("กำลังดึงข้อมูลสด…"):
                    rt = fetch_live(sel)
                if rt:
                    chg = rt.get("change") or 0
                    cc = "#3fb950" if chg >= 0 else "#f85149"
                    arr = "▲" if chg >= 0 else "▼"
                    cols_rt = st.columns(6)
                    cols_rt[0].metric("💰 ราคาสด", str(rt["price"]))
                    cols_rt[1].metric("📈 เปลี่ยน", f"{arr} {chg}%")
                    cols_rt[2].metric("🔼 High วันนี้", str(rt["high"]))
                    cols_rt[3].metric("🔽 Low วันนี้", str(rt["low"]))
                    cols_rt[4].metric("📊 Volume", rt["vol"])
                    cols_rt[5].metric("🏢 Mkt Cap", rt["cap"])

            if row:
                st.markdown("---")
                st.markdown("**📐 Technical Detail**")
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.markdown('<p style="color:#8b949e;font-size:0.75rem;font-weight:700;'
                                'text-transform:uppercase;letter-spacing:0.06em;">MOMENTUM</p>',
                                unsafe_allow_html=True)
                    st.metric("RSI (14)", row.get("RSI", "—"))
                    st.metric("MACD Line", row.get("MACD", "—"))
                    st.metric("MACD Histogram", row.get("MACD_H", "—"))
                    st.metric("Gem Score", row.get("Gem Score", "—"))
                with tc2:
                    st.markdown('<p style="color:#8b949e;font-size:0.75rem;font-weight:700;'
                                'text-transform:uppercase;letter-spacing:0.06em;">VOLUME</p>',
                                unsafe_allow_html=True)
                    st.metric("Vol ×20D", f'{row.get("Vol×20D", "—")}×')
                    st.metric("Vol ×3M", f'{row.get("Vol×3M", "—")}×')
                    st.metric("Accum", row.get("Accum", "—"))
                    st.metric("RS 20D", f'{row.get("RS 20D", "—")}%')
                with tc3:
                    st.markdown('<p style="color:#8b949e;font-size:0.75rem;font-weight:700;'
                                'text-transform:uppercase;letter-spacing:0.06em;">PERFORMANCE</p>',
                                unsafe_allow_html=True)
                    st.metric("YTD Return", f'{row.get("YTD%", "—")}%')
                    st.metric("52W Drawdown", f'{row.get("Drawdown%", "—")}%')
                    st.metric("P/E Ratio", row.get("P/E", "—"))
                    st.metric("Div Yield", f'{row.get("Div%", "—")}%')

    # ════════════════════════════════════════════════════════
    # TAB 4: BACKTESTER
    # ════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📈 Backtester — EMA Squeeze Strategy")
        st.caption("ทดสอบย้อนหลัง 2 ปี: ซื้อตอน EMA Bandwidth < 3% + ราคาเหนือ EMA200 "
                   "(เข้าซื้อที่ open ของแท่งถัดไปหลังสัญญาณเกิด ไม่ใช่ close ของแท่งสัญญาณเอง)")

        b1, b2, b3 = st.columns([3, 1, 1])
        with b1:
            bt_ticker = st.text_input("Ticker | ชื่อหุ้น", value="AAPL", key="bt_tk").upper()
        with b2:
            hold_d = st.selectbox("ถือกี่วัน | Hold Days", [10, 15, 20, 30], index=2, key="bt_hold")
        with b3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button("▶️ Run Backtest | เริ่มทดสอบ", key="bt_run")

        if run_bt and bt_ticker:
            with st.spinner(f"กำลัง Backtest {bt_ticker}…"):
                res = backtest(bt_ticker, hold_d)

            if "error" in res:
                st.error(f"❌ {res['error']}")
            elif res.get("n", 0) == 0:
                st.warning("ไม่พบ signal ใน 2 ปีที่ผ่านมา (ลองเปลี่ยน Ticker)")
            else:
                wc = "#3fb950" if res["win_rate"] >= 55 else "#d29922" if res["win_rate"] >= 45 else "#f85149"
                ac = "#3fb950" if res["avg"] > 0 else "#f85149"
                cards = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;">'
                cards += info_card("Trades", str(res["n"]))
                cards += info_card("Win Rate", f'{res["win_rate"]}%', wc)
                cards += info_card("Avg Return/Trade", f'{res["avg"]}%', ac)
                cards += info_card("Best", f'+{res["best"]}%', "#3fb950")
                cards += info_card("Worst", f'{res["worst"]}%', "#f85149")
                cards += '</div>'
                st.markdown(cards, unsafe_allow_html=True)

                # ── เปรียบเทียบกับ Buy & Hold + risk metrics (ใหม่ v3.0) ──
                strat_ret = res.get("strategy_compound_ret", 0)
                bh_ret = res.get("buy_hold_ret", 0)
                beat = strat_ret > bh_ret
                cmp_color = "#3fb950" if beat else "#f85149"
                cards2 = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 16px 0;">'
                cards2 += info_card("กลยุทธ์ (Compound)", f'{strat_ret:+.1f}%', cmp_color,
                                    "ผลรวมทุก trade ทบต้นต่อกัน")
                cards2 += info_card("Buy & Hold ช่วงเดียวกัน", f'{bh_ret:+.1f}%', "#79c0ff")
                cards2 += info_card("Max Drawdown", f'{res.get("max_drawdown", 0)}%', "#f85149",
                                    "จาก equity curve ของ trades")
                sharpe_v = res.get("sharpe")
                cards2 += info_card("Sharpe (ประมาณ)", f'{sharpe_v}' if sharpe_v is not None else "—", "#ab7df8")
                cards2 += '</div>'
                st.markdown(cards2, unsafe_allow_html=True)

                verdict = "✅ กลยุทธ์ทำได้ดีกว่าถือเฉยๆ ในช่วงที่ทดสอบ" if beat else \
                          "⚠️ ถือเฉยๆ (Buy & Hold) ทำผลตอบแทนได้ดีกว่ากลยุทธ์นี้ในช่วงที่ทดสอบ"
                st.info(verdict)

                with st.expander("⚠️ ข้อจำกัดของ Backtest นี้ (อ่านก่อนเชื่อตัวเลข)"):
                    st.caption(res.get("notes", ""))

                trades = res["trades"]
                df_bt = pd.DataFrame({"Return %": trades})
                bins = [-100, -40, -20, -10, -5, 0, 5, 10, 20, 40, 200]
                df_bt["bucket"] = pd.cut(df_bt["Return %"], bins=bins)
                vc = df_bt["bucket"].value_counts().sort_index()
                vc = vc[vc > 0]

                import streamlit.components.v1 as components
                bars = ""
                mx = max(vc.values) if len(vc) else 1
                for interval_b, cnt in vc.items():
                    pct = cnt / mx * 100
                    is_positive = interval_b.right > 0
                    col = "#3fb950" if is_positive else "#f85149"
                    label = f"{interval_b.left:.0f}% to {interval_b.right:.0f}%"
                    bars += (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                             f'<div style="color:#8b949e;font-size:0.75rem;width:120px;text-align:right;">{label}</div>'
                             f'<div style="background:{col};height:18px;width:{pct:.0f}%;border-radius:3px;min-width:2px;"></div>'
                             f'<div style="color:#e6edf3;font-size:0.78rem;">{cnt}</div></div>')
                chart_html = (f'<div style="background:#161b22;border:1px solid #30363d;'
                              f'border-radius:10px;padding:16px 20px;">'
                              f'<div style="color:#8b949e;font-size:0.78rem;margin-bottom:10px;">'
                              f'การกระจาย Return หลัง {hold_d} วัน</div>{bars}</div>')
                components.html(chart_html, height=max(len(vc) * 28 + 60, 200))

                with st.expander("ดู trades ทั้งหมด (พร้อมวันที่เข้า-ออก)"):
                    details = res.get("trade_details", [])
                    if details:
                        tdf = pd.DataFrame(details)
                        tdf.insert(0, "Trade #", range(1, len(tdf) + 1))
                        tdf["Result"] = tdf["ret"].apply(lambda x: "✅ Win" if x > 0 else "❌ Loss")
                        tdf = tdf.rename(columns={"ret": "Return %", "entry_date": "Entry", "exit_date": "Exit"})
                        st.dataframe(make_table(tdf), use_container_width=True)
                    else:
                        tdf = pd.DataFrame({"Trade #": range(1, len(trades) + 1), "Return %": trades})
                        tdf["Result"] = tdf["Return %"].apply(lambda x: "✅ Win" if x > 0 else "❌ Loss")
                        st.dataframe(make_table(tdf), use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 5: SECTOR MAP
    # ════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 🗺️ Sector Heatmap — Money Flow")
        st.caption("สแกน 5 หุ้นตัวแทนต่อ Sector เพื่อวัด momentum และ accumulation")

        run_sec = st.button("🔍 สแกน Sector Map | Scan Sectors", key="sec_btn")
        if run_sec:
            with st.spinner("กำลังสแกน 11 Sectors…"):
                sec_df = sector_heatmap_data()
                st.session_state["sec_df"] = sec_df

        if "sec_df" in st.session_state and not st.session_state["sec_df"].empty:
            sec_df = st.session_state["sec_df"]

            st.markdown("**📊 Gem Score ต่อ Sector (ยิ่งสูง = สัญญาณสะสมมากกว่า)**")
            mx_gem = sec_df["Avg Gem Score"].max() or 1
            for _, row in sec_df.iterrows():
                g_val = row["Avg Gem Score"]; a_val = row["Avg Accum"]
                rs_val = row["Avg RS 20D"]; bl_val = row["Bull %"]
                g_pct = g_val / mx_gem * 100 if mx_gem > 0 else 0
                g_col = "#ffd700" if g_val >= 7 else "#3fb950" if g_val >= 5 else "#26c6da" if g_val >= 3 else "#8b949e"
                rs_col = "#3fb950" if rs_val > 0 else "#f85149"
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;'
                    f'padding:10px 14px;margin:4px 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
                    f'<div style="color:#ffffff;font-weight:700;width:110px;font-size:0.88rem;">{row["Sector"]}</div>'
                    f'<div style="flex:1;min-width:100px;">'
                    f'<div style="background:{g_col};height:14px;width:{g_pct:.0f}%;border-radius:3px;min-width:3px;"></div></div>'
                    f'<div style="color:{g_col};font-weight:700;width:50px;font-size:0.85rem;">{g_val:.1f}</div>'
                    f'<div style="color:#8b949e;font-size:0.78rem;">Accum:<b style="color:#26c6da;"> {a_val:.1f}</b></div>'
                    f'<div style="color:#8b949e;font-size:0.78rem;">RS:<b style="color:{rs_col};"> {rs_val:+.1f}%</b></div>'
                    f'<div style="color:#8b949e;font-size:0.78rem;">Bull:<b style="color:#3fb950;"> {bl_val:.0f}%</b></div>'
                    f'<div style="color:#7d8590;font-size:0.7rem;">{row["Sample"]}</div>'
                    f'</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.dataframe(make_table(sec_df.drop(columns=["Sample"], errors="ignore")),
                         use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px;color:#8b949e;">
                <div style="font-size:2.5rem;">🗺️</div>
                <h3 style="color:#c9d1d9;">กด "สแกน Sector Map" เพื่อดู Money Flow</h3>
                <p>ใช้เวลาประมาณ 30-60 วินาที</p>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 6: WATCHLIST
    # ════════════════════════════════════════════════════════
    with tab6:
        st.markdown("### ⭐ Watchlist")
        st.caption("รายการหุ้นที่คุณเฝ้าดู — บันทึกถาวรบน disk ของแอป (อยู่ข้าม session/refresh ปกติ "
                   "แต่จะถูกล้างถ้า redeploy ใหม่จาก git push)")

        wc1, wc2, wc3 = st.columns([3, 1, 1])
        with wc1:
            new_tk = st.text_input("ชื่อหุ้น", placeholder="เช่น AAPL หรือ PTT.BK", key="wl_new")
        with wc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ เพิ่ม", key="wl_add2") and new_tk.strip():
                tk = new_tk.strip().upper()
                if tk not in st.session_state.watchlist:
                    st.session_state.watchlist.append(tk)
                    save_watchlist(st.session_state.watchlist)
        with wc3:
            st.markdown("<br>", unsafe_allow_html=True)
            rem_tk = st.selectbox("ลบออก | Remove", ["—"] + st.session_state.watchlist, key="wl_rem")
            if rem_tk != "—":
                if st.button("🗑️ ลบ", key="wl_del"):
                    st.session_state.watchlist.remove(rem_tk)
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()

        if not st.session_state.watchlist:
            st.info("ยังไม่มีหุ้นใน Watchlist — เพิ่มจากตารางด้านบนหรือพิมพ์ชื่อหุ้นเข้ามา")
        else:
            st.markdown(f"**{len(st.session_state.watchlist)} หุ้น** — "
                        f"{', '.join(st.session_state.watchlist)}")
            st.markdown("---")

            scan_wl = st.button("🔄 Scan Watchlist ทั้งหมด | Scan All", key="wl_scan")
            if scan_wl:
                with st.spinner("กำลังวิเคราะห์ Watchlist…"):
                    _, bundle_df_wl = load_prefetched_bundle()
                    wl_df_result = get_with_bundle_fallback(
                        st.session_state.watchlist, bundle_df_wl, max_live_fallback=50)
                    st.session_state["wl_df"] = wl_df_result

            if "wl_df" in st.session_state and not st.session_state["wl_df"].empty:
                wdf = st.session_state["wl_df"]
                wl_show = [c for c in ["Ticker", "Price", "Trend", "RSI", "EMA Pattern", "Squeeze",
                                       "Signal Age", "💎 Gem", "Accum", "RS 20D", "Signal", "YTD%", "Drawdown%"]
                           if c in wdf.columns]
                if "Signal Age" in wdf.columns:
                    wdf = wdf.copy()
                    wdf["Signal Age"] = wdf["Signal Age"].apply(
                        lambda x: f"{int(x)}d ago" if isinstance(x, (int, float)) and x >= 0 else "—")
                wsmap = {"Signal": _sty_signal, "💎 Gem": _sty_gem, "RSI": _sty_rsi,
                         "Squeeze": _sty_squeeze, "RS 20D": _sty_rs, "Accum": _sty_signal}
                st.dataframe(make_table(wdf[wl_show], wsmap),
                             use_container_width=True, height=400)

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
                        st.dataframe(make_table(bt_df, {"Win%": _sty_wr, "Avg Ret%": _sty_rs, "vs Buy&Hold%": _sty_rs}),
                                     use_container_width=True)


if __name__ == "__main__":
    main()
