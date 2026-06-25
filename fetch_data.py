"""
สคริปต์นี้ไม่ได้ถูกเรียกจากแอป Streamlit ตรงๆ — เป็นตัวที่ GitHub Actions
รันตามเวลา (ทุก 4 ชม. ดู .github/workflows/prefetch.yml) เพื่อดึงข้อมูลหุ้น
ทั้งหมดล่วงหน้า แล้วเซฟเป็นไฟล์ data/latest_scan.json ให้แอป Streamlit
อ่านตรงๆ แทนการไปยิง Yahoo Finance สดตอนมีคนเข้าดูหน้าเว็บ

ทำไมต้องแยกเป็นไฟล์นี้ ไม่ยัดเข้า app.py:
  - app.py ต้องรันใน Streamlit runtime (ใช้ st.session_state, ปุ่ม, sidebar ฯลฯ)
  - ตัวนี้แค่ "import app" มาดึงฟังก์ชันการสแกน/วิเคราะห์มาใช้ตรงๆ
    (กัน logic เพี้ยน/ซ้ำซ้อนกันระหว่าง 2 ที่) แล้วรันแบบ headless ไม่มีหน้าเว็บ

วิธีรันด้วยตัวเอง (ทดสอบ): python fetch_data.py
"""
import datetime
import json
import os
import time

import pandas as pd

import app  # ดึง analyze / batch_scan / resolve_tickers / UNIVERSE_OPTIONS / SECTOR_MAP
            # / make_bench_tuple / NOTABLE_SIGNALS มาใช้ตรงจาก app.py (ไม่ duplicate logic)
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "data", "latest_scan.json")
ALERTS_PATH = os.path.join(BASE_DIR, "data", "alerts.json")

# Universe ที่จะดึงล่วงหน้าให้ทั้งหมด (ไม่รวม Custom Tickers / Sector Focus
# เพราะ Sector Focus ใช้ ticker ที่อยู่ใน SECTOR_MAP ซึ่งรวมไว้แยกด้านล่างแล้ว)
PREFETCH_UNIVERSES = [
    "S&P 500 (503)",
    "Nasdaq 100 (101)",
    "Russell 2000 Small Cap",
    "US Broad Market (~700)",
    "หุ้นไทย SET/mai",
    "ETF Screener (70)",
]


def notify_telegram_from_env(message: str) -> bool:
    """เวอร์ชันสำหรับรันใน GitHub Action — อ่าน token จาก environment variable
    (ตั้งผ่าน GitHub repo Secrets ไม่ใช่ .streamlit/secrets.toml ของแอป)"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("ไม่ได้ตั้ง TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID ใน GitHub Secrets — ข้ามการแจ้งเตือน")
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"ส่ง Telegram ไม่สำเร็จ: {e}")
        return False


def load_old_signals() -> dict:
    """อ่าน bundle รอบก่อนหน้า (ก่อนที่ไฟล์นี้จะถูกเขียนทับด้วยรอบใหม่) เพื่อ
    เทียบว่ามีหุ้นไหนเปลี่ยนเป็นสัญญาณเด่นตั้งแต่รอบที่แล้ว"""
    if not os.path.exists(OUT_PATH):
        return {}
    try:
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("data", [])
        return {r.get("Ticker"): r.get("Signal") for r in rows if r.get("Ticker")}
    except Exception as e:
        print(f"อ่าน bundle รอบก่อนไม่ได้ (ข้ามไป ถือว่าไม่มีของเก่า): {e}")
        return {}


def main():
    print(f"[{datetime.datetime.now().isoformat()}] เริ่มดึงข้อมูลล่วงหน้า...")

    old_signals = load_old_signals()
    print(f"โหลด signal รอบก่อนหน้าได้ {len(old_signals)} ticker (ใช้เทียบหาสัญญาณใหม่)")

    print("รวบรวมรายชื่อหุ้นทั้งหมดจากทุก universe + sector...")
    all_tickers = set()
    for u in PREFETCH_UNIVERSES:
        try:
            ts = app.resolve_tickers(u, [], "")
            all_tickers.update(ts)
            print(f"  {u}: {len(ts)} ตัว")
        except Exception as e:
            print(f"  {u}: ดึงรายชื่อไม่สำเร็จ ({e}) — ข้าม universe นี้รอบนี้")
    for sector_tickers in app.SECTOR_MAP.values():
        all_tickers.update(sector_tickers)
    all_tickers = sorted(all_tickers)
    print(f"รวมหุ้น unique ทั้งหมดที่ต้องดึง: {len(all_tickers)} ตัว")

    bench_tuple = None
    try:
        spy_df = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
        bench_tuple = app.make_bench_tuple(spy_df)
        print("ดึง SPY benchmark สำเร็จ")
    except Exception as e:
        print(f"ดึง SPY benchmark ไม่สำเร็จ ({e}) — จะสแกนต่อโดยไม่มี Relative Strength")

    # v3.3: เดิมยิงทั้งหมดทีเดียวด้วย max_workers=8 — จาก log การรันจริงพบว่า
    # Yahoo เริ่ม Rate-limit (YFRateLimitError) หลังยิงต่อเนื่องสักพัก ทำให้
    # หุ้นท้ายๆของ universe ใหญ่หลุดไปจำนวนมาก ตอนนี้แบ่งเป็น chunk เล็กลง +
    # ใช้ concurrency ต่อ chunk ต่ำลง + พักระหว่าง chunk ให้ Yahoo "หายใจ"
    # ช้าลงแต่ได้ข้อมูลครบกว่าเดิมมาก — งานนี้ไม่มีคนรอ ไม่ต้องรีบ
    CHUNK_SIZE = 60
    PAUSE_BETWEEN_CHUNKS = 5  # วินาที

    all_dfs = []
    total = len(all_tickers)
    for i in range(0, total, CHUNK_SIZE):
        chunk = all_tickers[i:i + CHUNK_SIZE]
        chunk_df = app.batch_scan(tuple(chunk), "1y", "1d", bench_tuple, max_workers=4)
        all_dfs.append(chunk_df)
        done = min(i + CHUNK_SIZE, total)
        print(f"  ...สแกนแล้ว {done}/{total} (chunk นี้ได้ {len(chunk_df)}/{len(chunk)} ตัว)")
        if done < total:
            time.sleep(PAUSE_BETWEEN_CHUNKS)

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"สแกนสำเร็จ {len(df)} / {len(all_tickers)} ตัว (ที่เหลือคือดึงไม่สำเร็จ/delisted/rate-limit ชั่วคราว — รอบหน้าจะลองใหม่)")

    if df.empty:
        print("ผลลัพธ์ว่างเปล่าทั้งหมด — ไม่บันทึกทับไฟล์เดิม (กันข้อมูลเก่าหายเปล่าๆ)")
        return

    # ── หาสัญญาณใหม่ตั้งแต่รอบก่อน + แจ้งเตือน Telegram (ใหม่ v3.2) ──
    new_signals_map = dict(zip(df["Ticker"], df["Signal"]))
    new_hits = [
        {"ticker": t, "signal": s}
        for t, s in new_signals_map.items()
        if s in app.NOTABLE_SIGNALS and old_signals.get(t) != s
    ]
    print(f"พบสัญญาณใหม่ตั้งแต่รอบก่อน: {len(new_hits)} หุ้น")

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"generated_at": generated_at, "data": df.to_dict(orient="records")},
                   f, default=str, ensure_ascii=False)
    print(f"บันทึก {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)")

    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"generated_at": generated_at, "new_signals": new_hits}, f, ensure_ascii=False)
    print(f"บันทึก {ALERTS_PATH}")

    if new_hits and old_signals:
        # ส่ง Telegram เฉพาะตอนมีของรอบก่อนให้เทียบจริงๆ — รอบแรกสุด (ยังไม่มี
        # ไฟล์เก่าเลย) จะไม่ยิงแจ้งเตือนทั้ง list มาทีเดียว (จะดูเหมือน spam)
        msg = "🔔 สัญญาณใหม่ (auto-scan ทุก 4 ชม.): " + ", ".join(
            f"{h['ticker']} {h['signal']}" for h in new_hits[:25])
        notify_telegram_from_env(msg)
    elif new_hits:
        print(f"(รอบแรกสุด — มี {len(new_hits)} หุ้นที่เข้าเงื่อนไขเด่นอยู่แล้ว แต่ไม่ส่ง Telegram เพราะไม่มีรอบก่อนให้เทียบ)")

    print("เสร็จสิ้น")


if __name__ == "__main__":
    main()
