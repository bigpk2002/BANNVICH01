"""
╔══════════════════════════════════════════════════════════════╗
║   INSTITUTIONAL STOCK SCREENER  —  v3.0                      ║
║   Refactored: Accuracy + Speed/Reliability + Backtest + Code ║
╚══════════════════════════════════════════════════════════════╝
สรุปการเปลี่ยนแปลงจาก v2.0 (ดูรายละเอียดเต็มใน CHANGELOG.md):
  1. ความแม่นยำ: แก้บั๊ก relative_strength เทียบ "ตำแหน่ง" ข้ามตลาดที่ปฏิทิน
     วันเทรดต่างกัน (หุ้นไทย .BK vs SPY) + guard format ของ dividendYield
  2. ความเร็ว/เสถียร: ลด network call ต่อ ticker, แยก cache fundamentals
     ออกจาก cache ราคา, เพิ่ม retry+backoff, สแกนแบบ concurrent
  3. Backtest: เข้าซื้อที่ open แท่งถัดไป (ไม่ lookahead), เทียบ Buy&Hold,
     เพิ่ม Max Drawdown และ Sharpe โดยประมาณ
  4. โครงสร้างโค้ด: แยกเป็นโมดูลใน lib/ + เพิ่ม watchlist ที่ persist ข้าม
     session จริง + แจ้งเตือนสัญญาณใหม่ (in-app + Telegram แบบออปชัน)
"""
import datetime
import os

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from lib.utils import log_err
from lib.cache_store import (
    cache_key, load_disk_cache, save_disk_cache, cache_age_label, clear_cache_for,
    load_watchlist, save_watchlist, load_last_signals, save_last_signals,
)
from lib.universes import UNIVERSE_OPTIONS, SECTOR_MAP, resolve_tickers
from lib.analyzer import analyze, batch_scan, fetch_live, make_bench_tuple
from lib.backtest import backtest
from lib.sector_view import sector_heatmap_data
from lib.alerts import detect_new_signals, signals_snapshot, maybe_notify_telegram
from lib.styles import (
    inject_css, make_table, info_card,
    _sty_signal, _sty_rsi, _sty_gem, _sty_squeeze, _sty_rs, _sty_gs, _sty_wr,
)
from lib.tv_chart import tv_chart

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
                                help="ต้องตั้ง TELEGRAM_BOT_TOKEN และ TELEGRAM_CHAT_ID "
                                     "ใน .streamlit/secrets.toml ก่อน ถ้าไม่ตั้งจะไม่มีผลอะไร")

        st.markdown("---")
        run_btn = st.button("🚀 Run Screener | สแกนใหม่", use_container_width=True,
                            help="สแกนข้อมูลใหม่ทั้งหมด (ถ้ามี cache อยู่แล้วจะใช้ cache อัตโนมัติ ไม่ต้องกดซ้ำ)")

        with st.expander("💾 Export | ส่งออกข้อมูล"):
            if not st.session_state.df.empty:
                csv = st.session_state.df.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv,
                    f"screener_{datetime.date.today()}.csv", "text/csv",
                    use_container_width=True)
            else:
                st.caption("รัน Screener ก่อน")

        with st.expander("🗑️ ล้าง Cache | Clear Cache"):
            st.caption("Cache จะหมดอายุอัตโนมัติทุกวัน 04:00 น. (หลังตลาด US ปิด) "
                      "กดล้างเองได้ถ้าอยากดึงข้อมูลสดทันที")
            if st.button("ล้าง Cache ของ Universe นี้", use_container_width=True):
                tickers_for_clear = resolve_tickers(universe, sector_choice, custom_input)[:max_tk]
                if clear_cache_for(universe, tuple(tickers_for_clear), period, interval):
                    st.success("ล้างแล้ว — กด Run Screener เพื่อสแกนใหม่")
                else:
                    st.info("ยังไม่มี Cache สำหรับ Universe นี้")

        st.markdown("---")
        st.markdown(f"<p style='color:#7d8590;font-size:0.72rem;'>Data: Yahoo Finance<br>"
                    f"Cache: persist บน disk · Refresh 04:00 น. ทุกวัน<br>"
                    f"Watchlist: {len(st.session_state.watchlist)} หุ้น (persist ข้าม session)</p>",
                    unsafe_allow_html=True)

    # ── Resolve tickers ──────────────────────────────────────
    tickers_all = resolve_tickers(universe, sector_choice, custom_input)
    tickers_use = tickers_all[:max_tk]

    # ── Auto-load จาก disk cache ──────────────────────────────
    auto_loaded = False
    if not run_btn:
        cached = load_disk_cache(universe, tuple(tickers_use), period, interval)
        if cached is not None and not cached.empty:
            st.session_state.df = cached
            st.session_state.ran = True
            auto_loaded = True

    new_signal_hits = []

    # ── Run screener ──────────────────────────────────────────
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

        # ── แจ้งเตือนสัญญาณใหม่ตั้งแต่สแกนล่าสุด (ใหม่ v3.0) ──
        last_sig = load_last_signals(universe)
        new_signal_hits = detect_new_signals(df, last_sig)
        save_last_signals(universe, signals_snapshot(df))
        if new_signal_hits and notify_tg:
            msg = "🔔 สัญญาณใหม่ (" + universe + "): " + ", ".join(
                f"{h['ticker']} {h['signal']}" for h in new_signal_hits[:20])
            maybe_notify_telegram(msg)

    df = st.session_state.df

    # ── แสดงสถานะ cache ──────────────────────────────────────
    if st.session_state.ran and not df.empty:
        age_lbl = cache_age_label(universe, tuple(tickers_use), period, interval)
        if auto_loaded:
            st.markdown(
                f'<div style="background:#1c2128;border:1px solid #30363d;border-radius:8px;'
                f'padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                f'<span style="color:#3fb950;font-size:0.85rem;">💾 โหลดจาก Cache</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;">{age_lbl} · {universe} · '
                f'{len(df)} หุ้น</span>'
                f'<span style="color:#7d8590;font-size:0.75rem;">— ไม่ต้องกด Run ซ้ำ '
                f'(refresh อัตโนมัติทุกวัน 04:00 น. หลังตลาด US ปิด)</span>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="background:#1c2128;border:1px solid #238636;border-radius:8px;'
                f'padding:8px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                f'<span style="color:#3fb950;font-size:0.85rem;">✅ สแกนใหม่เสร็จแล้ว</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;">{age_lbl} · {universe} · '
                f'{len(df)} หุ้น · บันทึกแล้ว</span>'
                f'</div>', unsafe_allow_html=True)

        # ── แถบแจ้งเตือนสัญญาณใหม่ (ใหม่ v3.0) ──
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
                <h3 style="color:#c9d1d9;">เลือก Universe แล้วกด 🚀 Run Screener</h3>
                <p>รองรับ S&P500, Nasdaq100, Russell2000, SET, ETF และ Custom</p>
            </div>""", unsafe_allow_html=True)
        elif df.empty:
            st.error("⚠️ ไม่พบข้อมูล — ตรวจสอบ Ticker หรือการเชื่อมต่ออินเทอร์เน็ต")
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
                    wl_results = []
                    for tk in st.session_state.watchlist:
                        d = analyze(tk)
                        if d: wl_results.append(d)
                    st.session_state["wl_df"] = pd.DataFrame(wl_results)

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
