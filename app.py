import sys
import subprocess

# [강제 업데이트] google-generativeai 라이브러리를 최신으로 유지
try:
    import google.generativeai
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "google-generativeai"])
    except:
        pass

import os
import json
import time
import re
import logging
import random
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import FinanceDataReader as fdr
import requests

# 라이브러리 로드 확인
try:
    import pyupbit
    UPBIT_AVAILABLE = True
except ImportError:
    pyupbit = None
    UPBIT_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# -----------------------
# 1. 설정 및 초기화
# -----------------------
st.set_page_config(page_title="Unified Screener Rank (Risk Alert & Coin Radar)", layout="wide")

CONFIG_FILE = "screener_config.json"
FAV_FILE = "favorites.json"

# [NEW] 뉴스 필터링을 위한 핵심 키워드
IMPORTANT_KEYWORDS = [
    'earnings', 'revenue', 'profit', 'quarter', 'result', 'dividend', 'guidance',
    'surpass', 'beat', 'miss', 'report', 'sales', 'growth',
    'deal', 'contract', 'agreement', 'partnership', 'acquisition', 'merger', 'buyout',
    'collaboration', 'joint venture',
    'fda', 'approval', 'clearance', 'regulation', 'lawsuit', 'ban', 'launch', 'unveil',
    'surge', 'jump', 'plunge', 'tumble', 'soar', 'skyrocket', 'crash', 'record', 'rally',
    'ai', 'gpt', 'crypto', 'blockchain', 'ev', 'battery', 'semiconductor'
]

# (테마 키워드 딕셔너리는 나중을 위해 남겨두되, 표시는 하지 않음 - Baseline 규칙)
THEME_KEYWORDS = {
    "🤖로봇/AI": ["로봇", "로보틱스", "AI", "인공지능", "레인보우", "두산로보", "유진로봇", "휴림로봇", "셀바스AI", "마음AI", "솔트룩스", "코난테크놀로지", "엔젤로보틱스", "케이엔알시스템", "클로봇", "하이젠알앤엠"],
    "🔋2차전지": ["2차전지", "배터리", "양극재", "음극재", "리튬", "에코프로", "포스코퓨처", "LG에너지", "금양", "코스모신소재", "나노신소재", "대주전자", "신성델타테크"],
    "💾반도체": ["반도체", "HBM", "삼성전자", "SK하이닉스", "한미반도체", "주성엔지니어링", "이오테크닉스", "ISC", "하나마이크론", "SFA반도체", "네패스", "리노공업"],
    "🚗자동차/부품": ["현대차", "기아", "모비스", "만도", "화신", "에스엘", "성우하이텍", "서연이화", "현대모비스", "자율주행"],
    "💊바이오/제약": ["바이오", "제약", "삼성바이오", "셀트리온", "SK바이오", "유한양행", "한미약품", "알테오젠", "HLB", "레고켐", "에이비엘", "오스코텍"],
    "🚀방산/우주": ["방산", "우주", "한화에어로", "LIG넥스원", "한국항공우주", "현대로템", "풍산", "쎄트렉아이", "한화시스템"],
    "⚡전력/원전": ["전력", "변압기", "전선", "HD현대일렉", "효성중공업", "LS ELECTRIC", "제룡전기", "일진전기", "두산에너빌", "한전기술"],
    "🚢조선/해운": ["조선", "해운", "HD현대중공업", "삼성중공업", "한화오션", "HMM", "팬오션", "대한해운"],
    "🎮게임/엔터": ["게임", "엔터", "크래프톤", "엔씨소프트", "넷마블", "펄어비스", "하이브", "JYP", "에스엠", "YG", "스튜디오드래곤"],
    "💄화장품/미용": ["화장품", "미용", "아모레", "LG생활건강", "코스맥스", "한국콜마", "클리오", "파마리서치", "클래시스", "비올", "제이시스"],
    "🪙코인/STO": ["비트코인", "두나무", "우리기술투자", "한화투자증권", "위메이드", "다날", "갤럭시아머니"]
}

def get_stock_theme(name):
    """종목명에 기반하여 테마를 추정합니다."""
    for theme, keywords in THEME_KEYWORDS.items():
        for k in keywords:
            if k in name.replace(" ", ""):
                return theme
    return ""

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                for k in ["date_input", "check_date_input", "start_date_input", "end_date_input"]:
                    if k in config:
                        config[k] = datetime.strptime(config[k], "%Y-%m-%d").date()
                for k, v in config.items():
                    st.session_state[k] = v
        except: pass

def load_favorites():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def save_favorites(fav_list):
    try:
        with open(FAV_FILE, "w", encoding="utf-8") as f:
            json.dump(fav_list, f, indent=4, ensure_ascii=False)
    except: pass

# 세션 상태 초기화
if "config_loaded" not in st.session_state:
    st.session_state.df_results = None
    st.session_state.df_favorites = None
    if "df_accumulated" not in st.session_state: st.session_state.df_accumulated = pd.DataFrame()
    if "is_monitoring" not in st.session_state: st.session_state.is_monitoring = False
    if "ui_strict" not in st.session_state: st.session_state.ui_strict = True
    if "ui_ai_recom" not in st.session_state: st.session_state.ui_ai_recom = False
    
    if "use_bottom_filter" not in st.session_state: st.session_state.use_bottom_filter = False
    if "use_bb_trend_filter" not in st.session_state: st.session_state.use_bb_trend_filter = False
    if "use_bull_filter" not in st.session_state: st.session_state.use_bull_filter = False
    if "use_stoch_3beat" not in st.session_state: st.session_state.use_stoch_3beat = False 
    if "use_wam_filter" not in st.session_state: st.session_state.use_wam_filter = False 
    
    # [NEW] 시간봉별 RSI 필터 상태
    if "use_rsi_daily" not in st.session_state: st.session_state.use_rsi_daily = False
    if "use_rsi_60m" not in st.session_state: st.session_state.use_rsi_60m = False
    if "use_rsi_240m" not in st.session_state: st.session_state.use_rsi_240m = False
    if "rsi_threshold_val" not in st.session_state: st.session_state.rsi_threshold_val = 30 
    
    st.session_state.fav_list = load_favorites()
    
    if "saved_tf" not in st.session_state: st.session_state.saved_tf = "60분봉"
    if "chart_period_days" not in st.session_state: st.session_state.chart_period_days = 730
    if "chart_view_count" not in st.session_state: st.session_state.chart_view_count = 200
    if "use_adv_strategy" not in st.session_state: st.session_state.use_adv_strategy = False 
    
    if "show_ma5" not in st.session_state: st.session_state.show_ma5 = True
    if "show_ma20" not in st.session_state: st.session_state.show_ma20 = True
    if "show_ma60" not in st.session_state: st.session_state.show_ma60 = True
    if "show_ma120" not in st.session_state: st.session_state.show_ma120 = True
    if "show_resistance" not in st.session_state: st.session_state.show_resistance = True
    
    if "use_etf_sector_filter" not in st.session_state: st.session_state.use_etf_sector_filter = False
    if "selected_sectors" not in st.session_state: st.session_state.selected_sectors = []
    
    if "include_monthly_div" not in st.session_state: st.session_state.include_monthly_div = False
    if "gemini_api_key" not in st.session_state: st.session_state.gemini_api_key = ""

    load_config()
    st.session_state["config_loaded"] = True

# -----------------------
# 2. 유틸리티 & 포맷팅
# -----------------------
def to_krx_date_str(d: date) -> str:
    return d.strftime("%Y%m%d")

def to_scalar(x):
    try:
        if x is None: return np.nan
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if isinstance(x, pd.DataFrame): x = x.iloc[:, -1]
            if x.empty: return np.nan
            x = x.tail(1).iloc[0]
        elif isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x)
            if arr.size == 0: return np.nan
            x = arr.ravel()[-1]
        return float(x)
    except: return np.nan

def find_col_by_substr(df_or_cols, substr):
    if df_or_cols is None: return None
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else df_or_cols
    substr = str(substr).lower()
    for c in cols:
        try:
            name = c if isinstance(c, str) else "_".join(map(str, c))
            if substr in name.lower(): return c
        except: continue
    return None

def yf_date_range(end_date: date, days: int):
    end_dt = datetime.combine(end_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%d"), (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

def normalize_columns(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.droplevel(1)
        except: pass
    new_cols = {}
    for c in df.columns:
        c_str = str(c).lower()
        if "open" in c_str: new_cols[c] = "Open"
        elif "high" in c_str: new_cols[c] = "High"
        elif "low" in c_str: new_cols[c] = "Low"
        elif "close" in c_str: new_cols[c] = "Close"
        elif "volume" in c_str: new_cols[c] = "Volume"
    if new_cols: df = df.rename(columns=new_cols)
    return df

def safe_fmt(x):
    try:
        if pd.isna(x) or x is None or x == "": return "-"
        return "{:.1f}".format(float(x))
    except: return "-"

def safe_fmt_price(x):
    try:
        if pd.isna(x) or x is None or x == "": return "-"
        return "{:,.0f}".format(float(x))
    except: return "-"

def safe_fmt_profit(x):
    try:
        if pd.isna(x) or x is None or x == "": return "-"
        val = float(x)
        color = "red" if val > 0 else "blue"
        return f":{color}[{val:+.2f}%]"
    except: return "-"

def to_yf_ticker(code, market):
    if market in ["US_STOCK", "US", "US_ETF"]: 
        return str(code).upper()
    code = str(code).zfill(6)
    suffix = ".KQ" if market == "KOSDAQ" else ".KS"
    return code + suffix

# -----------------------
# 3. 데이터 Fetcher
# -----------------------
@st.cache_data(ttl=3600)
def fetch_krx_tickers_fdr():
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        df_kospi = df_kospi[['Code', 'Name', 'Marcap']].copy()
        df_kosdaq = df_kosdaq[['Code', 'Name', 'Marcap']].copy()
        return df_kospi, df_kosdaq
    except: return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=86400)
def get_sp500_tickers_fdr(limit=1000):
    tickers = []
    try:
        df_sp = fdr.StockListing('S&P500')
        for _, row in df_sp.iterrows():
            tickers.append({"code": row['Symbol'], "name": row['Name'], "market": "US_STOCK"})
    except: pass
    try:
        df_ndq = fdr.StockListing('NASDAQ')
        for _, row in df_ndq.iterrows():
            tickers.append({"code": row['Symbol'], "name": row['Name'], "market": "US_STOCK"})
    except: pass
    unique_tickers = {t['code']: t for t in tickers}.values()
    final_list = list(unique_tickers)
    if len(final_list) < 50:
        fallback_list = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "QQQ", "SPY"]
        return [{"code": t, "name": t, "market": "US_STOCK"} for t in fallback_list][:limit]
    return final_list[:limit]

@st.cache_data(ttl=86400)
def get_us_leveraged_etfs():
    major_lev_etfs = [
        {"code": "BITU", "name": "ProShares Ultra Bitcoin (2x Bull)", "market": "US_ETF"},
        {"code": "SBIT", "name": "ProShares UltraShort Bitcoin (2x Bear)", "market": "US_ETF"}, 
        {"code": "BITI", "name": "ProShares Short Bitcoin (1x Bear)", "market": "US_ETF"},
        {"code": "BITX", "name": "2x Bitcoin Strategy ETF", "market": "US_ETF"},
        {"code": "CONL", "name": "GraniteShares 2x Long COIN", "market": "US_ETF"},
        {"code": "COND", "name": "GraniteShares 2x Short COIN", "market": "US_ETF"},
        {"code": "TSLL", "name": "Direxion Daily TSLA Bull 2x", "market": "US_ETF"},
        {"code": "TSLQ", "name": "AXS TSLA Bear Daily ETF (1x Bear)", "market": "US_ETF"},
        {"code": "TSLZ", "name": "Direxion Daily TSLA Bear 2x", "market": "US_ETF"},
        {"code": "NVDL", "name": "GraniteShares 2x Long NVDA", "market": "US_ETF"},
        {"code": "NVDD", "name": "GraniteShares 2x Short NVDA", "market": "US_ETF"},
        {"code": "AMZU", "name": "Direxion Daily AMZN Bull 2x", "market": "US_ETF"},
        {"code": "TQQQ", "name": "ProShares UltraPro QQQ (3x Bull)", "market": "US_ETF"},
        {"code": "SQQQ", "name": "ProShares UltraPro Short QQQ (3x Bear)", "market": "US_ETF"},
        {"code": "UPRO", "name": "ProShares UltraPro S&P500 (3x Bull)", "market": "US_ETF"},
        {"code": "SPXU", "name": "ProShares UltraPro Short S&P500 (3x Bear)", "market": "US_ETF"},
        {"code": "QLD", "name": "ProShares Ultra QQQ (2x Bull)", "market": "US_ETF"},
        {"code": "QID", "name": "ProShares UltraShort QQQ (2x Bear)", "market": "US_ETF"},
        {"code": "TNA", "name": "Direxion Small Cap Bull 3x", "market": "US_ETF"},
        {"code": "TZA", "name": "Direxion Small Cap Bear 3x", "market": "US_ETF"},
        {"code": "SOXL", "name": "Direxion Daily Semiconductor Bull 3x", "market": "US_ETF"},
        {"code": "SOXS", "name": "Direxion Daily Semiconductor Bear 3x", "market": "US_ETF"},
        {"code": "YINN", "name": "Direxion Daily FTSE China Bull 3x", "market": "US_ETF"},
        {"code": "YANG", "name": "Direxion Daily FTSE China Bear 3x", "market": "US_ETF"},
        {"code": "LABU", "name": "Direxion Biotech Bull 3x", "market": "US_ETF"},
        {"code": "LABD", "name": "Direxion Biotech Bear 3x", "market": "US_ETF"},
        {"code": "BOIL", "name": "ProShares Ultra Bloomberg Natural Gas (2x)", "market": "US_ETF"},
        {"code": "KOLD", "name": "ProShares UltraShort Bloomberg Natural Gas (2x)", "market": "US_ETF"},
        {"code": "FNGU", "name": "MicroSectors FANG+ Index 3x", "market": "US_ETF"},
        {"code": "FNGD", "name": "MicroSectors FANG+ Index -3x", "market": "US_ETF"},
        {"code": "BULZ", "name": "MicroSectors Solactive FANG & Innovation 3x", "market": "US_ETF"},
        {"code": "BERZ", "name": "MicroSectors Solactive FANG & Innovation -3x", "market": "US_ETF"},
    ]
    return major_lev_etfs

@st.cache_data(ttl=86400)
def fetch_filtered_etf_targets_fdr():
    valid_etfs = []
    major_brands = ["KODEX", "TIGER", "KBSTAR", "RISE", "ACE", "SOL", "HANARO", "ARIRANG", "KOSEF", "PLUS", "KOACT"]
    exclude_keywords = ["채권", "국채", "국고채", "회사채", "KOFR", "CD금리", "머니마켓", "단기", "액티브"]
    try:
        df = fdr.StockListing('ETF/KR')
        for _, row in df.iterrows():
            name = row['Name']
            code = row['Symbol']
            if any(brand in name.upper() for brand in major_brands):
                if not any(ex in name for ex in exclude_keywords):
                    valid_etfs.append({"code": code, "market": "ETF", "name": name})
    except: pass
    return valid_etfs

def get_etf_sector_keywords():
    return {
        "반도체/IT": ["반도체", "IT", "HBM", "AI", "인공지능", "로봇", "소부장", "시스템"],
        "2차전지/전기차": ["2차전지", "배터리", "전기차", "리튬", "양극재", "에코프로"],
        "바이오/헬스": ["바이오", "헬스케어", "제약", "의료", "생명"],
        "금융/은행": ["은행", "금융", "증권", "보험", "지주"],
        "자동차/운송": ["자동차", "운송", "기아", "현대차", "모빌리티"],
        "화학/철강/건설": ["화학", "철강", "건설", "중공업", "기계", "조선"],
        "소비재/미디어": ["소비", "유통", "화장품", "음식료", "미디어", "게임", "컨텐츠", "웹툰", "엔터"],
        "원자재/에너지": ["원유", "가스", "에너지", "구리", "골드", "달러", "원자재"],
        "인버스/레버리지": ["인버스", "레버리지", "2X", "선물"]
    }

@st.cache_data(ttl=300)
def fetch_upbit_data(ticker, interval, count):
    if not UPBIT_AVAILABLE: return pd.DataFrame()
    max_retries = 3
    for i in range(max_retries):
        try:
            time.sleep(random.uniform(0.15, 0.4)) 
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            if df is not None and not df.empty:
                return normalize_columns(df)
        except Exception: 
            time.sleep(1.0)
            pass
    return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_all_upbit_krw_tickers():
    if not UPBIT_AVAILABLE: return []
    try: 
        tickers = pyupbit.get_tickers(fiat="KRW")
        return tickers if tickers else []
    except: return []

def build_coin_list(selected_coins):
    if not selected_coins: return []
    return [{"code": c, "market": "UPBIT", "name": c} for c in selected_coins]

@st.cache_data(ttl=86400)
def get_global_ticker_map(trade_date_str):
    all_items = []
    try:
        kp, kd = fetch_krx_tickers_fdr()
        for _, row in kp.iterrows(): all_items.append(f"[KOSPI] {row['Name']} ({row['Code']})")
        for _, row in kd.iterrows(): all_items.append(f"[KOSDAQ] {row['Name']} ({row['Code']})")
    except: pass
    try:
        etfs = fetch_filtered_etf_targets_fdr()
        for e in etfs: all_items.append(f"[ETF] {e['name']} ({e['code']})")
    except: pass
    try:
        us = get_sp500_tickers_fdr(limit=800)
        for u in us: all_items.append(f"[US] {u['name']} ({u['code']})")
    except: pass
    try:
        us_lev = get_us_leveraged_etfs()
        for u in us_lev: all_items.append(f"[US_ETF] {u['name']} ({u['code']})")
    except: pass
    if UPBIT_AVAILABLE:
        try:
            coins = fetch_all_upbit_krw_tickers()
            for c in coins:
                all_items.append(f"[COIN] {c} ({c})")
        except: pass
    return sorted(all_items)

# -----------------------
# 4. 분석 로직들
# -----------------------
def compute_stoch_safe(df, k, d, smooth_k):
    try:
        st_df = ta.stoch(df["High"], df["Low"], df["Close"], k=k, d=d, smooth_k=smooth_k)
        if st_df is None: return None, None
        k_col = find_col_by_substr(st_df, "k"); d_col = find_col_by_substr(st_df, "d")
        return to_scalar(st_df[k_col].iloc[-1]), to_scalar(st_df[d_col].iloc[-1])
    except: return None, None

def check_stoch_logic_score(df, stoch_params_list):
    passed_count = 0
    first_k_val = 50 
    try:
        p1 = stoch_params_list[0]
        k1, d1 = compute_stoch_safe(df, p1["k"], p1["d"], p1["s"])
        if k1 is not None: first_k_val = k1
    except: pass
    for p in stoch_params_list:
        k, d = compute_stoch_safe(df, p["k"], p["d"], p["s"])
        if k is not None and d is not None and k > d: passed_count += 1
    return passed_count, first_k_val

def calculate_indicator_score(df, toggles, weights, sma_config):
    out = {"score": 0.0, "reasons": [], "metrics": {}}
    if df is None or len(df) < 20: return out
    close = df["Close"]
    mas = {}
    needed = {5,20,60,120,240}
    for w in needed:
        try: mas[w] = to_scalar(ta.sma(close, length=w).iloc[-1])
        except: mas[w] = np.nan
    
    if sma_config.get("5_20") and mas[5] > mas[20]: out["score"] += weights["sma_5_20"]
    if sma_config.get("20_60") and mas[20] > mas[60]: out["score"] += weights["sma_20_60"]
    if sma_config.get("60_120") and mas[60] > mas[120]: out["score"] += weights["sma_60_120"]
    if sma_config.get("120_240") and mas[120] > mas[240]: out["score"] += weights["sma_120_240"]

    if toggles.get("macd"):
        try:
            m = ta.macd(close)
            m_v = to_scalar(m[find_col_by_substr(m,"macd")].iloc[-1])
            m_s = to_scalar(m[find_col_by_substr(m,"sig")].iloc[-1])
            if m_v > m_s: out["score"] += weights["macd"]
        except: pass
    if toggles.get("rsi"):
        try:
            r = to_scalar(ta.rsi(close, 14).iloc[-1])
            out["metrics"]["RSI"] = r 
            if r < 45: out["score"] += weights["rsi"]
        except: pass
    if toggles.get("bb"):
        try:
            b = ta.bbands(close, 20)
            bl = to_scalar(b[find_col_by_substr(b,"bbl")].iloc[-1])
            if close.iloc[-1] <= bl * 1.02: out["score"] += weights["bb"]
        except: pass
    if toggles.get("adx"):
        try:
            a = ta.adx(df["High"], df["Low"], close)
            av = to_scalar(a[find_col_by_substr(a,"ADX")].iloc[-1])
            out["metrics"]["ADX"] = av 
            if av >= 20: out["score"] += weights["adx"]
        except: pass
    if toggles.get("mfi"):
        try:
            m = ta.mfi(df["High"], df["Low"], close, df["Volume"])
            mv = to_scalar(m.iloc[-1])
            out["metrics"]["MFI"] = mv 
            if mv < 40: out["score"] += weights["mfi"]
        except: pass
    if toggles.get("supertrend"):
        try:
            st_val = ta.supertrend(df["High"], df["Low"], close)
            dc = find_col_by_substr(st_val, "d_")
            if dc and st_val[dc].iloc[-1] > 0: out["score"] += weights["supertrend"]
        except: pass
    return out

def analyze_trends_by_indicators(df):
    try:
        if df is None or len(df) < 60: return "-", "-", "-", "-", "-", "-", "-", "-"
        stoch_val = ta.stoch(df["High"], df["Low"], df["Close"], k=5, d=3, smooth_k=3)
        if stoch_val is not None:
            k = stoch_val[find_col_by_substr(stoch_val, "k")].iloc[-1]
            d = stoch_val[find_col_by_substr(stoch_val, "d")].iloc[-1]
            short = "📈상승" if k > d else "📉하락"
        else: short = "-"
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is not None:
            m = macd[find_col_by_substr(macd, "macd")].iloc[-1]
            s = macd[find_col_by_substr(macd, "macds")].iloc[-1]
            if pd.isna(s): s = macd[find_col_by_substr(macd, "sig")].iloc[-1]
            mid = "📈상승" if m > s else "📉하락"
        else: mid = "-"
        rsi = ta.rsi(df["Close"], length=14)
        long = "📈강세" if rsi is not None and rsi.iloc[-1] > 50 else "📉약세"
        try:
            obv = ta.obv(df["Close"], df["Volume"])
            obv_sig = "💰매집" if obv is not None and ta.sma(obv, length=20) is not None and obv.iloc[-1] > ta.sma(obv, length=20).iloc[-1] else "💸매도"
        except: obv_sig = "-"
        try:
            cci = ta.cci(df["High"], df["Low"], df["Close"], length=20)
            cci_sig = "📈상승" if cci is not None and cci.iloc[-1] > 0 else "📉하락"
        except: cci_sig = "-"
        try:
            willr = ta.willr(df["High"], df["Low"], df["Close"], length=14)
            if willr is not None:
                w_val = willr.iloc[-1]
                will_sig = f"{w_val:.0f}"
            else: will_sig = "-"
        except: will_sig = "-"
        ichi_sig = "-"
        try:
            ichi = ta.ichimoku(df["High"], df["Low"], df["Close"])[0] 
            span_a = ichi[find_col_by_substr(ichi, "ISA_")]
            span_b = ichi[find_col_by_substr(ichi, "ISB_")]
            last_c = df["Close"].iloc[-1]
            if last_c > span_a.iloc[-1] and last_c > span_b.iloc[-1]: ichi_sig = "☁️구름위(안전)"
            elif last_c < span_a.iloc[-1] and last_c < span_b.iloc[-1]: ichi_sig = "⛈️구름아래(위험)"
            else: ichi_sig = "🌫️구름안(혼조)"
        except: pass
        psar_sig = "-"
        try:
            psar = ta.psar(df["High"], df["Low"], df["Close"])
            p_cols = find_col_by_substr(psar, "PSAR") 
            if p_cols is None: 
                pl = psar[find_col_by_substr(psar, "PSARl")].iloc[-1]
                ps = psar[find_col_by_substr(psar, "PSARs")].iloc[-1]
                cur_p = pl if not pd.isna(pl) else ps
            else:
                cur_p = psar[p_cols].iloc[-1]
            if not pd.isna(cur_p):
                psar_sig = "🟢상승" if df["Close"].iloc[-1] > cur_p else "🔴하락"
        except: pass
        return short, mid, long, obv_sig, cci_sig, will_sig, ichi_sig, psar_sig
    except: return "-", "-", "-", "-", "-", "-", "-", "-"

def get_ai_signal(ichi, psar, obv, willr, rsi):
    if "구름위" in str(ichi) and "상승" in str(psar) and "매집" in str(obv):
        return "🔥🔥강력추세"
    try: w = float(willr)
    except: w = -50
    if w <= -80 and "매집" in str(obv):
        return "💎바닥줍줍"
    try: r = float(rsi)
    except: r = 50
    if r >= 75:
        return "🚨과열주의"
    if "상승" in str(psar) and "구름아래" not in str(ichi):
        return "🟢상승세"
    if "구름아래" in str(ichi) and "하락" in str(psar):
        return "☠️하락위험"
    return "🟡관망"

# -----------------------
# 5. 메인 프로세스 함수
# -----------------------
def process_single_ticker_range(tk_info, yf_start, yf_end, target_tfs, stoch_params, toggles, weights, min_vol, sma_config, is_favorite=False, strategy="default", apply_strict=True, start_date=None, end_date=None):
    market = tk_info["market"]
    ticker_code = tk_info["code"]
    is_coin = (market in ["UPBIT", "COIN"])
    
    if is_coin: yf_ticker = ticker_code
    else: yf_ticker = to_yf_ticker(ticker_code, market)
    
    # 시간봉 매핑 (UI 이름 -> YFinance Interval)
    tf_map = {
        "일봉 (Daily)": "1d",
        "주봉 (Weekly)": "1wk",
        "60분봉 (1H)": "60m",
        "240분봉 (4H)": "60m" # 240분은 60분 데이터로 대략적 분석
    }
    
    results_list = []
    
    # [수정] 선택된 각 시간봉별로 데이터를 다운로드하고 분석 수행
    for tf_name in target_tfs:
        interval = tf_map.get(tf_name, "1d")
        
        try:
            if is_coin:
                # 업비트는 interval 포맷이 다름
                upbit_interval = "day"
                if "분" in tf_name: upbit_interval = "minute60" if "60" in tf_name else "minute240"
                elif "주" in tf_name: upbit_interval = "week"
                df_d_full = fetch_upbit_data(ticker_code, upbit_interval, 365)
            else:
                df_d_full = yf.download(yf_ticker, start=yf_start, end=yf_end, interval=interval, progress=False, threads=False)
                if df_d_full is not None and not df_d_full.empty:
                    if df_d_full.index.tz is not None:
                        df_d_full.index = df_d_full.index.tz_localize(None)
                df_d_full = normalize_columns(df_d_full)
            
            if df_d_full is None or df_d_full.empty: continue

            if is_favorite:
                target_dates = [df_d_full.index[-1]] if len(df_d_full) > 0 else []
            else:
                if start_date and end_date:
                    target_dates = pd.date_range(start=start_date, end=end_date)
                    target_dates = [d for d in target_dates if d in df_d_full.index]
                    if not target_dates and len(df_d_full) > 0:
                        last_dt = df_d_full.index[-1]
                        if start_date <= last_dt.date() <= end_date:
                            target_dates = [last_dt]
                else:
                    target_dates = [df_d_full.index[-1]] if len(df_d_full) > 0 else []

            for cur_date in target_dates:
                df_d = df_d_full[df_d_full.index <= cur_date]
                if len(df_d) < 15: continue 

                short_t, mid_t, long_t, obv_t, cci_t, will_t, ichi_t, psar_t = analyze_trends_by_indicators(df_d)
                last_close = to_scalar(df_d["Close"].iloc[-1])
                profit_rate = np.nan
                try:
                    curr_close_real = to_scalar(df_d_full["Close"].iloc[-1])
                    if last_close > 0 and curr_close_real > 0:
                        profit_rate = ((curr_close_real - last_close) / last_close) * 100
                except: pass

                if strategy == "bottom" and not is_favorite:
                    rsi_val = 50
                    try: rsi_val = ta.rsi(df_d["Close"], 14).iloc[-1]
                    except: pass
                    if rsi_val > 60: continue
                    if "매도" in obv_t: continue

                total_score = 0.0
                stoch_match_count = 0
                tf_match_details = []
                final_metrics = {}
                if not is_favorite:
                    vol = to_scalar(df_d["Volume"].tail(15).mean()) if "Volume" in df_d.columns else 0
                    if min_vol > 0 and vol < min_vol: continue

                cnt_d, k_val_s1 = check_stoch_logic_score(df_d, stoch_params)
                stoch_match_count += cnt_d
                total_score += (cnt_d * weights["stoch"])
                
                i_res_d = calculate_indicator_score(df_d, toggles, weights, sma_config)
                total_score += i_res_d["score"]
                final_metrics = i_res_d["metrics"]
                
                if cnt_d > 0: tf_match_details.append(f"Stoch({cnt_d})")
                if i_res_d["reasons"]: tf_match_details.append(",".join(i_res_d["reasons"]))

                ai_signal = get_ai_signal(ichi_t, psar_t, obv_t, will_t, final_metrics.get("RSI", 50))

                strict_passed = True
                if not is_favorite and strategy != "bottom" and apply_strict:
                    try:
                        if sma_config.get("5_20"):
                            c5 = ta.sma(df_d["Close"], length=5).iloc[-1]
                            c20 = ta.sma(df_d["Close"], length=20).iloc[-1]
                            if not (c5 > c20): strict_passed = False
                        if sma_config.get("20_60") and strict_passed:
                            c20 = ta.sma(df_d["Close"], length=20).iloc[-1]
                            c60 = ta.sma(df_d["Close"], length=60).iloc[-1]
                            if not (c20 > c60): strict_passed = False
                        if sma_config.get("60_120") and strict_passed:
                            c60 = ta.sma(df_d["Close"], length=60).iloc[-1]
                            c120 = ta.sma(df_d["Close"], length=120).iloc[-1]
                            if not (c60 > c120): strict_passed = False
                    except: strict_passed = False
                
                if apply_strict and not strict_passed and not is_favorite:
                    continue

                bb_pos = "N/A"
                try:
                    b = ta.bbands(df_d["Close"], length=20, std=2)
                    l_col = find_col_by_substr(b, "BBL"); m_col = find_col_by_substr(b, "BBM"); u_col = find_col_by_substr(b, "BBU")
                    lower = b[l_col].iloc[-1]; mid = b[m_col].iloc[-1]; upper = b[u_col].iloc[-1]
                    cur = df_d["Close"].iloc[-1]
                    if cur < lower: bb_pos = "0_과매도(Band하단)"
                    elif lower <= cur < mid: bb_pos = "1_약세(하단~중단)"
                    elif mid <= cur < upper: bb_pos = "2_강세(중단~상단)"
                    else: bb_pos = "3_초강세(Band상단돌파)"
                except: pass

                super_dir = "N/A"
                try:
                    st_data = ta.supertrend(df_d["High"], df_d["Low"], df_d["Close"], length=10, multiplier=3.0)
                    d_col = find_col_by_substr(st_data, "d_")
                    if d_col is not None:
                        dir_val = st_data[d_col].iloc[-1]
                        super_dir = "📈상승" if dir_val > 0 else "📉하락"
                except: pass

                # 사용자 요청 반영: 테마 태깅 제거하고 원래 종목명만 사용
                display_name = tk_info["name"]

                res_dict = {
                    "market": market, "code": ticker_code, "name": display_name, "ticker": yf_ticker,
                    "score": total_score, 
                    "stoch_hits": stoch_match_count, 
                    "Close": last_close, 
                    "수익률": profit_rate,
                    "포착일": cur_date.strftime("%Y-%m-%d"),
                    "AI신호": ai_signal,
                    "ADX": final_metrics.get("ADX"), "MFI": final_metrics.get("MFI"), "RSI": final_metrics.get("RSI"),
                    "Stoch_K": k_val_s1, 
                    "단기": short_t, "중기": mid_t, "장기": long_t, 
                    "수급(OBV)": obv_t, "사이클(CCI)": cci_t, "WillR": will_t,
                    "일목(구름)": ichi_t, "파라볼릭": psar_t,
                    "Reasons": " | ".join(tf_match_details),
                    "StrictPass": strict_passed,
                    "BB_Position": bb_pos,
                    "Supertrend_Status": super_dir,
                    "Timeframe": tf_name
                }
                results_list.append(res_dict)
        except: continue

    return results_list

@lru_cache(maxsize=512)
def fetch_financial_growth(ticker):
    try:
        t = yf.Ticker(ticker)
        fin = t.financials
        if fin is None or fin.empty: return np.nan, np.nan
        ni = None; oi = None
        if 'Net Income' in fin.index: ni = fin.loc['Net Income']
        if 'Operating Income' in fin.index: oi = fin.loc['Operating Income']
        elif 'EBIT' in fin.index: oi = fin.loc['EBIT']
        ni_g = np.nan; oi_g = np.nan
        if ni is not None and len(ni) >= 2:
            if ni.iloc[1] != 0: ni_g = ((ni.iloc[0]-ni.iloc[1])/abs(ni.iloc[1]))*100
        if oi is not None and len(oi) >= 2:
            if oi.iloc[1] != 0: oi_g = ((oi.iloc[0]-oi.iloc[1])/abs(oi.iloc[1]))*100
        return ni_g, oi_g
    except: return np.nan, np.nan

# -----------------------
# [AI 뉴스 및 차트 리포트 생성 함수]
# -----------------------
def judge_news_relevance(title):
    score = 0
    title_lower = title.lower()
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in title_lower: score += 1
    return score

def display_detailed_ai_report_v3(ticker, market, name, row_data=None):
    st.markdown(f"### 📰 **{name} ({ticker}) AI 심층 분석**")
    
    is_coin = (market in ["UPBIT", "COIN"])
    map_ub = "day"
    period_days = 300
    
    # 1. 차트 영역
    with st.spinner(f"📈 {name} 차트 로딩 중..."):
        try:
            df = pd.DataFrame()
            if is_coin:
                df = pyupbit.get_ohlcv(ticker, interval=map_ub, count=period_days)
            else:
                df = yf.download(ticker, period=f"{period_days}d", interval="1d", progress=False)
            
            df = normalize_columns(df)
            
            if not df.empty:
                st.line_chart(df["Close"], use_container_width=True)
                last_close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2]
                diff = last_close - prev_close
                pct = (diff / prev_close) * 100
                color = "red" if diff > 0 else "blue"
                st.markdown(f"**현재가:** {last_close:,.0f}원 ( :{color}[{diff:+,.0f} / {pct:+.2f}%] )")
            else:
                st.warning("차트 데이터가 없습니다.")
        except: 
            st.error("차트 데이터를 불러오는 데 실패했습니다.")

    st.markdown("---")

    # 2. [NEW] Gemini AI 심층 분석 (스마트 모델 감지)
    st.subheader("🤖 Gemini AI 시황/수급/매매전략 분석")

    news_context = ""
    try:
        yf_tick = yf.Ticker(ticker)
        news_list = yf_tick.news
        if news_list:
            scored_news = []
            for n in news_list:
                title = n.get('title', '제목 없음')
                pub_time = n.get('providerPublishTime', 0)
                date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                score = judge_news_relevance(title)
                scored_news.append({'title': title, 'date': date_str, 'score': score})
            scored_news.sort(key=lambda x: (x['score'], x['date']), reverse=True)
            top_news = [f"- {item['date']}: {item['title']}" for item in scored_news[:5]]
            news_context = "\n".join(top_news)
        else:
            news_context = "(최근 뉴스 데이터 없음)"
    except:
        news_context = "(뉴스 데이터 불러오기 실패)"

    if row_data is not None:
        signal = row_data.get("AI신호", "-")
        score = row_data.get("score", 0)
        short_t = row_data.get("단기", "-")
        mid_t = row_data.get("중기", "-")
        rsi = row_data.get("RSI", 0)
        obv = row_data.get("수급(OBV)", "-")
    else:
        signal = "분석중"; score = 0; short_t = "-"; mid_t = "-"; rsi = 0; obv = "-"

    prompt_text = f"""
나는 주식 트레이더야. 다음 종목에 대해 심층 분석해줘.

1. **분석 대상**: {name} (코드: {ticker})
2. **현재 기술적 상태 (자체 알고리즘 분석 결과)**:
   - AI 추천 신호: {signal}
   - 종합 기술 점수: {score}점
   - 추세: 단기({short_t}), 중기({mid_t})
   - RSI: {rsi} (과매수/과매도 판단용)
   - 수급(OBV): {obv}
3. **최근 주요 뉴스 헤드라인 (참고용)**:
{news_context}

**[요청 사항]**
1. **시황 & 이슈**: 위 뉴스들을 참고하여 이 종목의 최근 상승/하락 원인이 되는 핵심 재료를 요약해줘.
2. **수급 분석**: 위 기술적 지표(OBV 등)와 뉴스를 바탕으로 수급 흐름(매집 여부 등)을 분석해줘.
3. **매매 전략**: 기술적 지표와 이슈를 종합했을 때, **적정 매수 진입가**와 **1차/2차 목표 매도가**를 구체적인 가격으로 추천해줘.
"""

    api_key = st.session_state.get("gemini_api_key", "")
    
    if GENAI_AVAILABLE and api_key:
        if st.button("✨ Gemini로 지금 바로 분석하기 (Click)"):
            with st.spinner("🤖 Gemini가 사용 가능한 최신 모델을 찾는 중입니다..."):
                try:
                    genai.configure(api_key=api_key)
                    
                    target_model = None
                    try:
                        for m in genai.list_models():
                            if 'generateContent' in m.supported_generation_methods:
                                if 'gemini-1.5-pro' in m.name:
                                    target_model = m.name
                                    break
                                elif 'gemini-1.5-flash' in m.name:
                                    target_model = m.name
                                elif 'gemini-pro' in m.name and target_model is None:
                                    target_model = m.name
                    except:
                        pass

                    fallback_models = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
                    if target_model:
                        if target_model not in fallback_models:
                            fallback_models.insert(0, target_model)
                    
                    response = None
                    success_model = ""
                    last_error = ""

                    for model_name in fallback_models:
                        try:
                            safe_name = model_name if "models/" in model_name else f"models/{model_name}"
                            model = genai.GenerativeModel(safe_name)
                            response = model.generate_content(prompt_text)
                            success_model = safe_name
                            break
                        except Exception as e:
                            last_error = e
                            continue
                    
                    if response:
                        st.success(f"✅ 분석 완료! (Used Model: {success_model})")
                        st.markdown(response.text)
                    else:
                        st.error("❌ 모든 Gemini 모델 호출에 실패했습니다.")
                        st.error(f"마지막 에러 메시지: {last_error}")
                        st.caption("팁: API 키가 올바른지, 혹은 구글 클라우드 결제 계정이 필요한 프로젝트인지 확인해주세요.")
                        
                except Exception as e:
                    st.error(f"시스템 오류 발생: {e}")
    else:
        if not GENAI_AVAILABLE:
            st.warning("⚠️ `google-generativeai` 라이브러리가 설치되지 않았습니다. 앱을 재시작하면 자동 설치됩니다.")
        else:
            st.info("💡 사이드바에 **Gemini API 키**를 입력하면 여기서 바로 결과를 볼 수 있습니다! (지금은 수동 모드)")
        
        st.code(prompt_text, language="text")
        st.link_button("🚀 Google Gemini 열기 (붙여넣기 하러 가기)", "https://gemini.google.com")

# -----------------------
# 6. UI 및 대시보드 (시장 위험 감지 포함)
# -----------------------
def get_market_risk_status():
    status = {}
    start_dt = datetime.now() - timedelta(days=365)
    
    def analyze_index_risk(ticker, name):
        try:
            df = yf.download(ticker, start=start_dt, progress=False, threads=False)
            df = normalize_columns(df)
            if df is None or len(df) < 60: return None
            
            close = df["Close"].iloc[-1]
            ma5 = df["Close"].rolling(5).mean().iloc[-1]
            ma20 = df["Close"].rolling(20).mean().iloc[-1]
            rsi = ta.rsi(df["Close"], 14).iloc[-1]
            
            risk_level = "안전"
            color = "green"
            msg = "상승 추세 유지 중"
            
            if close < ma5:
                risk_level = "주의 (단기조정)"
                color = "orange"
                msg = "5일선 이탈! 단기 하락 가능성"
            
            if close < ma20:
                risk_level = "위험 (추세하락)"
                color = "red"
                msg = "20일선 붕괴! 현금확보/인버스 고려"
                
            if rsi > 70:
                risk_level = "경고 (과열)"
                color = "orange"
                msg = f"RSI {rsi:.0f} (과매수 구간, 조정 주의)"
            
            return {"name": name, "level": risk_level, "color": color, "msg": msg, "close": close, "is_danger": (close < ma20 or close < ma5)}
        except: return None

    status["KOSPI"] = analyze_index_risk("^KS11", "코스피")
    status["KOSDAQ"] = analyze_index_risk("^KQ11", "코스닥")
    status["NASDAQ"] = analyze_index_risk("^IXIC", "나스닥")
    return status

def get_coin_market_status():
    if not UPBIT_AVAILABLE: return None
    
    status = {}
    try:
        btc_price = pyupbit.get_current_price("KRW-BTC")
        df_btc = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=60)
        btc_ma5 = df_btc['close'].rolling(5).mean().iloc[-1]
        btc_change = (btc_price - df_btc['close'].iloc[-2]) / df_btc['close'].iloc[-2] * 100
        
        status["BTC"] = {
            "name": "비트코인 (대장주)", 
            "price": btc_price, 
            "change": btc_change,
            "trend": "상승세" if btc_price > btc_ma5 else "조정/하락"
        }
        
        eth_price = pyupbit.get_current_price("KRW-ETH")
        df_eth = pyupbit.get_ohlcv("KRW-ETH", interval="day", count=60)
        eth_change = (eth_price - df_eth['close'].iloc[-2]) / df_eth['close'].iloc[-2] * 100
        
        status["ETH"] = {
            "name": "이더리움 (플랫폼)", 
            "price": eth_price, 
            "change": eth_change,
            "trend": "상승세" if eth_price > df_eth['close'].rolling(5).mean().iloc[-1] else "조정/하락"
        }
        
        alts = ["KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-DOGE", "KRW-SEI"]
        alt_changes = []
        for a in alts:
            try:
                curr = pyupbit.get_current_price(a)
                prev = pyupbit.get_ohlcv(a, interval="day", count=2)['close'].iloc[0]
                alt_changes.append((curr - prev)/prev * 100)
            except: pass
            
        avg_alt_change = sum(alt_changes) / len(alt_changes) if alt_changes else 0
        
        status["ALTS"] = {
            "name": "알트코인 평균 (시장심리)",
            "price": 0, 
            "change": avg_alt_change,
            "trend": "불장🔥" if avg_alt_change > 1.0 else ("공포🥶" if avg_alt_change < -1.0 else "횡보")
        }
        
        return status
    except: return None

def get_market_status_full():
    status = {}
    start_dt = datetime.now() - timedelta(days=730)
    try:
        usd = yf.download("KRW=X", start=start_dt, progress=False, threads=False)
        vix = yf.download("^VIX", start=start_dt, progress=False, threads=False)
        usd = normalize_columns(usd); vix = normalize_columns(vix)
        if not usd.empty:
            close = usd["Close"].iloc[-1]; ma20 = usd["Close"].rolling(20).mean().iloc[-1]; prev = usd["Close"].iloc[-2]
            status["USD"] = {"val": f"{close:.1f}원", "delta": f"{close-prev:+.1f}", "msg": "안정" if close < ma20 else "주의", "color": "green" if close < ma20 else "red"}
        if not vix.empty:
            close = vix["Close"].iloc[-1]; prev = vix["Close"].iloc[-2]
            status["VIX"] = {"val": f"{close:.2f}", "delta": f"{close-prev:+.2f}", "msg": "안정" if close < 20 else "공포", "color": "green" if close < 20 else "red"}

        def analyze_index(ticker, name, is_coin=False):
            try:
                if is_coin and UPBIT_AVAILABLE: df = pyupbit.get_ohlcv(ticker, interval="day", count=365)
                else: df = yf.download(ticker, start=start_dt, progress=False, threads=False)
                df = normalize_columns(df)
                if df is None or len(df) < 240: return None
                close = df["Close"].iloc[-1]; ma20 = df["Close"].rolling(20).mean().iloc[-1]; ma60 = df["Close"].rolling(60).mean().iloc[-1]; ma120 = df["Close"].rolling(120).mean().iloc[-1]; ma240 = df["Close"].rolling(240).mean().iloc[-1]
                st_view = "📈상승" if close > ma20 else "📉하락"; mt_view = "📈상승" if ma20 > ma60 else "📉하락"; lt_view = "📈상승" if ma120 > ma240 else "📉하락"
                return {"name": name, "price": close, "st": st_view, "mt": mt_view, "lt": lt_view, "color_st": "red" if "하락" in st_view else "blue", "color_mt": "red" if "하락" in mt_view else "blue", "color_lt": "red" if "하락" in lt_view else "blue"}
            except: return None

        status["KOSPI"] = analyze_index("^KS11", "코스피")
        status["KOSDAQ"] = analyze_index("^KQ11", "코스닥")
        status["NASDAQ"] = analyze_index("^IXIC", "나스닥")
        status["BTC"] = analyze_index("KRW-BTC", "비트코인", is_coin=True)
    except: pass
    return status

@st.cache_data(ttl=3600)
def predict_kospi_levels():
    try:
        df = yf.download("^KS11", period="1y", interval="1d", progress=False, threads=False)
        df = normalize_columns(df)
        if df.empty: return None
        last = df.iloc[-1]; close = last["Close"]; high = last["High"]; low = last["Low"]
        
        pivot = (high+low+close)/3; r1 = (2*pivot)-low; s1 = (2*pivot)-high; r2 = pivot+(high-low); s2 = pivot-(high-low)
        sentiment = "중립 (기술적 분석)"
        
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        ma60 = df["Close"].rolling(60).mean().iloc[-1] # 장기 추세
        std = df["Close"].rolling(20).std().iloc[-1]
        m_h = ma20+(std*2); m_l = ma20-(std*2)
        
        rsi = ta.rsi(df["Close"], 14).iloc[-1]
        
        if close > ma20: sentiment = "📈 상승 추세 (MA20 상회)"
        else: sentiment = "📉 하락 추세 (MA20 하회)"
        
        today_dt = datetime.now()
        
        next_day = today_dt + timedelta(days=1)
        if next_day.weekday() >= 5:
            next_day += timedelta(days=(7 - next_day.weekday()))
        daily_range_str = f"({next_day.strftime('%m.%d')} 기준)"
        
        week_end = today_dt + timedelta(days=7)
        weekly_range_str = f"({today_dt.strftime('%m.%d')} ~ {week_end.strftime('%m.%d')})"
        
        month_end = today_dt + timedelta(days=30)
        monthly_range_str = f"({today_dt.strftime('%m.%d')} ~ {month_end.strftime('%m.%d')})"

        daily_view = "🟡 중립"
        if close > pivot and close > ma5: daily_view = "📈 상승"
        elif close < pivot and close < ma5: daily_view = "📉 하락"
        
        weekly_view = "📉 조정"
        if close > ma20: weekly_view = "📈 상승"
        
        monthly_view = "☁️ 약세"
        if close > ma60: monthly_view = "🌞 강세"

        return {
            "current": close, "sentiment": sentiment, 
            "d_high": r1, "d_low": s1, 
            "m_high": m_h, "m_low": m_l,
            "daily_view": daily_view, "weekly_view": weekly_view, "monthly_view": monthly_view,
            "daily_range": daily_range_str, "weekly_range": weekly_range_str, "monthly_range": monthly_range_str,
            "rsi": rsi
        }
    except: return None

# 화면 표시
st.title("Unified Screener Rank & Market Risk Alert")

st.markdown("### 🚨 주식 시장 위험 감지 (Stock Market Risk)")
risk_data = get_market_risk_status()
if risk_data:
    rc1, rc2, rc3 = st.columns(3)
    for idx, (key, col) in enumerate(zip(["KOSPI", "KOSDAQ", "NASDAQ"], [rc1, rc2, rc3])):
        if key in risk_data and risk_data[key]:
            d = risk_data[key]
            col.markdown(f"""
            <div style="border: 2px solid {d['color']}; border-radius: 10px; padding: 10px; text-align: center;">
                <h4>{d['name']}</h4>
                <h2 style="color: {d['color']}; margin: 0;">{d['level']}</h2>
                <p style="font-size: 14px; margin-top: 5px;">{d['msg']}</p>
                <p style="font-size: 12px; color: #666;">현재가: {d['close']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if d['is_danger']:
                st.warning(f"⚠️ **{d['name']} 하락 신호 감지!** 개별 종목 매수를 자제하고, **인버스(Inverse) ETF**를 주목하세요.")

st.divider()

st.markdown("### 🪙 가상화폐 시장 트렌드 (Crypto Radar)")
coin_data = get_coin_market_status()
if coin_data:
    cc1, cc2, cc3 = st.columns(3)
    
    d = coin_data["BTC"]
    clr = "red" if d['change'] > 0 else "blue"
    cc1.markdown(f"""
    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; border:1px solid #eee;">
        <div style="font-size:16px; font-weight:bold;">{d['name']}</div>
        <div style="font-size:24px; font-weight:bold; color:{clr};">{d['price']:,.0f}</div>
        <div style="color:{clr};">{d['change']:+.2f}%</div>
        <hr style="margin:5px 0;">
        <div style="font-size:12px; color:#555;">추세: {d['trend']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    d = coin_data["ETH"]
    clr = "red" if d['change'] > 0 else "blue"
    cc2.markdown(f"""
    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; border:1px solid #eee;">
        <div style="font-size:16px; font-weight:bold;">{d['name']}</div>
        <div style="font-size:24px; font-weight:bold; color:{clr};">{d['price']:,.0f}</div>
        <div style="color:{clr};">{d['change']:+.2f}%</div>
        <hr style="margin:5px 0;">
        <div style="font-size:12px; color:#555;">추세: {d['trend']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    d = coin_data["ALTS"]
    clr = "red" if d['change'] > 0 else "blue"
    cc3.markdown(f"""
    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; border:1px solid #eee;">
        <div style="font-size:16px; font-weight:bold;">{d['name']}</div>
        <div style="font-size:24px; font-weight:bold; color:{clr};">{d['change']:+.2f}%</div>
        <div style="color:#888;">(상위 알트 평균 등락)</div>
        <hr style="margin:5px 0;">
        <div style="font-size:12px; color:#555;">분위기: {d['trend']}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("⚠️ 업비트 데이터 로딩 실패 또는 pyupbit 미설치")

st.divider()

m_data = get_market_status_full()

if m_data:
    c1, c2 = st.columns(2)
    if "USD" in m_data:
        d = m_data["USD"]
        c1.metric("달러/원 환율", d["val"], d["delta"], delta_color="inverse")
        c1.caption(f":{d['color']}[{d['msg']}]")
    if "VIX" in m_data:
        d = m_data["VIX"]
        c2.metric("VIX (공포지수)", d["val"], d["delta"], delta_color="inverse")
        c2.caption(f":{d['color']}[{d['msg']}]")
    st.markdown("---")
    st.markdown("### 🌍 4대 시장 추세 (이평선 기준)")
    st.markdown("""<style>.trend-card { background-color:#f9f9f9; padding:15px; border-radius:10px; border:1px solid #e0e0e0; text-align:center; } .t-row { display:flex; justify-content:space-between; margin-bottom:5px; font-size:14px; }</style>""", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    idx_targets = [("KOSPI", mc1), ("KOSDAQ", mc2), ("NASDAQ", mc3), ("BTC", mc4)]
    for k, col in idx_targets:
        if k in m_data and m_data[k]:
            d = m_data[k]
            with col:
                st.markdown(f"""<div class="trend-card"><div style="font-size:18px; font-weight:bold;">{d['name']}</div><div style="font-size:22px; font-weight:bold; color:#333;">{d['price']:,.0f}</div><hr style="margin:10px 0;"><div class="t-row"><span>단기(20일):</span><span style="color:{d['color_st']}; font-weight:bold;">{d['st']}</span></div><div class="t-row"><span>중기(60일):</span><span style="color:{d['color_mt']}; font-weight:bold;">{d['mt']}</span></div><div class="t-row"><span>장기(240일):</span><span style="color:{d['color_lt']}; font-weight:bold;">{d['lt']}</span></div></div>""", unsafe_allow_html=True)

st.divider()
st.subheader("🎯 AI 코스피 지수 예측 (피벗 및 추세)")
kp_pred = predict_kospi_levels()
if kp_pred:
    st.markdown("#### 🧭 **기간별 AI 방향성 추천 (확률 기반)**")
    st.caption("※ 옵션 만기일 기준이 아닌, 이동평균선(MA) 기반의 추세 예측입니다.")
    
    p1, p2, p3 = st.columns(3)
    p1.metric(label=f"[단기] 일별 {kp_pred['daily_range']}", value=kp_pred['daily_view'], help="피벗 포인트 및 5일 이동평균선 기준")
    p2.metric(label=f"[중기] 주별 {kp_pred['weekly_range']}", value=kp_pred['weekly_view'], help="20일 이동평균선(생명선) 기준")
    p3.metric(label=f"[장기] 월별 {kp_pred['monthly_range']}", value=kp_pred['monthly_view'], help="60일 이동평균선(수급선) 기준")
    st.markdown("---")
    
    k1, k2, k3 = st.columns(3)
    k1.info(f"**현재 지수**: {kp_pred['current']:,.2f}\n\n**추세**: {kp_pred['sentiment']}\n\n**RSI**: {kp_pred['rsi']:.1f}")
    k2.success(f"**[내일 단기 저항/지지]**\n\n🔼 저항(고점): **{kp_pred['d_high']:,.2f}**\n\n🔽 지지(저점): **{kp_pred['d_low']:,.2f}**")
    k3.warning(f"**[이번 달 밴드폭 예상]**\n\n🔼 고점: **{kp_pred['m_high']:,.2f}**\n\n🔽 저점: **{kp_pred['m_low']:,.2f}**")
else: st.error("지수 데이터 로드 실패")
st.divider()

# -----------------------
# [설정값 저장 함수]
# -----------------------
def save_config_to_file(silent=True):
    keys_to_save = [
        "target_tfs", "date_input", "check_date_input", "start_date_input", "end_date_input", 
        "period_input", "use_krx", "top_n_each", "include_etf", "use_us", "top_n_us", "include_us_lev",
        "scan_coins", "sel_coins", "sma_5_20", "sma_20_60", "sma_60_120", "sma_120_240", 
        "w_sma_5_20", "w_sma_20_60", "w_sma_60_120", "w_sma_120_240", "chk_rsi", "chk_macd", 
        "chk_stoch", "chk_bb", "chk_adx", "chk_mfi", "chk_super", "w_macd", "w_rsi", 
        "w_adx", "w_stoch", "w_bb", "w_mfi", "w_super", "st1_k", "st1_d", "st1_s", 
        "st2_k", "st2_d", "st2_s", "st3_k", "st3_d", "st3_s", "max_w", "min_vol", 
        "min_cap", "top_k_input", "strict_filter", "use_backtest", "fin_filter_mode", 
        "strategy_opt", "chart_period_days", "chart_view_count", "use_adv_strategy",
        "show_ma5", "show_ma20", "show_ma60", "show_ma120", "show_resistance",
        "use_etf_sector_filter", "selected_sectors",
        "ui_strict", "ui_ai_recom", 
        "use_bottom_filter", "use_bb_trend_filter", "use_bull_filter", "use_stoch_3beat", 
        "use_wam_filter", 
        "include_monthly_div",
        "gemini_api_key",       
        "use_rsi_daily", "use_rsi_60m", "use_rsi_240m", "rsi_threshold_val" 
    ]
    data = {}
    for k in keys_to_save:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, (date, datetime)): val = val.strftime("%Y-%m-%d")
            data[k] = val
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
        if not silent: st.toast("✅ 설정 저장 완료")
    except: pass

def auto_save():
    save_config_to_file(silent=True)

today = date.today()
def default_trade_date():
    return today

trade_date_str = to_krx_date_str(st.session_state.get("date_input", default_trade_date()))

# -----------------------
# 사이드바 (Auto-save 적용)
# -----------------------
st.sidebar.title("설정 & 즐겨찾기")

if UPBIT_AVAILABLE:
    st.sidebar.success("✅ 업비트(pyupbit) 연결됨")
else:
    st.sidebar.error("❌ pyupbit 미설치 (터미널: pip install pyupbit)")

with st.sidebar.expander("★ 즐겨찾기 추가/삭제 (통합)", expanded=True):
    all_tickers_map = get_global_ticker_map(trade_date_str)
    search_sel = st.selectbox(
        "종목 검색 (이름 또는 코드)", 
        options=[""] + all_tickers_map,
        placeholder="예: 삼성, Apple, 비트코인...",
        index=0
    )
    if st.button("즐겨찾기 추가"):
        if search_sel and search_sel.strip():
            match = re.match(r"\[(.*?)\] (.*) \((.*)\)", search_sel)
            if not match:
                candidates = [x for x in all_tickers_map if search_sel.upper() in x.upper()]
                if candidates:
                    search_sel = candidates[0]
                    match = re.match(r"\[(.*?)\] (.*) \((.*)\)", search_sel)
            if match:
                mkt, nm, cd = match.groups()
                new_item = {"market": mkt, "code": cd, "name": nm}
                if not any(x['code'] == cd for x in st.session_state.fav_list):
                    st.session_state.fav_list.append(new_item)
                    save_favorites(st.session_state.fav_list)
                    st.success(f"추가완료: {nm}")
                    st.rerun()
                else: st.warning("이미 목록에 있습니다.")
            else:
                st.error("종목을 정확히 선택해주세요.")
        else: st.warning("종목을 선택해주세요.")

    st.markdown("---")
    st.markdown("**현재 목록**")
    if st.session_state.fav_list:
        for idx, item in enumerate(st.session_state.fav_list):
            c1, c2 = st.columns([3, 1])
            c1.text(f"{item['name']}")
            if c2.button("X", key=f"del_{idx}"):
                st.session_state.fav_list.pop(idx)
                save_favorites(st.session_state.fav_list)
                st.rerun()
    else: st.caption("비어있음")

# [NEW] Gemini API 입력창 & 진단 버튼
st.sidebar.markdown("---")
with st.sidebar.expander("🤖 Gemini API 설정 (자동 분석용)", expanded=True):
    st.caption("Google Gemini API 키를 입력하면, 사이트 이동 없이 바로 AI 분석 결과를 볼 수 있습니다.")
    gemini_api_key_input = st.text_input("Google API Key 입력", value=st.session_state.gemini_api_key, type="password")
    
    if gemini_api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = gemini_api_key_input
        auto_save()
        st.rerun()
    
    if st.button("🛠️ API 키 진단 (모델 확인)"):
        if not st.session_state.gemini_api_key:
            st.error("API 키를 먼저 입력해주세요.")
        else:
            try:
                genai.configure(api_key=st.session_state.gemini_api_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success(f"✅ 사용 가능한 모델 ({len(models)}개):")
                st.code("\n".join(models))
            except Exception as e:
                st.error(f"❌ 키 확인 실패: {e}")

    if not st.session_state.gemini_api_key:
        st.info("👉 [API 키 무료 발급받기](https://aistudio.google.com/app/apikey)")

st.sidebar.markdown("### 1. 분석 시간봉 선택")
target_tfs = st.sidebar.multiselect("데이터 수집 시간봉", ["일봉 (Daily)", "주봉 (Weekly)", "60분봉 (1H)", "240분봉 (4H)"], default=["일봉 (Daily)"], key="target_tfs", on_change=auto_save)

st.sidebar.markdown("---")
with st.sidebar.expander("🛠 고급 필터 옵션 (자동/수동 공통)", expanded=True):
    st.markdown("**1. 기본 필터**")
    ui_strict = st.checkbox("🔥 강력 필터 (이평선 정배열 등)", key="ui_strict", on_change=auto_save)
    ui_ai_recom = st.checkbox("🤖 AI 추천 신호만 보기 (관망 제외)", key="ui_ai_recom", on_change=auto_save)
    
    st.markdown("---")
    st.markdown("**2. 전략 필터 (선택)**")
    
    st.markdown("##### 🚀 W·A·M 바닥 콤보 (신규)")
    use_wam_filter = st.checkbox("✅ W·A·M 필터 적용", key="use_wam_filter", on_change=auto_save)
    st.caption("조건: WillR(-80↓) + ADX(20↑) + MFI(40↓)")

    st.markdown("##### 🌊 스토캐스틱 3박자 (강력 추천)")
    use_stoch_3beat = st.checkbox("✅ 3박자 모두 상승만 보기", key="use_stoch_3beat", on_change=auto_save)
    st.caption("조건: 단기/중기/장기 스토캐스틱 모두 K > D")

    st.markdown("##### 💎 바닥권 보물찾기")
    use_bottom_filter = st.checkbox("✅ 보물찾기 필터 적용", key="use_bottom_filter", on_change=auto_save)
    st.caption("조건: WillR(-80) + OBV(매집) + MFI(40) + Stoch_K(20)")
    
    st.markdown("##### 🎯 볼린저밴드+추세 급등")
    use_bb_trend_filter = st.checkbox("✅ 급등 추세 필터 적용", key="use_bb_trend_filter", on_change=auto_save)
    st.caption("조건: BB중단 위(강세) + RSI(50↑) + MACD(상승)")
    
    st.markdown("##### 🔥 불타는 승률 필터")
    use_bull_filter = st.checkbox("✅ 승률 필터 적용", key="use_bull_filter", on_change=auto_save)
    st.caption("조건: Supertrend(상승) + ADX(25↑) + MFI(50↑)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. 이평선(SMA) 정배열 조건")
c1, c2 = st.sidebar.columns(2)
with c1:
    sma_5_20 = st.checkbox("5 > 20 (단기)", True, key="sma_5_20", on_change=auto_save)
    w_sma_5_20 = st.number_input("점수(5>20)", 0.0, 50.0, 2.0, 0.5, key="w_sma_5_20", on_change=auto_save)
    sma_60_120 = st.checkbox("60 > 120 (중장기)", False, key="sma_60_120", on_change=auto_save)
    w_sma_60_120 = st.number_input("점수(60>120)", 0.0, 50.0, 3.0, 0.5, key="w_sma_60_120", on_change=auto_save)
with c2:
    sma_20_60 = st.checkbox("20 > 60 (중기)", True, key="sma_20_60", on_change=auto_save)
    w_sma_20_60 = st.number_input("점수(20>60)", 0.0, 50.0, 3.0, 0.5, key="w_sma_20_60", on_change=auto_save)
    sma_120_240 = st.checkbox("120 > 240 (장기)", False, key="sma_120_240", on_change=auto_save)
    w_sma_120_240 = st.number_input("점수(120>240)", 0.0, 50.0, 5.0, 0.5, key="w_sma_120_240", on_change=auto_save)

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. 날짜 설정")
use_backtest = st.sidebar.checkbox("🕰️ 타임머신(백테스팅) 모드 사용", False, key="use_backtest", on_change=auto_save)

if use_backtest:
    st.sidebar.caption("👇 분석할 과거 기간(시작~종료)을 선택하세요.")
    an_start_date = st.sidebar.date_input("① 분석 시작일", default_trade_date(), key="start_date_input", on_change=auto_save)
    an_end_date = st.sidebar.date_input("② 분석 종료일", default_trade_date(), key="end_date_input", on_change=auto_save)
    st.sidebar.caption("👇 이 날짜 기준으로 수익률을 계산합니다.")
    check_date = st.sidebar.date_input("③ 수익 확인일 (현재)", date.today(), key="check_date_input", on_change=auto_save)
    if (an_end_date - an_start_date).days > 30:
        st.sidebar.warning("⚠️ 기간이 너무 길면 속도가 느려질 수 있습니다.")
else:
    an_start_date = None
    an_end_date = None
    check_date = date.today()

if check_date.weekday() >= 5: 
    st.sidebar.info(f"📅 오늘은 {check_date.strftime('%A')}입니다.\n- 주식: 금요일 종가 기준\n- 코인: 실시간 데이터")

period_days_input = st.sidebar.number_input("데이터 기간(일)", 30, 3650, 200, 30, key="period_input", on_change=auto_save)

st.sidebar.markdown("### 4. 시장 선택 (랭킹용)")
use_krx = st.sidebar.checkbox("국내 주식 (KRX)", True, key="use_krx", on_change=auto_save)
top_n_each = st.sidebar.slider("KRX 탐색 수", 50, 3000, 2000, disabled=not use_krx, key="top_n_each", on_change=auto_save)

include_etf = st.sidebar.checkbox("국내 ETF 포함 (메이저 브랜드만)", False, key="include_etf", on_change=auto_save)
use_etf_sector_filter = False
selected_sectors = []

if include_etf:
    use_etf_sector_filter = st.sidebar.checkbox("└─ 📂 ETF 섹터별로 골라보기", value=False, key="use_etf_sector_filter", on_change=auto_save)
    if use_etf_sector_filter:
        sector_map = get_etf_sector_keywords()
        selected_sectors = st.sidebar.multiselect(
            "   (섹터 다중 선택 가능)",
            options=list(sector_map.keys()),
            key="selected_sectors",
            on_change=auto_save
        )

include_monthly_div = st.sidebar.checkbox("💎 월배당 ETF (Monthly Payout)", False, key="include_monthly_div", on_change=auto_save)

use_us_stock = st.sidebar.checkbox("미국 주식 (S&P 500)", False, key="use_us", on_change=auto_save)
top_n_us = st.sidebar.slider("미국 탐색 수", 50, 3000, 2000, disabled=not use_us_stock, key="top_n_us", on_change=auto_save) 
include_us_lev = st.sidebar.checkbox("미국 레버리지/인버스 ETF (3x/2x)", False, key="include_us_lev", on_change=auto_save)
auto_scan_all_coins = st.sidebar.checkbox("업비트 전체 코인", False, key="scan_coins", disabled=not UPBIT_AVAILABLE, on_change=auto_save)

st.sidebar.markdown("---")
st.sidebar.markdown("### 5. 스토캐스틱 / 보조지표")
c1, c2, c3 = st.sidebar.columns(3)
with c1:
    st.markdown("**S1 (단기)**")
    st1_k = st.number_input("S1 K", 1, 100, 5, key="st1_k", on_change=auto_save)
    st1_d = st.number_input("S1 D", 1, 100, 3, key="st1_d", on_change=auto_save)
    st1_s = st.number_input("S1 S", 1, 100, 3, key="st1_s", on_change=auto_save)
with c2:
    st.markdown("**S2 (중기)**")
    st2_k = st.number_input("S2 K", 1, 100, 10, key="st2_k", on_change=auto_save)
    st2_d = st.number_input("S2 D", 1, 100, 6, key="st2_d", on_change=auto_save)
    st2_s = st.number_input("S2 S", 1, 100, 6, key="st2_s", on_change=auto_save)
with c3:
    st.markdown("**S3 (장기)**")
    st3_k = st.number_input("S3 K", 1, 100, 20, key="st3_k", on_change=auto_save)
    st3_d = st.number_input("S3 D", 1, 100, 12, key="st3_d", on_change=auto_save)
    st3_s = st.number_input("S3 S", 1, 100, 12, key="st3_s", on_change=auto_save)

use_rsi = st.sidebar.checkbox("RSI", True, key="chk_rsi", on_change=auto_save)
use_macd = st.sidebar.checkbox("MACD", True, key="chk_macd", on_change=auto_save)
use_stoch = st.sidebar.checkbox("Stochastic", True, key="chk_stoch", on_change=auto_save) 
use_bb = st.sidebar.checkbox("Bollinger", False, key="chk_bb", on_change=auto_save)
use_adx = st.sidebar.checkbox("ADX", True, key="chk_adx", on_change=auto_save)
use_mfi = st.sidebar.checkbox("MFI", True, key="chk_mfi", on_change=auto_save)
use_supertrend = st.sidebar.checkbox("Supertrend", True, key="chk_super", on_change=auto_save)

c1, c2 = st.sidebar.columns(2)
with c1:
    w_macd = st.number_input("MACD 점수", 0.0, 100.0, 3.0, 1.0, key="w_macd", on_change=auto_save)
    w_rsi = st.number_input("RSI 점수", 0.0, 100.0, 3.0, 1.0, key="w_rsi", on_change=auto_save)
    w_adx = st.number_input("ADX 점수", 0.0, 100.0, 2.0, 1.0, key="w_adx", on_change=auto_save)
with c2:
    w_stoch = st.number_input("Stoch 점수", 0.0, 100.0, 2.0, 1.0, key="w_stoch", on_change=auto_save)
    w_bb = st.number_input("BB 점수", 0.0, 100.0, 2.0, 1.0, key="w_bb", on_change=auto_save)
    w_mfi = st.number_input("MFI 점수", 0.0, 100.0, 2.0, 1.0, key="w_mfi", on_change=auto_save)
    w_supertrend = st.number_input("Supertrend 점수", 0.0, 100.0, 5.0, 1.0, key="w_super", on_change=auto_save)

max_workers = st.sidebar.slider("병렬 작업 수 (업비트 안전: 20 이하 권장)", 2, 100, 10, key="max_w", on_change=auto_save)
min_avg_volume = st.sidebar.number_input("최소 거래량", 0, step=1000, key="min_vol", on_change=auto_save)
min_market_cap_okr = st.sidebar.number_input("최소 시총(억)", 0, step=10, key="min_cap", on_change=auto_save)
top_k = st.sidebar.slider("출력 개수", 5, 5000, 2000, key="top_k_input", on_change=auto_save) 

strategy_option = st.sidebar.radio(
    "🔎 검색 전략 선택", 
    ["bottom", "default"], 
    key="strategy_opt",
    index=0,
    on_change=auto_save,
    format_func=lambda x: "📈 기본 (추세 추종)" if x=="default" else "💎 바닥권 매집 (저점 매수)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 6. 재무 필터 (선택)")
fin_filter_mode = st.sidebar.radio(
    "재무 건전성 조건 (주식 Only)",
    ["사용안함 (속도빠름)", "순이익 성장", "영업이익 성장", "성장성 OR (둘중하나)", "성장성 AND (둘다)"],
    key="fin_filter_mode",
    on_change=auto_save
)
st.sidebar.caption("⚠️ 재무 필터를 켜면 분석 시간이 더 소요됩니다.")

st.sidebar.markdown("---")
# [추가] 자동 감시 기능 UI
st.sidebar.markdown("### 🔄 자동 감시 모드")
monitor_interval_min = st.sidebar.number_input("감시 주기 (분)", 1, 120, 10)

col_m1, col_m2 = st.sidebar.columns(2)
with col_m1:
    if st.button("▶ 감시 시작", type="primary"):
        st.session_state.is_monitoring = True
        st.rerun()
with col_m2:
    if st.button("⏹ 감시 정지"):
        st.session_state.is_monitoring = False
        st.rerun()

c1, c2 = st.sidebar.columns(2)
with c1: run_button = st.button("스크리닝 실행 (수동)", type="primary")
with c2: 
    if st.button("설정 저장 (수동)"): save_config_to_file(silent=False)
if st.sidebar.button("캐시 초기화"):
    st.cache_data.clear()
    st.rerun()

# -----------------------
# 메인 출력부
# -----------------------
tab_all, tab_fav, tab_accum = st.tabs(["🔍 전체 종목 랭킹", "★ 즐겨찾기 모음", "🕒 실시간 포착 모음(New)"])

stoch_params = [{"k":st1_k,"d":st1_d,"s":st1_s},{"k":st2_k,"d":st2_d,"s":st2_s},{"k":st3_k,"d":st3_d,"s":st3_s}]
toggles = {
    "rsi": use_rsi, "macd": use_macd, "bb": use_bb, "stoch": use_stoch,
    "adx": use_adx, "mfi": use_mfi, "supertrend": use_supertrend
}
weights = {
    "sma_5_20": w_sma_5_20, "sma_20_60": w_sma_20_60, "sma_60_120": w_sma_60_120, "sma_120_240": w_sma_120_240,
    "rsi": w_rsi, "macd": w_macd, "bb": w_bb, "stoch": w_stoch,
    "adx": w_adx, "mfi": w_mfi, "supertrend": w_supertrend
}
sma_config = {"5_20": sma_5_20, "20_60": sma_20_60, "60_120": sma_60_120, "120_240": sma_120_240}

ys, ye = yf_date_range(check_date, int(period_days_input))

# [추가] 고급 필터링 로직 (공통 사용)
def apply_advanced_filters(df, use_strict, use_ai_recom, use_stoch_3beat, use_bottom, use_bb, use_bull, use_wam,
                           use_rsi_daily, use_rsi_60m, use_rsi_240m, rsi_threshold_val):
    if df.empty: return df
    
    if use_strict and "StrictPass" in df.columns:
        df = df[df["StrictPass"] == True]
    
    if use_ai_recom and "AI신호" in df.columns:
        df = df[~df["AI신호"].str.contains("관망|하락위험|과열주의")]
    
    if use_stoch_3beat and "stoch_hits" in df.columns:
        df = df[df["stoch_hits"] >= 3]

    if use_bottom:
        df["_willr"] = pd.to_numeric(df["WillR"], errors='coerce')
        df["_mfi"] = pd.to_numeric(df["MFI"], errors='coerce')
        c_w = df["_willr"] <= -80
        c_o = df["수급(OBV)"].str.contains("매집", na=False)
        c_m = df["_mfi"] <= 40
        c_s = df["Stoch_K"] <= 20
        df = df[c_w & c_o & c_m & c_s]

    if use_bb:
        if "BB_Position" in df.columns:
            df["_rsi"] = pd.to_numeric(df["RSI"], errors='coerce')
            c_bb = df["BB_Position"].str.contains("2_|3_", na=False) 
            c_rsi = df["_rsi"] >= 50
            c_macd = df["중기"].str.contains("상승", na=False)
            df = df[c_bb & c_rsi & c_macd]
    
    if use_bull:
        df["_adx"] = pd.to_numeric(df["ADX"], errors='coerce')
        df["_mfi"] = pd.to_numeric(df["MFI"], errors='coerce')
        c_sup = df["Supertrend_Status"].str.contains("상승", na=False)
        c_adx = df["_adx"] >= 25
        c_mfi = df["_mfi"] >= 50
        df = df[c_sup & c_adx & c_mfi]
        
    if use_wam:
        df["_willr"] = pd.to_numeric(df["WillR"], errors='coerce')
        df["_adx"] = pd.to_numeric(df["ADX"], errors='coerce')
        df["_mfi"] = pd.to_numeric(df["MFI"], errors='coerce')
        c_w = df["_willr"] <= -80
        c_a = df["_adx"] >= 20
        c_m = df["_mfi"] <= 40
        df = df[c_w & c_a & c_m]

    rsi_conditions = []
    if "Timeframe" not in df.columns:
        if use_rsi_daily or use_rsi_60m or use_rsi_240m:
            df = df[pd.to_numeric(df["RSI"], errors='coerce') <= rsi_threshold_val]
    else:
        if use_rsi_daily:
            rsi_conditions.append((df["Timeframe"].str.contains("일봉")) & (pd.to_numeric(df["RSI"], errors='coerce') <= rsi_threshold_val))
        if use_rsi_60m:
            rsi_conditions.append((df["Timeframe"].str.contains("60분")) & (pd.to_numeric(df["RSI"], errors='coerce') <= rsi_threshold_val))
        if use_rsi_240m:
            rsi_conditions.append((df["Timeframe"].str.contains("240분")) & (pd.to_numeric(df["RSI"], errors='coerce') <= rsi_threshold_val))
        
        if rsi_conditions:
            final_rsi_condition = rsi_conditions[0]
            for cond in rsi_conditions[1:]:
                final_rsi_condition |= cond 
            df = df[final_rsi_condition]
        
    return df

# [추가] 스크리닝 로직을 함수화하여 수동 실행 및 자동 감시 모두 사용
def run_screening_logic():
    targets = []
    if use_krx:
        kp, kd = fetch_krx_tickers_fdr()
        if min_market_cap_okr > 0:
            th = min_market_cap_okr * 100_000_000
            if not kp.empty: kp = kp[kp["Marcap"] >= th]
            if not kd.empty: kd = kd[kd["Marcap"] >= th]
        
        def add_t(df, mkt):
            if df.empty: return
            for _, row in df.sort_values("Marcap", ascending=False).head(top_n_each).iterrows():
                targets.append({"code":row['Code'], "market":mkt, "name":row['Name']})
                
        add_t(kp, "KOSPI"); add_t(kd, "KOSDAQ")
    
    if include_etf:
        etf_list = fetch_filtered_etf_targets_fdr()
        if use_etf_sector_filter and selected_sectors:
            filtered_etf_list = []
            sector_keywords_map = get_etf_sector_keywords()
            target_keywords = []
            for sector_name in selected_sectors:
                if sector_name in sector_keywords_map:
                    target_keywords.extend(sector_keywords_map[sector_name])
            
            for etf in etf_list:
                etf_name = etf["name"].upper().replace(" ", "")
                if any(k.upper() in etf_name for k in target_keywords):
                    filtered_etf_list.append(etf)
            targets.extend(filtered_etf_list)
        else:
            targets.extend(etf_list)

    if include_monthly_div:
        try:
            df_all_etf = fdr.StockListing('ETF/KR')
            for _, row in df_all_etf.iterrows():
                if "월배당" in row['Name']:
                    targets.append({"code": row['Symbol'], "market": "ETF", "name": row['Name']})
        except: pass
    
    if use_us_stock:
        us_tickers = get_sp500_tickers_fdr(limit=top_n_us)
        targets.extend(us_tickers)
    
    if include_us_lev:
        us_lev_list = get_us_leveraged_etfs()
        targets.extend(us_lev_list)

    if auto_scan_all_coins and UPBIT_AVAILABLE:
        coins = build_coin_list(fetch_all_upbit_krw_tickers())
        targets.extend(coins)
    
    unique_targets = {t['code']: t for t in targets}.values()
    targets = list(unique_targets)

    final_results = []
    use_strict_sidebar = False 

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_single_ticker_range, tk, ys, ye, target_tfs, stoch_params, toggles, weights, min_avg_volume, sma_config, False, strategy_option, use_strict_sidebar, an_start_date, an_end_date): tk for tk in targets}
        for f in as_completed(futures):
            res_list = f.result()
            if res_list: final_results.extend(res_list)
    
    if final_results and fin_filter_mode != "사용안함 (속도빠름)":
        filtered_results = []
        for res in final_results:
            if res["market"] in ["UPBIT", "COIN", "ETF", "US_ETF"]:
                filtered_results.append(res)
                continue
            
            ni_g, oi_g = fetch_financial_growth(res["ticker"])
            n_ok = (not np.isnan(ni_g) and ni_g > 0)
            o_ok = (not np.isnan(oi_g) and oi_g > 0)
            
            pass_filter = False
            if fin_filter_mode == "순이익 성장" and n_ok: pass_filter = True
            elif fin_filter_mode == "영업이익 성장" and o_ok: pass_filter = True
            elif fin_filter_mode == "성장성 OR (둘중하나)" and (n_ok or o_ok): pass_filter = True
            elif fin_filter_mode == "성장성 AND (둘다)" and (n_ok and o_ok): pass_filter = True
            
            if pass_filter:
                filtered_results.append(res)
        
        final_results = filtered_results

    risk_status = get_market_risk_status()
    is_market_danger = False
    
    if risk_status:
        for k in ["KOSPI", "KOSDAQ"]:
            if k in risk_status and risk_status[k] is not None and risk_status[k].get('is_danger', False):
                is_market_danger = True
    
    if final_results:
        for res in final_results:
            nm = res['name']
            
            if "월배당" in nm:
                res['score'] += 210
                res['AI신호'] = "💎월배당Top"

            elif is_market_danger:
                if "인버스" in nm or "곱버스" in nm or "Inverse" in nm or "Short" in nm:
                    res['score'] += 200
                    res['AI신호'] = "🚨시장위험대응"

    return pd.DataFrame(final_results).sort_values(["포착일", "score"], ascending=[False, False]) if final_results else pd.DataFrame()

# [추가] 자동 감시 실행 로직
if st.session_state.is_monitoring:
    placeholder = st.empty()
    st.toast(f"🔄 자동 감시가 시작되었습니다. ({monitor_interval_min}분 간격)")
    
    while True:
        with placeholder.container():
            st.info(f"🔄 [현재 시간: {datetime.now().strftime('%H:%M:%S')}] 종목 스캔 중... (화면을 닫지 마세요)")
            
            raw_df = run_screening_logic()
            new_df = apply_advanced_filters(raw_df, 
                                            st.session_state.ui_strict, 
                                            st.session_state.ui_ai_recom,
                                            st.session_state.use_stoch_3beat,
                                            st.session_state.use_bottom_filter,
                                            st.session_state.use_bb_trend_filter,
                                            st.session_state.use_bull_filter,
                                            st.session_state.use_wam_filter, 
                                            st.session_state.use_rsi_daily,
                                            st.session_state.use_rsi_60m,
                                            st.session_state.use_rsi_240m,
                                            st.session_state.rsi_threshold_val)
            
            if not new_df.empty:
                combined = pd.concat([new_df, st.session_state.df_accumulated])
                combined = combined.drop_duplicates(subset=['code'], keep='first')
                combined = combined.reset_index(drop=True) 
                
                st.session_state.df_accumulated = combined
                st.session_state.df_results = raw_df
            
            found_count = len(new_df)
            msg = f"✅ 스캔 완료! 필터 조건에 맞는 {found_count}개 종목 발견."
            if found_count > 0: st.success(msg)
            else: st.warning("필터 조건에 맞는 종목이 없습니다.")
            
            wait_time = monitor_interval_min * 60
            progress_text = "다음 스캔까지 대기 중입니다... (정지하려면 사이드바 '⏹ 감시 정지' 클릭)"
            my_bar = st.progress(0, text=progress_text)
            
            step = 100
            if wait_time < 100: step = int(wait_time)
            
            for i in range(step):
                time.sleep(wait_time / step)
                my_bar.progress(int((i+1) / step * 100), text=f"{progress_text} ({int(wait_time - (i+1)*(wait_time/step))}초 남음)")
            
            st.rerun()

# 수동 실행 버튼 로직
if run_button:
    if not target_tfs:
        st.error("시간봉을 하나 이상 선택해주세요.")
    else:
        with st.spinner("수동 분석 실행 중..."):
            raw = run_screening_logic()
            st.session_state.df_results = raw 

# -----------------------
# [탭 1] 전체 종목 랭킹
# -----------------------
with tab_all:
    st.markdown("### 🏆 전체 시장 랭킹")
    st.info("💡 **팁:** 표에서 원하는 종목 행을 클릭하면 하단에 **AI 상세 분석 리포트**가 표시됩니다.")

    if st.session_state.df_results is not None and not st.session_state.df_results.empty:
        df = st.session_state.df_results.copy()
        
        with st.expander("🛠 검색 결과 필터링 (클릭하여 펼치기)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_strict = st.checkbox("🔥 강력 필터 (이평선 등)", value=st.session_state.ui_strict)
                f_3beat = st.checkbox("🌊 스토캐스틱 3박자", value=st.session_state.use_stoch_3beat)
                f_wam = st.checkbox("🚀 W·A·M 콤보 (WillR+ADX+MFI)", value=st.session_state.use_wam_filter)
            with c2:
                f_ai = st.checkbox("🤖 AI 추천 신호만 보기", value=st.session_state.ui_ai_recom)
                f_bottom = st.checkbox("💎 바닥권 보물찾기", value=st.session_state.use_bottom_filter)
            with c3:
                f_bb = st.checkbox("🎯 볼린저밴드 급등", value=st.session_state.use_bb_trend_filter)
                f_bull = st.checkbox("🔥 불타는 승률 필터", value=st.session_state.use_bull_filter)
            
            st.divider()
            with st.container(border=True):
                st.subheader("📉 RSI 과매도 심층 필터 (시간봉별)")
                rsi_th = st.slider("👇 기준값 선택 (이 값보다 낮으면 포착)", 20, 40, st.session_state.rsi_threshold_val, key="rsi_threshold_val", on_change=auto_save)
                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                with r1: 
                    st.markdown("**[일봉]**")
                    st.checkbox(f"RSI {rsi_th}↓", key="use_rsi_daily", value=st.session_state.use_rsi_daily, on_change=auto_save)
                with r2: 
                    st.markdown("**[60분봉]**")
                    st.checkbox(f"RSI {rsi_th}↓", key="use_rsi_60m", value=st.session_state.use_rsi_60m, on_change=auto_save)
                with r3: 
                    st.markdown("**[240분봉]**")
                    st.checkbox(f"RSI {rsi_th}↓", key="use_rsi_240m", value=st.session_state.use_rsi_240m, on_change=auto_save)
                
                if st.session_state.use_rsi_60m and "60분봉 (1H)" not in st.session_state.target_tfs:
                    st.warning("⚠️ 주의: 사이드바에서 [60분봉]을 선택해야 60분 데이터를 가져올 수 있습니다!")

        df = apply_advanced_filters(df, f_strict, f_ai, f_3beat, f_bottom, f_bb, f_bull, f_wam,
                                    st.session_state.use_rsi_daily, 
                                    st.session_state.use_rsi_60m, 
                                    st.session_state.use_rsi_240m,
                                    rsi_th)
        
        c1, c2 = st.columns(2)
        min_stoch_hit = c1.slider("최소 스토캐스틱 만족 횟수", 0, 10, 0, key="t2_slider") 
        min_final_score = c2.number_input("최소 총점 (Score)", 0.0, step=1.0, key="t2_score")
        
        df = df[(df["stoch_hits"] >= min_stoch_hit) & (df["score"] >= min_final_score)]

        trend_opt = st.radio("추세 조건 선택", ["전체 보기", "3박자 모두 상승", "단기+중기 상승", "2개 이상 상승"], index=0, horizontal=True)
        if trend_opt == "3박자 모두 상승":
            df = df[(df["단기"].str.contains("상승|강세")) & (df["중기"].str.contains("상승|강세")) & (df["장기"].str.contains("상승|강세"))]
        elif trend_opt == "단기+중기 상승":
            df = df[(df["단기"].str.contains("상승|강세")) & (df["중기"].str.contains("상승|강세"))]
        elif trend_opt == "2개 이상 상승":
             c_t = df["단기"].str.contains("상승|강세").astype(int) + df["중기"].str.contains("상승|강세").astype(int) + df["장기"].str.contains("상승|강세").astype(int)
             df = df[c_t >= 2]
        
        df = df.sort_values(["포착일", "score"], ascending=[False, False]).head(top_k).reset_index(drop=True)
        df = df.drop_duplicates(subset=['code'], keep='first')
        
        # [수정] Timeframe 제거 확인
        cols = ["포착일", "market", "ticker", "name", "AI신호", "RSI", "score", "수익률", "Close", "단기", "중기", "장기", "수급(OBV)", "사이클(CCI)", "WillR", "ADX", "MFI", "일목(구름)", "파라볼릭", "Reasons"]
        avail = [c for c in cols if c in df.columns]
        
        event = st.dataframe(
            df[avail].style.format({
                "score": safe_fmt, "Close": safe_fmt_price, "수익률": safe_fmt_profit, "ADX": safe_fmt, "MFI": safe_fmt, "RSI": safe_fmt
            }).background_gradient(subset=["score"], cmap="Greens"),
            on_select="rerun", 
            selection_mode="single-row",
            use_container_width=True
        )
        
        if len(event.selection.rows) > 0:
            sel_idx = event.selection.rows[0]
            sel_row = df.iloc[sel_idx]
            st.markdown("---")
            display_detailed_ai_report_v3(sel_row["ticker"], sel_row["market"], sel_row["name"], row_data=sel_row)
        
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("결과 다운로드", csv, "rank_result.csv", "text/csv")
    else: st.warning("결과가 없거나 실행되지 않았습니다.")

# -----------------------
# [탭 2] 즐겨찾기 모음
# -----------------------
with tab_fav:
    st.markdown("### ★ 내 관심 종목 상세 분석")
    st.info("💡 **팁:** 표에서 원하는 종목 행을 클릭하면 하단에 **AI 상세 분석 리포트**가 표시됩니다.")
    
    if st.button("🔄 즐겨찾기 종목만 빠르게 분석하기"):
        if st.session_state.fav_list:
             with st.spinner("분석 중..."):
                fav_results = []
                for item in st.session_state.fav_list:
                    res_list = process_single_ticker_range(
                        item, ys, ye, target_tfs, stoch_params, toggles, weights, 0, sma_config, is_favorite=True, strategy="default", apply_strict=False, 
                        start_date=None, end_date=None 
                    )
                    if res_list: fav_results.extend(res_list) 
                if fav_results:
                    st.session_state.df_favorites = pd.DataFrame(fav_results).sort_values("포착일", ascending=False)
                else: st.warning("데이터를 가져올 수 없습니다.")

    with st.expander("🔍 즐겨찾기 결과 내 재검색 (클릭)", expanded=False):
         c1, c2, c3 = st.columns(3)
         with c1: use_bottom_filter_fav = st.checkbox("보물찾기 필터 적용", key="fav_bottom")
         with c2: use_bb_trend_filter_fav = st.checkbox("급등 추세 필터 적용", key="fav_bb_trend")
         with c3: use_bull_filter_fav = st.checkbox("승률 필터 적용", key="fav_bull")
         use_wam_filter_fav = st.checkbox("🚀 W·A·M 콤보 필터 적용", key="fav_wam")

    if st.session_state.df_favorites is not None and not st.session_state.df_favorites.empty:
        df_fav = st.session_state.df_favorites.copy()

        if use_bottom_filter_fav:
            df_fav["_willr"] = pd.to_numeric(df_fav["WillR"], errors='coerce')
            df_fav["_mfi"] = pd.to_numeric(df_fav["MFI"], errors='coerce')
            c_w = df_fav["_willr"] <= -80
            c_o = df_fav["수급(OBV)"].str.contains("매집", na=False)
            c_m = df_fav["_mfi"] <= 40
            c_s = df_fav["Stoch_K"] <= 20
            df_fav = df_fav[c_w & c_o & c_m & c_s]
            
        if use_bb_trend_filter_fav:
            if "BB_Position" in df_fav.columns:
                df_fav["_rsi"] = pd.to_numeric(df_fav["RSI"], errors='coerce')
                c_bb = df_fav["BB_Position"].str.contains("2_|3_", na=False)
                c_rsi = df_fav["_rsi"] >= 50
                c_macd = df_fav["중기"].str.contains("상승", na=False)
                df_fav = df_fav[c_bb & c_rsi & c_macd]

        if use_bull_filter_fav:
            df_fav["_adx"] = pd.to_numeric(df_fav["ADX"], errors='coerce')
            df_fav["_mfi"] = pd.to_numeric(df_fav["MFI"], errors='coerce')
            c_sup = df_fav["Supertrend_Status"].str.contains("상승", na=False)
            c_adx = df_fav["_adx"] >= 25
            c_mfi = df_fav["_mfi"] >= 50
            df_fav = df_fav[c_sup & c_adx & c_mfi]
            
        if use_wam_filter_fav:
            df_fav["_willr"] = pd.to_numeric(df_fav["WillR"], errors='coerce')
            df_fav["_adx"] = pd.to_numeric(df_fav["ADX"], errors='coerce')
            df_fav["_mfi"] = pd.to_numeric(df_fav["MFI"], errors='coerce')
            c_w = df_fav["_willr"] <= -80
            c_a = df_fav["_adx"] >= 20
            c_m = df_fav["_mfi"] <= 40
            df_fav = df_fav[c_w & c_a & c_m]
        
        df_fav = df_fav.drop_duplicates(subset=['code'], keep='first')

        cols = ["포착일", "market", "ticker", "name", "AI신호", "score", "수익률", "Close", "단기", "중기", "장기", "수급(OBV)", "사이클(CCI)", "WillR", "ADX", "MFI", "RSI", "일목(구름)", "파라볼릭", "Reasons"]
        avail = [c for c in cols if c in df_fav.columns]
        
        event_fav = st.dataframe(
            df_fav[avail].style.format({
                "score": safe_fmt, "Close": safe_fmt_price, "수익률": safe_fmt_profit, "ADX": safe_fmt, "MFI": safe_fmt, "RSI": safe_fmt
            }).background_gradient(subset=["score"], cmap="Oranges"),
            on_select="rerun", 
            selection_mode="single-row",
            use_container_width=True
        )

        if len(event_fav.selection.rows) > 0:
            sel_idx = event_fav.selection.rows[0]
            sel_row = df_fav.iloc[sel_idx]
            st.markdown("---")
            display_detailed_ai_report_v3(sel_row["ticker"], sel_row["market"], sel_row["name"], row_data=sel_row)

    else: st.info("👆 위 버튼을 눌러 분석을 실행하거나, 종목을 추가해주세요.")

# -----------------------
# [탭 3] 실시간 포착 모음 (누적)
# -----------------------
with tab_accum:
    st.markdown("### 🕒 실시간 자동 감시 포착 리스트")
    st.markdown("자동 감시 모드(`▶ 감시 시작`)가 켜져 있는 동안 포착된 종목들이 이곳에 누적됩니다.")
    
    if st.button("🗑️ 누적 기록 초기화"):
        st.session_state.df_accumulated = pd.DataFrame()
        st.rerun()

    if not st.session_state.df_accumulated.empty:
        df_acc = st.session_state.df_accumulated.copy()
        
        df_acc = df_acc.loc[:, ~df_acc.columns.duplicated()]
        df_acc = df_acc.reset_index(drop=True)
        
        # [NEW] 실시간 탭에도 메인 탭과 동일한 필터링 UI 적용
        with st.expander("🛠 누적 결과 필터링 (클릭하여 펼치기)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_strict_acc = st.checkbox("🔥 강력 필터 (이평선 등)", value=st.session_state.ui_strict, key="acc_strict")
                f_3beat_acc = st.checkbox("🌊 스토캐스틱 3박자", value=st.session_state.use_stoch_3beat, key="acc_3beat")
                f_wam_acc = st.checkbox("🚀 W·A·M 콤보 (WillR+ADX+MFI)", value=st.session_state.use_wam_filter, key="acc_wam")
            with c2:
                f_ai_acc = st.checkbox("🤖 AI 추천 신호만 보기", value=st.session_state.ui_ai_recom, key="acc_ai")
                f_bottom_acc = st.checkbox("💎 바닥권 보물찾기", value=st.session_state.use_bottom_filter, key="acc_bottom")
            with c3:
                f_bb_acc = st.checkbox("🎯 볼린저밴드 급등", value=st.session_state.use_bb_trend_filter, key="acc_bb")
                f_bull_acc = st.checkbox("🔥 불타는 승률 필터", value=st.session_state.use_bull_filter, key="acc_bull")
            
            st.divider()
            with st.container(border=True):
                st.subheader("📉 RSI 과매도 심층 필터 (시간봉별)")
                rsi_th_acc = st.slider("👇 기준값 선택 (이 값보다 낮으면 포착)", 20, 40, st.session_state.rsi_threshold_val, key="acc_rsi_th")
                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                with r1: 
                    st.markdown("**[일봉]**")
                    f_rsi_daily_acc = st.checkbox(f"RSI {rsi_th_acc}↓", value=st.session_state.use_rsi_daily, key="acc_rsi_daily")
                with r2: 
                    st.markdown("**[60분봉]**")
                    f_rsi_60m_acc = st.checkbox(f"RSI {rsi_th_acc}↓", value=st.session_state.use_rsi_60m, key="acc_rsi_60m")
                with r3: 
                    st.markdown("**[240분봉]**")
                    f_rsi_240m_acc = st.checkbox(f"RSI {rsi_th_acc}↓", value=st.session_state.use_rsi_240m, key="acc_rsi_240m")

        df_acc = apply_advanced_filters(df_acc, f_strict_acc, f_ai_acc, f_3beat_acc, f_bottom_acc, f_bb_acc, f_bull_acc, f_wam_acc,
                                        f_rsi_daily_acc, f_rsi_60m_acc, f_rsi_240m_acc, rsi_th_acc)
        
        c1, c2 = st.columns(2)
        min_stoch_hit_acc = c1.slider("최소 스토캐스틱 만족 횟수", 0, 10, 0, key="acc_slider_stoch") 
        min_final_score_acc = c2.number_input("최소 총점 (Score)", 0.0, step=1.0, key="acc_score_input")
        
        df_acc = df_acc[(df_acc["stoch_hits"] >= min_stoch_hit_acc) & (df_acc["score"] >= min_final_score_acc)]

        trend_opt_acc = st.radio("추세 조건 선택", ["전체 보기", "3박자 모두 상승", "단기+중기 상승", "2개 이상 상승"], index=0, horizontal=True, key="acc_trend_opt")
        if trend_opt_acc == "3박자 모두 상승":
            df_acc = df_acc[(df_acc["단기"].str.contains("상승|강세")) & (df_acc["중기"].str.contains("상승|강세")) & (df_acc["장기"].str.contains("상승|강세"))]
        elif trend_opt_acc == "단기+중기 상승":
            df_acc = df_acc[(df_acc["단기"].str.contains("상승|강세")) & (df_acc["중기"].str.contains("상승|강세"))]
        elif trend_opt_acc == "2개 이상 상승":
             c_t = df_acc["단기"].str.contains("상승|강세").astype(int) + df_acc["중기"].str.contains("상승|강세").astype(int) + df_acc["장기"].str.contains("상승|강세").astype(int)
             df_acc = df_acc[c_t >= 2]

        df_acc = df_acc.drop_duplicates(subset=['code'], keep='first')

        # [수정] Timeframe 제거 확인
        cols = ["포착일", "market", "ticker", "name", "AI신호", "RSI", "score", "수익률", "Close", "단기", "중기", "장기", "수급(OBV)", "사이클(CCI)", "WillR", "ADX", "MFI", "일목(구름)", "파라볼릭", "Reasons"]
        avail = [c for c in cols if c in df_acc.columns]
        
        # [NEW] 실시간 포착 탭에도 클릭 시 상세 분석 기능 활성화
        event_acc = st.dataframe(
            df_acc[avail].style.format({
                "score": safe_fmt, "Close": safe_fmt_price, "수익률": safe_fmt_profit, "ADX": safe_fmt, "MFI": safe_fmt, "RSI": safe_fmt
            }).background_gradient(subset=["score"], cmap="Blues"),
            on_select="rerun", 
            selection_mode="single-row",
            use_container_width=True
        )
        
        if len(event_acc.selection.rows) > 0:
            sel_idx = event_acc.selection.rows[0]
            sel_row = df_acc.iloc[sel_idx]
            st.markdown("---")
            display_detailed_ai_report_v3(sel_row["ticker"], sel_row["market"], sel_row["name"], row_data=sel_row)

        st.caption(f"총 {len(df_acc)}개의 종목이 감시 기간 동안 포착되었습니다.")
    else:
        st.info("아직 자동 감시로 포착된 종목이 없습니다. 사이드바에서 `▶ 감시 시작`을 눌러주세요.")