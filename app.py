import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, feedparser, urllib.parse
import yfinance as yf

# -------------------- Config & titre --------------------
st.set_page_config(page_title="Crypto Analyst – Saïd", layout="wide")
st.title("🧠 Crypto Analyst – Saïd")
st.caption("Analyse SR, Ichimoku, RSI, plan Entrée/TP/SL + alertes + actus. (Éducatif)")

# -------------------- Presets --------------------
PRESETS = {
    "XRP (H4)": {"symbol": "XRP","source": "auto","tf": "4h","history_days": 180,"limit": 500,
        "alerts": {"price_up": 3.10,"price_down": 2.40,"rsi_over_enabled": True,"rsi_over": 70,
                   "rsi_under_enabled": True,"rsi_under": 35,"sr_cross_enabled": True}},
    "SOL (H4)": {"symbol": "SOL","source": "auto","tf": "4h","history_days": 180,"limit": 500,
        "alerts": {"price_up": 220.0,"price_down": 160.0,"rsi_over_enabled": True,"rsi_over": 70,
                   "rsi_under_enabled": True,"rsi_under": 35,"sr_cross_enabled": True}},
}

# -------------------- Data loaders --------------------
def _to_df(klines):
    cols = ['Open time','Open','High','Low','Close','Volume',
            'Close time','Quote asset vol','Trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(klines, columns=cols)
    for c in ['Open','High','Low','Close','Volume']: df[c] = df[c].astype(float)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    return df.rename(columns={'Open time':'Date'})[['Date','Open','High','Low','Close','Volume']].set_index('Date')

def load_from_binance(symbol, interval="1h", limit=500):
    sym = symbol.upper().replace("-","")
    sym = sym if sym.endswith("USDT") else sym+"USDT"
    r = requests.get(
        f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={interval}&limit={int(limit)}",
        timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and 'code' in data: raise RuntimeError(str(data))
    return _to_df(data)

def load_from_mexc(symbol, interval="1h", limit=500):
    sym = symbol.upper().replace("-","")
    sym = sym if sym.endswith("USDT") else sym+"USDT"
    r = requests.get(
        f"https://api.mexc.com/api/v3/klines?symbol={sym}&interval={interval}&limit={int(limit)}",
        timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get('code') not in (None,'200'): raise RuntimeError(str(data))
    return _to_df(data)

def load_from_yahoo(symbol, interval="60m", period_days=180):
    ticker = symbol.upper()
    ticker = ticker if "-" in ticker else f"{ticker}-USD"
    data = yf.download(ticker, period=f"{period_days}d", interval=interval,
                       progress=False, auto_adjust=False)
    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance n'a renvoyé aucun historique.")
    data = data.rename(columns=str.title); data.index.name = "Date"
    return data[['Open','High','Low','Close','Volume']]

def load_ohlc(symbol, source, interval, limit=500, period_days=180):
    if source == "auto":
        for src in ("binance","mexc","yahoo"):
            try:
                if src == "binance": return load_from_binance(symbol, interval, limit), "binance"
                if src == "mexc":    return load_from_mexc(symbol, interval, limit), "mexc"
                iv = "60m" if interval.endswith("h") else ("1d" if interval.endswith("d") else "60m")
                return load_from_yahoo(symbol, iv, period_days), "yahoo"
            except Exception:
                continue
        raise RuntimeError("Aucune source n’a pu fournir des données.")
    if source == "binance": return load_from_binance(symbol, interval, limit), "binance"
    if source == "mexc":    return load_from_mexc(symbol, interval, limit), "mexc"
    if source == "yahoo":
        iv = "60m" if interval.endswith("h") else ("1d" if interval.endswith("d") else "60m")
        return load_from_yahoo(symbol, iv, period_days), "yahoo"
    raise ValueError("Source inconnue.")

# -------------------- Indicators & SR --------------------
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0,np.nan)
    return 100 - (100/(1+rs))

def ichimoku(df, conv=9, base=26, span_b_period=52, disp=26):
    high, low, close = df['High'], df['Low'], df['Close']
    tenkan = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
    kijun  = (high.rolling(base).max() + low.rolling(base).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(disp)
    span_b = ((high.rolling(span_b_period).max() + low.rolling(span_b_period).min()) / 2).shift(disp)
    chikou = close.shift(-disp)
    idx = df.index
    out = pd.DataFrame(index=idx)
    out['tenkan'] = tenkan.reindex(idx)
    out['kijun']  = kijun.reindex(idx)
    out['span_a'] = span_a.reindex(idx)
    out['span_b'] = span_b.reindex(idx)
    out['chikou'] = chikou.reindex(idx)
    return out
    
def swing_points(df, window=3):
    highs = df['High'].rolling(window*2+1, center=True).apply(lambda x: float(x[window]==x.max()))
    lows  = df['Low'].rolling(window*2+1, center=True).apply(lambda x: float(x[window]==x.min()))
    hi = highs.fillna(0).astype(bool); lo = lows.fillna(0).astype(bool)
    return df.loc[hi,'High'], df.loc[lo,'Low']

def cluster_levels(levels, px_tol=None):
    vals = levels.dropna().values
    if vals.size==0: return []
    med = np.median(vals); px_tol = px_tol or max(med*0.0025, 1e-12)
    vals = np.sort(vals); clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - np.mean(clusters[-1])) <= px_tol: clusters[-1].append(v)
        else: clusters.append([v])
    return sorted(float(np.mean(c)) for c in clusters)

def sr_bands_from_swings(df, window=3, px_tol=None):
    sh, sl = swing_points(df, window=window)
    return cluster_levels(sl, px_tol), cluster_levels(sh, px_tol)

def generate_plan(price, supports, resistances, kijun=None, rsi_val=None):
    below = [s for s in supports if s < price]
    entries = (below[-2:] if len(below)>=2 else
               ([below[-1], below[-1]*0.985] if len(below)==1 else [price*0.985, price*0.97]))
    sl = min(entries)*0.988
    above = [r for r in resistances if r > price]
    tps = ([price*1.02, price*1.04, price*1.06] if len(above)==0 else
           ([above[0], above[0]*1.02, above[0]*1.04] if len(above)==1 else above[:3]))
    notes=[]
    if kijun is not None: notes.append("Au-dessus de Kijun → haussier." if price>=kijun else "Sous Kijun → prudence.")
    if rsi_val is not None:
        notes.append("RSI ≥ 70: surachat." if rsi_val>=70 else ("RSI ≤ 35: zone d'accumulation." if rsi_val<=35 else "RSI neutre."))
    return {"entries":entries,"sl":sl,"tps":tps,"notes":notes}

# -------------------- State par défaut --------------------
defaults = {"sym":"XRP","src":"auto","tf_key":"4h","hist_days":180,"limit_bars":500,
            "current_preset":"(Aucun)","alert_price_up":None,"alert_price_down":None,
            "alert_rsi_over_enabled":False,"alert_rsi_over":70,"alert_rsi_under_enabled":False,"alert_rsi_under":35,
            "enable_sr_cross_alerts":True}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# -------------------- UI haut & sidebar --------------------
st.markdown("### 🔎 Choix rapide")
quick = st.radio("Sélection rapide", ["XRP","SOL","BTC","ETH","WLFI","LILPEPE","Autre"], horizontal=True, index=0)
if quick!="Autre" and st.session_state["sym"]!=quick:
    st.session_state["sym"]=quick; st.rerun()

with st.sidebar:
    st.header("Paramètres")
    preset_name = st.selectbox("Preset", ["(Aucun)"]+list(PRESETS.keys()))
    if preset_name!=st.session_state.get("current_preset","(Aucun)"):
        st.session_state["current_preset"]=preset_name
        if preset_name!="(Aucun)":
            p=PRESETS[preset_name]
            st.session_state["sym"]=p["symbol"]; st.session_state["src"]=p["source"]
            st.session_state["tf_key"]=p["tf"]; st.session_state["hist_days"]=p["history_days"]
            st.session_state["limit_bars"]=p["limit"]
            al=p["alerts"]
            st.session_state["alert_price_up"]=float(al["price_up"])
            st.session_state["alert_price_down"]=float(al["price_down"])
            st.session_state["alert_rsi_over_enabled"]=bool(al["rsi_over_enabled"])
            st.session_state["alert_rsi_over"]=int(al["rsi_over"])
            st.session_state["alert_rsi_under_enabled"]=bool(al["rsi_under_enabled"])
            st.session_state["alert_rsi_under"]=int(al["rsi_under"])
            st.session_state["enable_sr_cross_alerts"]=bool(al["sr_cross_enabled"])
        st.rerun()

    symbol = st.text_input("Symbole / Ticker", value=st.session_state["sym"], key="sym")
    source = st.selectbox("Source de données", ["auto","binance","mexc","yahoo"],
                          index=["auto","binance","mexc","yahoo"].index(st.session_state["src"]), key="src")
    tf = st.selectbox("Timeframe", ["1h","4h","1d"],
                      index=["1h","4h","1d"].index(st.session_state["tf_key"]), key="tf_key")
    history_days = st.slider("Historique (jours) – Yahoo seulement", 30, 720, st.session_state["hist_days"], key="hist_days")
    limit = st.slider("Bougies (Binance/MEXC)", 200, 1000, st.session_state["limit_bars"], step=50, key="limit_bars")
    st.divider()
    show_ichimoku = st.checkbox("Ichimoku", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)

# -------------------- Chargement des données --------------------
with st.spinner("Chargement des cours…"):
    try:
        df, used_source = load_ohlc(symbol, source, interval=tf, limit=limit, period_days=history_days)
    except Exception as e:
        st.error(f"Impossible de charger les données : {e}")
        st.stop()

if df is None or df.empty:
    st.warning("Aucune donnée pour ce symbole/timeframe."); st.stop()

ichi = ichimoku(df) if show_ichimoku else None
rsi_series = rsi(df['Close'], 14) if show_rsi else None
s_levels, r_levels = sr_bands_from_swings(df, window=3, px_tol=None)

latest_price = float(df['Close'].iloc[-1])
prev_close = float(df['Close'].iloc[-2]) if len(df)>=2 else latest_price
last_rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not np.isnan(rsi_series.iloc[-1]) else None
last_kijun = float(ichi['kijun'].iloc[-1]) if ichi is not None and not np.isnan(ichi['kijun'].iloc[-1]) else None
plan = generate_plan(latest_price, s_levels, r_levels, kijun=last_kijun, rsi_val=last_rsi)

# -------------------- Graphique & panneau --------------------
col1, col2 = st.columns([3,2])

with col1:
    st.subheader(f"Graphique {symbol.upper()} ({used_source}) – {tf}")
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    for s in s_levels[-8:]: fig.add_hline(y=s, line_width=1, opacity=0.45)
    for r in r_levels[:8]:  fig.add_hline(y=r, line_width=1, opacity=0.55)
    if show_ichimoku and ichi is not None:
        fig.add_trace(go.Scatter(x=df.index, y=ichi['tenkan'], name="Tenkan", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=ichi['kijun'],  name="Kijun",  mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=ichi['span_a'], name="Span A", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=ichi['span_b'], name="Span B", mode="lines"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=560, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    if show_rsi and rsi_series is not None:
        st.subheader("RSI (14)")
        st.line_chart(rsi_series)

with col2:
    st.subheader("📍 Supports / Résistances")
    st.markdown("**Supports** (derniers):")
    st.write([round(x,6) for x in s_levels[-6:]] or "—")
    st.markdown("**Résistances** (prochaines):")
    st.write([round(x,6) for x in r_levels[:6]] or "—")
    st.subheader("🎯 Plan (éducatif)")
    st.markdown(f"**Dernier prix**: {latest_price:,.6f}")
    st.markdown("**Entrées DCA**:"); st.write([round(x,6) for x in plan['entries']])
    st.markdown("**Stop-loss**:");    st.write(round(plan['sl'],6))
    st.markdown("**Take-profits**:"); st.write([round(x,6) for x in plan['tps']])
    if plan['notes']: st.info(" — ".join(plan['notes']))

# -------------------- Alertes --------------------
st.divider(); st.subheader("🔔 Alertes personnalisées")
cA, cB, cC = st.columns(3)
with cA:
    st.markdown("**Prix ↑ au-dessus de**")
    if st.session_state["alert_price_up"] is None and r_levels:
        st.session_state["alert_price_up"] = round(float(next((r for r in r_levels if r>latest_price), latest_price*1.02)),6)
    st.session_state["alert_price_up"] = st.number_input("Seuil haussier", value=float(st.session_state["alert_price_up"]), format="%.6f")
with cB:
    st.markdown("**Prix ↓ en dessous de**")
    if st.session_state["alert_price_down"] is None and s_levels:
        st.session_state["alert_price_down"] = round(float(next((s for s in reversed(s_levels) if s<latest_price), latest_price*0.98)),6)
    st.session_state["alert_price_down"] = st.number_input("Seuil baissier", value=float(st.session_state["alert_price_down"]), format="%.6f")
with cC:
    st.markdown("**RSI**")
    st.session_state["alert_rsi_over_enabled"]  = st.checkbox("Alerte RSI ≥", value=st.session_state["alert_rsi_over_enabled"])
    st.session_state["alert_rsi_over"]          = st.slider("Seuil RSI ≥", 50, 90, int(st.session_state["alert_rsi_over"]), disabled=not st.session_state["alert_rsi_over_enabled"])
    st.session_state["alert_rsi_under_enabled"] = st.checkbox("Alerte RSI ≤", value=st.session_state["alert_rsi_under_enabled"])
    st.session_state["alert_rsi_under"]         = st.slider("Seuil RSI ≤", 10, 50, int(st.session_state["alert_rsi_under"]), disabled=not st.session_state["alert_rsi_under_enabled"])

st.toggle("Cassure SR auto", value=st.session_state["enable_sr_cross_alerts"], key="enable_sr_cross_alerts")

triggered=[]
up=st.session_state["alert_price_up"]; down=st.session_state["alert_price_down"]
if up   and latest_price>=up:   triggered.append(f"Prix a dépassé {up:.6f} ↑")
if down and latest_price<=down: triggered.append(f"Prix est passé sous {down:.6f} ↓")
if rsi_series is not None:
    r = float(rsi_series.iloc[-1])
    if st.session_state["alert_rsi_over_enabled"]  and r>=float(st.session_state["alert_rsi_over"]):  triggered.append(f"RSI ≥ {int(st.session_state['alert_rsi_over'])} (actuel: {r:.1f})")
    if st.session_state["alert_rsi_under_enabled"] and r<=float(st.session_state["alert_rsi_under"]): triggered.append(f"RSI ≤ {int(st.session_state['alert_rsi_under'])} (actuel: {r:.1f})")
if st.session_state["enable_sr_cross_alerts"]:
    cu=[r for r in r_levels[:5] if prev_close<r<=latest_price]
    cd=[s for s in s_levels[-5:] if prev_close>s>=latest_price]
    if cu: triggered.append("Cassure de résistance: " + ", ".join(f"{x:.6f}" for x in cu))
    if cd: triggered.append("Cassure de support: " + ", ".join(f"{x:.6f}" for x in cd))
st.success("**Alertes déclenchées :**\n\n- " + "\n- ".join(triggered)) if triggered else st.info("Aucune alerte déclenchée sur la dernière bougie.")

# -------------------- Actus --------------------
st.divider(); st.subheader("🗞️ Actus récentes")
def fetch_news(query, lang="fr", region="FR", max_items=8):
    q = urllib.parse.quote_plus(f"{query} crypto OR token OR coin")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={region}&ceid={region}:{lang}")
    items=[]
    for e in feed.entries[:max_items]:
        items.append({"title": getattr(e,"title","(titre)"), "link": getattr(e,"link","#"), "published": getattr(e,"published","")})
    return items
query = st.text_input("Mot-clé actus", value=f"{symbol}")
if st.button("Rechercher des actualités"):
    items = fetch_news(query)
    if not items: st.warning("Aucune actualité trouvée.")
    for it in items: st.markdown(f"- [{it['title']}]({it['link']}) — {it['published']}")

st.caption("© 2025 – Saïd — Ceci n'est pas un conseil financier.")
