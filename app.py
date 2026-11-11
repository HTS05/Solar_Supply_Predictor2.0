"""Solar Pulse dashboard for weather-based solar farm planning."""

import json
from datetime import date, timedelta

import requests
import streamlit as st

from src.weatherapi import get_forecast_weather
from src.solar_predictor import (
    METRICS_FILE,
    get_feature_range_warnings,
    load_data,
    load_saved_model,
    predict_ghi,
)


st.set_page_config(page_title="Solar Pulse | Farm Planner", page_icon="☼", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#f5f7f2; --muted:#93a0a5; --canvas:#0b1117; --panel:#111a22; --line:rgba(194,214,211,.12); --mint:#9ee7ca; --lime:#d8f36a; --orange:#ffbd73; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:var(--canvas); color:var(--ink); }
    .block-container { max-width:1420px; padding:2rem 3rem 4rem; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stToolbar"] { visibility:hidden; }
    [data-testid="stSidebar"] { background:#0f171e; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding:1.5rem 1.2rem; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color:var(--muted); }
    h1, h2, h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.04em; }
    h1 { font-size:3.1rem !important; line-height:1 !important; }
    .stButton > button { border-radius:12px; border:1px solid rgba(158,231,202,.35); background:var(--mint); color:#08120f !important; font-weight:700; min-height:2.7rem; }
    .stButton > button p { color:#08120f !important; }
    .stButton > button:hover { border-color:var(--lime); background:var(--lime); color:#08120f !important; }
    .brand { display:flex; align-items:center; gap:.65rem; margin-bottom:2rem; }
    .brand-mark { width:2.2rem; height:2.2rem; display:grid; place-items:center; border-radius:10px; background:var(--lime); color:#172117; font-size:1.4rem; font-weight:700; }
    .brand-name { color:var(--ink); font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700; letter-spacing:.06em; }
    .brand-sub { color:var(--muted); font-size:.68rem; letter-spacing:.16em; margin-top:.1rem; }
    .hero { background:radial-gradient(circle at 82% 22%,rgba(158,231,202,.18),transparent 26%),linear-gradient(135deg,#15252d 0%,#111a22 62%,#182e2a 100%); border:1px solid rgba(158,231,202,.18); border-radius:24px; padding:2.5rem 2.7rem 2.4rem; position:relative; overflow:hidden; }
    .hero:after { content:''; position:absolute; width:280px; height:280px; border:1px solid rgba(216,243,106,.2); border-radius:50%; right:-90px; top:-140px; box-shadow:0 0 0 28px rgba(216,243,106,.03),0 0 0 58px rgba(216,243,106,.025); }
    .eyebrow { color:var(--lime); font-size:.72rem; font-weight:700; letter-spacing:.2em; text-transform:uppercase; margin-bottom:1rem; }
    .hero h1 { color:var(--ink); margin:0; max-width:700px; }
    .hero-copy { color:#b8c5c6; font-size:1rem; line-height:1.7; margin:1.15rem 0 0; max-width:700px; }
    .pill { display:inline-block; border:1px solid rgba(194,214,211,.18); border-radius:999px; color:#c5d0cf; font-size:.74rem; padding:.45rem .75rem; margin:.7rem .4rem 0 0; background:rgba(5,10,14,.18); }
    .pill strong { color:var(--mint); }
    .section-kicker { color:var(--muted); font-size:.7rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; margin:2rem 0 .8rem; }
    .kpi { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.25rem; min-height:112px; }
    .kpi-label { color:var(--muted); font-size:.76rem; letter-spacing:.04em; }
    .kpi-value { color:var(--ink); font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:600; margin-top:.4rem; }
    .kpi-value.mint { color:var(--mint); } .kpi-value.lime { color:var(--lime); }
    .kpi-note { color:#738187; font-size:.72rem; margin-top:.25rem; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.35rem 1.45rem; min-height:240px; }
    .panel-title { color:var(--ink); font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:600; }
    .panel-subtitle { color:var(--muted); font-size:.78rem; margin:.35rem 0 1.1rem; }
    .result-number { color:var(--mint); font-family:'Space Grotesk',sans-serif; font-size:3.2rem; font-weight:600; letter-spacing:-.07em; line-height:1; margin:1rem 0 .35rem; }
    .result-unit { color:var(--muted); font-size:.82rem; }
    .signal-row { align-items:center; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; padding:.65rem 0; }
    .signal-row:last-child { border-bottom:0; }
    .signal-name { color:var(--muted); font-size:.8rem; } .signal-value { color:var(--ink); font-size:.86rem; font-weight:600; }
    .footer-note { color:#66747a; font-size:.72rem; margin-top:2.5rem; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return load_saved_model()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else {}


def metric_card(label, value, note, tone=""):
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value {tone}'>{value}</div><div class='kpi-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def signal_rows(values):
    labels = [
        ("Temperature", "Temperature", "°C"),
        ("Humidity", "Humidity", "%"),
        ("Cloud coverage", "Cloud_Coverage", "%"),
        ("Wind speed", "Wind_Speed", "m/s"),
        ("Precipitation", "Precipitation", "mm/day"),
        ("Pressure", "Pressure", "kPa"),
    ]
    return "".join(
        f"<div class='signal-row'><span class='signal-name'>{label}</span><span class='signal-value'>{values[key]:.2f} {unit}</span></div>"
        for label, key, unit in labels
    )


data = get_data()
model = get_model()
metrics = get_metrics()
cities = sorted(data["City"].unique())
min_date = data["Date"].min().date()
today = date.today()
max_date = today + timedelta(days=16)


with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='brand-mark'>☼</div><div><div class='brand-name'>SOLAR PULSE</div><div class='brand-sub'>SUPPLY FORECAST</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Forecast setup")
    city = st.selectbox("Solar farm location", cities)
    selected_date = st.date_input("Forecast date", value=today, min_value=min_date, max_value=max_date)
    st.caption("Weather values stay empty until you request them from the API.")
    fetch_button = st.button("Get weather & estimate", type="primary", width="stretch")
    st.markdown("### Farm details")
    daily_need = st.number_input("Electricity needed (kWh/day)", min_value=0.0, value=1000.0, step=100.0)
    panel_area = st.number_input("Panel area (m²)", min_value=0.0, value=1000.0, step=100.0)
    panel_efficiency = st.slider("Panel efficiency (%)", 5.0, 35.0, 20.0, .5)
    system_performance = st.slider("System performance (%)", 50.0, 100.0, 80.0, 1.0)


request_key = (city, selected_date.isoformat())
if st.session_state.get("weather_key") != request_key:
    st.session_state.weather_ready = False

if fetch_button:
    try:
        with st.spinner("Requesting weather data..."):
            st.session_state.weather_values = get_forecast_weather(city, selected_date)
        st.session_state.weather_key = request_key
        st.session_state.weather_ready = True
    except (RuntimeError, ValueError, OSError, requests.RequestException) as error:
        st.session_state.weather_ready = False
        st.error(f"Weather API request failed: {error}")


st.markdown(
    f"<div class='hero'><div class='eyebrow'>Solar farm planning / {selected_date.strftime('%d %b %Y')}</div>"
    "<h1>Know your farm's<br>solar potential.</h1>"
    "<p class='hero-copy'>Choose a location and date, fetch the weather signal, then compare the predicted solar generation with the farm's daily electricity requirement.</p>"
    f"<span class='pill'><strong>LOCATION</strong> {city}</span><span class='pill'><strong>MODEL</strong> Random Forest</span><span class='pill'><strong>INPUT</strong> API weather + farm details</span></div>",
    unsafe_allow_html=True,
)

if not st.session_state.get("weather_ready"):
    st.markdown("<div class='section-kicker'>Waiting for weather signal</div>", unsafe_allow_html=True)
    st.info("Select the farm location and date, then click “Get weather & estimate”. The weather fields will be supplied by the API; you do not enter them manually.")
    st.stop()


weather = st.session_state.weather_values
input_values = {**weather, "Date": selected_date, "City": city}
prediction = predict_ghi(model, input_values)
prediction_error = float(metrics.get("mean_absolute_error", 0.47))
prediction_low = max(0.0, prediction - prediction_error)
prediction_high = prediction + prediction_error
energy_factor = panel_area * (panel_efficiency / 100) * (system_performance / 100)
generation_low = prediction_low * energy_factor
generation_high = prediction_high * energy_factor
fulfilled_low = min(generation_low, daily_need)
fulfilled_high = min(generation_high, daily_need)
remaining_low = max(daily_need - generation_high, 0.0)
remaining_high = max(daily_need - generation_low, 0.0)
fulfillment_low = fulfilled_low / daily_need * 100 if daily_need else 0.0
fulfillment_high = fulfilled_high / daily_need * 100 if daily_need else 0.0
warnings = get_feature_range_warnings(data, input_values)
source = "Open-Meteo forecast" if selected_date >= today else "NASA POWER historical data"

st.markdown("<div class='section-kicker'>Weather signal received</div>", unsafe_allow_html=True)
weather_col, result_col = st.columns([1, 1.2], gap="large")
with weather_col:
    st.markdown(
        f"<div class='panel'><div class='panel-title'>API weather profile</div><div class='panel-subtitle'>{source} · values used by the model</div>{signal_rows(weather)}</div>",
        unsafe_allow_html=True,
    )
with result_col:
    st.markdown(
        f"<div class='panel'><div class='panel-title'>Predicted solar signal</div><div class='panel-subtitle'>Expected daily irradiance range for {city}</div><div class='result-number'>{prediction_low:.2f}–{prediction_high:.2f}</div><div class='result-unit'>kWh/m²/day predicted GHI · based on model MAE</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-kicker'>Farm coverage result</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("EST. GENERATION", f"{generation_low:,.1f}–{generation_high:,.1f}", "kWh/day expected range", "mint")
with col2:
    metric_card("ELECTRICITY NEED", f"{daily_need:,.1f}", "kWh/day", "lime")
with col3:
    metric_card("FULFILLED", f"{fulfillment_low:.1f}–{fulfillment_high:.1f}%", "expected coverage range", "mint")
with col4:
    metric_card("ENERGY LEFT", f"{remaining_low:,.1f}–{remaining_high:,.1f}", "kWh/day expected range", "lime" if remaining_high == 0 else "")

if remaining_low == 0:
    st.success(f"The farm is expected to cover the requirement, with a possible surplus up to {max(generation_high - daily_need, 0.0):,.1f} kWh/day.")
else:
    st.warning(f"The farm may need between {remaining_low:,.1f} and {remaining_high:,.1f} kWh/day from the grid or battery.")
if warnings:
    st.caption("Model input note: " + " · ".join(warnings))

with st.expander("How this result is calculated"):
    st.write("Predicted GHI × panel area × panel efficiency × system performance = estimated solar energy.")
    st.write(f"Model validation R²: {metrics.get('r2_score', '—')} · Data: {len(data):,} balanced historical records.")

st.markdown("<div class='footer-note'>SOLAR PULSE · Weather signal → GHI estimate → farm coverage result</div>", unsafe_allow_html=True)
