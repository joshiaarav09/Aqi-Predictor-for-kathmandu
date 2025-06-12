import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Use seaborn theme
sns.set_theme(style="whitegrid")

# Import all processing functions and config from feature_engineering.py
from feature_engineering import (
    fetch_purpleair_120h,
    fetch_weather_matching,
    fetch_future_weather_48h,
    add_full_features,
    SENSOR_IDS,
    SENSOR_LOCATIONS
)

def pm25_to_aqi(pm):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    aqi_values = []
    for value in pm:
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= value <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
                aqi_values.append(round(aqi))
                break
        else:
            aqi_values.append(500)
    return aqi_values

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#a8e05f", "Air quality is good. Safe to go outside."
    elif aqi <= 100:
        return "Moderate", "#fdd74b", "Air quality is acceptable. Sensitive individuals should be cautious."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#fe9b57", "Consider reducing prolonged outdoor exposure."
    elif aqi <= 200:
        return "Unhealthy", "#fe6a69", "Avoid outdoor activity. Wear a mask."
    elif aqi <= 300:
        return "Very Unhealthy", "#a97abc", "Stay indoors. Use air purifiers."
    else:
        return "Hazardous", "#a87383", "Stay indoors. Health alert!"

# Load model and scalers
model = load_model("pm25_model_lstm.h5", compile=False)
scaler_past = joblib.load("scaler_past.pkl")
scaler_future = joblib.load("scaler_future.pkl")
scaler_y = joblib.load("scaler_y.pkl")

with open("scaler_past_features.json") as f:
    past_feats = json.load(f)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        html, body, [data-testid="stApp"] {
            background-color: white !important;
            color: black !important;
        }
        [data-testid="stMetric"] {
            background-color: white !important;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        .stExpander, .stDataFrame, .element-container, .block-container {
            background-color: white !important;
        }
        div[data-baseweb="select"] > div {
            background-color: white !important;
            color: black !important;
            border: 1px solid #ccc;
        }
        button[kind="secondary"] {
            background-color: white !important;
            color: black !important;
            border: 1px solid #ccc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌫️ Kathmandu PM2.5 Forecast (Next 48 Hours)")

sensor = st.selectbox("Select Monitoring Location", list(SENSOR_IDS.keys()))

if st.button("📊 Generate Forecast"):
    with st.spinner("Fetching sensor and weather data, preparing features, and predicting..."):
        try:
            df_pm = fetch_purpleair_120h(sensor)
            pm_end_time = df_pm["timestamp"].max()
            df_weather = fetch_weather_matching(pm_end_time, sensor)

            start_time = pm_end_time - pd.Timedelta(hours=119)
            df_pm = df_pm[df_pm["timestamp"] >= start_time].reset_index(drop=True)
            df_weather = df_weather[df_weather["timestamp"] >= start_time].reset_index(drop=True)

            df_weather = df_weather.rename(columns={
                "temperature_2m": "temp", "relative_humidity_2m": "rhum", "pressure_msl": "pres",
                "wind_speed_10m": "wspd", "wind_direction_10m": "wdir",
                "precipitation": "prcp", "dew_point_2m": "dwpt"
            })
            df_merged = pd.merge(df_pm, df_weather, on="timestamp", how="inner")
            df_merged = add_full_features(df_merged).dropna().reset_index(drop=True)

            X_past = df_merged[past_feats].iloc[-76:]
            X_past_scaled = scaler_past.transform(X_past)
            X_past_input = X_past_scaled.reshape(1, 76, -1)

            future_df = fetch_future_weather_48h(sensor)
            future_df = future_df.rename(columns={
                "temperature_2m": "temp", "relative_humidity_2m": "rhum", "pressure_msl": "pres",
                "wind_speed_10m": "wspd", "wind_direction_10m": "wdir",
                "precipitation": "prcp", "dew_point_2m": "dwpt"
            })
            future_df = add_full_features(future_df)

            fut_base = ['temp', 'rhum', 'pres', 'wspd', 'wdir', 'prcp', 'dwpt']
            flat_fut_data = []
            for f in fut_base:
                flat_fut_data.extend(future_df[f].iloc[:24].tolist())
            future_feats = [f"{f}_fut{i+1}" for i in range(24) for f in fut_base]
            X_future = pd.DataFrame([flat_fut_data], columns=future_feats)
            X_future_scaled = scaler_future.transform(X_future)
            X_future_input = X_future_scaled.reshape(1, -1)

            y_pred_scaled = model.predict([X_past_input, X_future_input])
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            day1 = y_pred[:24]
            day2 = y_pred[24:]

            aqi_day1 = pm25_to_aqi(day1)
            aqi_day2 = pm25_to_aqi(day2)

            avg_aqi1 = round(np.mean(aqi_day1))
            avg_aqi2 = round(np.mean(aqi_day2))

            cat1, color1, msg1 = get_aqi_category(avg_aqi1)
            cat2, color2, msg2 = get_aqi_category(avg_aqi2)

            st.success("✅ Prediction Complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div style='background-color:{color1}; padding:1.5rem; border-radius:10px;'>
                    <h3 style='text-align:center;'>Tomorrow</h3>
                    <h1 style='text-align:center;'>{avg_aqi1} AQI</h1>
                    <p style='text-align:center;'>{cat1} – {msg1}</p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📈 See Day 1 Detailed Chart"):
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    sns.lineplot(x=range(1, 25), y=day1, marker="o", ax=ax1)
                    ax1.set_title("PM2.5 Forecast – Tomorrow")
                    ax1.set_xlabel("Hour")
                    ax1.set_ylabel("PM2.5 (µg/m³)")
                    st.pyplot(fig1)

            with col2:
                st.markdown(f"""
                <div style='background-color:{color2}; padding:1.5rem; border-radius:10px;'>
                    <h3 style='text-align:center;'>Day After Tomorrow</h3>
                    <h1 style='text-align:center;'>{avg_aqi2} AQI</h1>
                    <p style='text-align:center;'>{cat2} – {msg2}</p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📉 See Day 2 Detailed Chart"):
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    sns.lineplot(x=range(25, 49), y=day2, marker="o", ax=ax2)
                    ax2.set_title("PM2.5 Forecast – Day After Tomorrow")
                    ax2.set_xlabel("Hour")
                    ax2.set_ylabel("PM2.5 (µg/m³)")
                    st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
