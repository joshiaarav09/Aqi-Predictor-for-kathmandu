import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json



# --- API-KEY----
PURPLEAIR_API_KEY = 'C74FD585-44A3-11F0-81BE-42010A80001F'

SENSOR_IDS = {
    "Manbhawan": 50861,
    "Himalayan College of Engineering": 194785,
    "Rabi Bhawan": 215389,
    "Department of Physics PMC": 194797,
    "Indrayani Temple": 108062
}

SENSOR_LOCATIONS = {
    "Manbhawan": {"lat": 27.6764, "lon": 85.3251},
    "Himalayan College of Engineering": {"lat": 27.6889, "lon": 85.3314},
    "Rabi Bhawan": {"lat": 27.7034, "lon": 85.2903},
    "Department of Physics PMC": {"lat": 27.6812, "lon": 85.3097},
    "Indrayani Temple": {"lat": 27.7103, "lon": 85.2817}
}

# Holiday Dates
is_holiday82 = pd.to_datetime([
    "2025-08-09","2025-08-10","2025-08-16","2025-08-31","2025-09-06",
    "2025-09-15","2025-09-19","2025-09-22","2025-09-29","2025-09-30",
    "2025-10-01","2025-10-02","2025-10-03","2025-10-04","2025-10-20",
    "2025-10-21","2025-10-22","2025-10-23","2025-10-24","2025-10-27",
    "2025-11-05","2025-11-11","2025-12-03","2025-12-04","2025-12-25",
    "2025-12-30","2026-01-11","2026-01-15","2026-01-19","2026-01-23",
    "2026-01-30","2026-02-15","2026-02-18","2026-02-19","2026-03-02",
    "2026-03-03","2026-03-08","2026-03-18","2026-03-27"
])

# --- Generating features ---
def add_full_features(df):
    df = df.copy()

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([6, 7, 8, 9, 17, 18, 19, 20]).astype(int)
    df["is_holiday"] = df["timestamp"].dt.normalize().isin(is_holiday82).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Interaction features
    df["temp_rhum_interact"] = df["temp"] * df["rhum"]
    df["wind_energy"] = 0.5 * 1.225 * (df["wspd"] ** 3)

    # Rolling statistics
    if "pm2.5_atm" in df.columns:
        df["pm2.5_rolling_mean_6h"] = df["pm2.5_atm"].rolling(6, min_periods=1).mean()
        df["pm2.5_rolling_std_6h"] = df["pm2.5_atm"].rolling(6, min_periods=1).std().fillna(0)

        df["pm2.5_yesterday"] = df["pm2.5_atm"].shift(24)
        df["pm2.5_max_12h"] = df["pm2.5_atm"].rolling(window=12, min_periods=1).max()
        df["pm2.5_min_12h"] = df["pm2.5_atm"].rolling(window=12, min_periods=1).min()

        # Lag features (1 to 24)
        for i in range(1, 25):
            df[f"pm2.5_lag{i}"] = df["pm2.5_atm"].shift(i)

    # One-hot encoding for month (month_1 to month_12)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)

    return df



# --- Fetching data from API---
def fetch_purpleair_120h(sensor_name):
    sensor_id = SENSOR_IDS[sensor_name]
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=120)
    url = f"https://api.purpleair.com/v1/sensors/{sensor_id}/history"
    params = {
        "start_timestamp": int(start_time.timestamp()),
        "end_timestamp": int(end_time.timestamp()),
        "average": 60,
        "fields": "pm2.5_atm,pm2.5_cf_1,pm1.0_atm,pm10.0_atm"
    }
    headers = {"X-API-Key": PURPLEAIR_API_KEY}
    resp = requests.get(url, params=params, headers=headers)
    data = resp.json()
    df = pd.DataFrame(data["data"], columns=data["fields"])
    df["timestamp"] = pd.to_datetime(df["time_stamp"], unit="s")
    return df.sort_values("timestamp").reset_index(drop=True)

def fetch_weather_matching(pm_end_time, location_name):
    coords = SENSOR_LOCATIONS[location_name]
    start_time = pm_end_time - timedelta(hours=119)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,precipitation,wind_speed_10m,wind_direction_10m",
        "start_date": start_time.date().isoformat(),
        "end_date": pm_end_time.date().isoformat(),
        "timezone": "Asia/Kathmandu"
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df[(df["timestamp"] >= start_time) & (df["timestamp"] <= pm_end_time)].reset_index(drop=True)

def fetch_future_weather_48h(location_name):
    coords = SENSOR_LOCATIONS[location_name]
    now_nepal = datetime.utcnow() + timedelta(hours=5, minutes=45)
    start_time = now_nepal.replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=48)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,precipitation,wind_speed_10m,wind_direction_10m",
        "start_date": start_time.date().isoformat(),
        "end_date": end_time.date().isoformat(),
        "timezone": "Asia/Kathmandu"
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df[(df["timestamp"] >= start_time) & (df["timestamp"] < end_time)].reset_index(drop=True)

