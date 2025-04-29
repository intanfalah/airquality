import streamlit as st
import pandas as pd
import seaborn as sns
import requests
from io import StringIO
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./cleaned_data.csv")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["date"] = df["datetime"].dt.date
    return df

df = load_data()

# Sidebar filter
st.sidebar.header("Filter Data")
stations = df["station"].unique()
selected_stations = st.sidebar.multiselect("Pilih Stasiun", stations, default=stations)
date_range = st.sidebar.date_input("Rentang Waktu", [df["date"].min(), df["date"].max()])

# Filter data
filtered_df = df[(df["station"].isin(selected_stations)) &
                 (df["date"] >= date_range[0]) &
                 (df["date"] <= date_range[1])]

# 1️⃣ Tren PM2.5
st.subheader("📊 Tren Kualitas Udara per Stasiun (PM2.5)")
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(data=filtered_df, x="datetime", y="PM2.5", hue="station", ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Waktu")
plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
plt.title("Tren PM2.5 dari Waktu ke Waktu")
st.pyplot(fig)

# 2️⃣ Korelasi antar parameter
st.subheader("🔍 Korelasi Parameter Kualitas Udara")
cols = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
corr_matrix = df[cols].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
plt.title("Matriks Korelasi Antar Parameter")
st.pyplot(fig)

# 3️⃣ Boxplot antar stasiun
st.subheader("📌 Perbandingan Kualitas Udara Antarstasiun")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_df, x="station", y="PM2.5")
plt.xticks(rotation=45)
plt.xlabel("Stasiun")
plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
plt.title("Distribusi PM2.5 di Setiap Stasiun")
st.pyplot(fig)

# 4️⃣ Forecasting
st.subheader("📈 Forecasting PM2.5 dengan ARIMA")
selected_station = st.selectbox("Pilih Stasiun untuk Forecasting", stations)
station_df = df[df["station"] == selected_station].set_index("datetime")["PM2.5"]
station_daily = station_df.resample("D").mean().fillna(method="ffill")

if station_daily.dropna().shape[0] > 30:
    model = ARIMA(station_daily, order=(5, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(station_daily, label="Data Aktual")
    ax.plot(forecast, label="Forecast (30 Hari)", linestyle="--", color="red")
    ax.set_title(f"Forecasting PM2.5 - {selected_station}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Data tidak cukup untuk dilakukan prediksi.")