import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from v import run_7_day_weather_prediction
from datetime import datetime

st.set_page_config(page_title="AI Weather Forecast", layout="wide")

# --- UI Layout ---
st.title("☁️ AI-Powered Weather Forecast")
st.markdown("Get AI-predicted hourly weather conditions for the next 7 days 🌍")

# Input
location = st.text_input("Enter a City (e.g., Ahmedabad, Junagadh, Seoul)", "Ahmedabad")

# Trigger prediction
if st.button("Predict Weather"):
    with st.spinner("Fetching forecast and predicting..."):
        result_df = run_7_day_weather_prediction(location)

    if result_df is None or result_df.empty:
        st.warning("No results available. Check if forecast labels match training data.")
    else:
        # Today’s forecast
        today = datetime.today().date()
        today_data = result_df[pd.to_datetime(result_df["time"]).dt.date == today]
        st.subheader(f"📍 Current Forecast for {location}")
        
        # Prepare today's data for current time
        today_data["datetime"] = pd.to_datetime(today_data["time"])
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        closest_idx = (today_data["datetime"] - now).abs().idxmin()
        current_row = today_data.loc[closest_idx]

        # Create 4 side-by-side columns for metrics 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡 Temperature", f"{current_row['temperature_2m']} °C")
        col2.metric("💧 Humidity", f"{current_row['relative_humidity_2m']}%")
        col3.metric("🌬 Wind Speed", f"{current_row['windspeed_10m']} m/s")
        col4.metric("🔮 Predicted", current_row['predicted_condition'])

        today_data["hrs."] = pd.to_datetime(today_data["time"]).dt.strftime('%H:%M')

        # Hourly Forecast Today
        st.markdown("### 🕒 Hourly Forecast Today")
        if not today_data.empty:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(pd.to_datetime(today_data["time"]), today_data["temperature_2m"], label="Temperature (°C)", marker='o')
            ax.set_ylabel("Temperature (°C)")
            ax.set_xlabel("Time")   
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            st.dataframe(today_data[["hrs.", "temperature_2m", "relative_humidity_2m", "windspeed_10m"]])
        else:
            st.info("No hourly data for today available.")

        # Weekly View
        st.markdown("### 📅 Weekly Summary")
        result_df["date"] = pd.to_datetime(result_df["time"]).dt.date
        result_df["hrs."] = pd.to_datetime(result_df["time"]).dt.strftime('%H:%M')
        weekly = result_df.groupby("date").agg({
            "temperature_2m": ["min", "max"],
            "relative_humidity_2m": "mean",
            "windspeed_10m": "mean",
            "predicted_condition": lambda x: x.value_counts().idxmax()
        }).reset_index()
        weekly.columns = ["Date", "Min Temp (°C)", "Max Temp (°C)", "Avg Humidity (%)", "Avg Wind (m/s)", "Most Likely Condition"]
        st.table(weekly)

        # Raw results
        with st.expander("🔍 See full hourly prediction"):
            full_df = result_df[["date", "hrs.", "temperature_2m", "relative_humidity_2m", "windspeed_10m", "predicted_condition"]]
            st.dataframe(full_df)
