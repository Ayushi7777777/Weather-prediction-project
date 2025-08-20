import pandas as pd
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# --- Weather Condition Map ---
condition_map = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow fall", 73: "Moderate snow fall",
    75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm w/ slight hail",
    99: "Thunderstorm w/ heavy hail",
}

# --- Utility Functions ---

def get_lat_lon(location: str):
    geolocator = Nominatim(user_agent="weather_ml_app")
    loc = geolocator.geocode(location)
    return loc.latitude, loc.longitude

def fetch_weather_data(lat, lon, start_date, end_date, historical=True):
    base_url = "https://archive-api.open-meteo.com/v1/era5" if historical else "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,relative_humidity_2m,windspeed_10m,weathercode,"
            "cloudcover,dewpoint_2m,precipitation,surface_pressure"
        ),
        "timezone": "auto"
    }
    response = requests.get(base_url, params=params)
    return response.json().get("hourly", {})

def prepare_dataframe(data, is_train=True):
    df = pd.DataFrame(data)
    for col in ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "weathercode",
                "cloudcover", "dewpoint_2m", "precipitation", "surface_pressure"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["time"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    if is_train:
        df.dropna(inplace=True)
        df["weathercode"] = df["weathercode"].astype(int)
    else:
        df.dropna(subset=[
            "temperature_2m", "relative_humidity_2m", "windspeed_10m",
            "cloudcover", "dewpoint_2m", "precipitation", "surface_pressure"
        ], inplace=True)

    return df

# --- Benchmarking Multiple Models ---

def train_and_select_best_model(df_train):
    from sklearn.preprocessing import LabelEncoder

    X = df_train[[
        "temperature_2m", "relative_humidity_2m", "windspeed_10m",
        "cloudcover", "dewpoint_2m", "precipitation", "surface_pressure",
        "hour", "day_of_week"
    ]]

    # Encode y labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_train["weathercode"])
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_encoded)

    models = {
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "KNeighbors": KNeighborsClassifier()
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X, y_encoded, sample_weight=weights)
        else:
            model.fit(X, y_encoded)
        preds = model.predict(X)
        score = f1_score(y_encoded, preds, average="weighted")
        # print(f"{name} - F1 Score: {score:.4f}")
        if score > best_score:
            best_model = model
            best_score = score
            best_name = name

    # print(f"\n Best model: {best_name} (F1 Score = {best_score:.4f})")
    return best_model, le

# --- Final Evaluation ---

def predict_and_evaluate(model, df_test, le):
    X_test = df_test[[
        "temperature_2m", "relative_humidity_2m", "windspeed_10m",
        "cloudcover", "dewpoint_2m", "precipitation", "surface_pressure",
        "hour", "day_of_week"
    ]]

    y_true = df_test["weathercode"].values

    # Only keep samples with labels seen during training
    seen_labels = set(le.classes_)
    mask = df_test["weathercode"].isin(seen_labels)
    df_test = df_test[mask]
    X_test = X_test[mask]
    y_true = df_test["weathercode"].values

    if len(df_test) == 0:
        # print("No overlapping labels between train and test. Evaluation skipped.")
        return pd.DataFrame()

    y_true_encoded = le.transform(y_true)
    y_pred_encoded = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)

    df_test["predicted_weathercode"] = y_pred
    df_test["predicted_condition"] = df_test["predicted_weathercode"].map(condition_map)
    df_test["actual_condition"] = df_test["weathercode"].map(condition_map)

    # print("\n Classification Report:")
    # print(classification_report(y_true, y_pred))

    return df_test[[
        "time", "temperature_2m", "relative_humidity_2m", "windspeed_10m",
        "weathercode", "actual_condition", "predicted_weathercode", "predicted_condition"
    ]]

# --- Runner ---

def run_7_day_weather_prediction(location):
    # print(f"\n Location: {location}")
    lat, lon = get_lat_lon(location)

    hist_start = "2021-01-01"
    hist_end = "2025-01-01"
    hist_data = fetch_weather_data(lat, lon, hist_start, hist_end, historical=True)
    df_train = prepare_dataframe(hist_data, is_train=True)

    model, le = train_and_select_best_model(df_train)

    forecast_start = datetime.today().strftime('%Y-%m-%d')
    forecast_end = (datetime.today() + timedelta(days=6)).strftime('%Y-%m-%d')
    forecast_data = fetch_weather_data(lat, lon, forecast_start, forecast_end, historical=False)
    df_test = prepare_dataframe(forecast_data, is_train=False)
    df_test = df_test.dropna(subset=["weathercode"])
    df_test["weathercode"] = df_test["weathercode"].astype(int)

    results = predict_and_evaluate(model, df_test, le)
    return results

# Run it
# run_7_day_weather_prediction("Dubai")