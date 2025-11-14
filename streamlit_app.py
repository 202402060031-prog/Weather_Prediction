
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "model/temp_max_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Seattle Weather — Temp Max Predictor")
st.markdown("Upload a dataset or enter features manually to predict **temp_max** (°C).")

uploaded = st.file_uploader("Upload a CSV (optional). If none uploaded, sample data will be used.", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Uploaded data preview:")
    st.dataframe(df.head())
else:
    st.write("Using sample rows from packaged dataset.")
    df = pd.read_csv("data/sample_input.csv")
    st.dataframe(df.head())

st.sidebar.header("Manual input (single prediction)")
precipitation = st.sidebar.number_input("Precipitation (mm)", value=float(df['precipitation'].iloc[0]))
temp_min = st.sidebar.number_input("Temp Min (°C)", value=float(df['temp_min'].iloc[0]))
wind = st.sidebar.number_input("Wind (km/h)", value=float(df['wind'].iloc[0]))
date_str = st.sidebar.text_input("Date (YYYY-MM-DD)", value=str(df['date'].iloc[0]))
weather = st.sidebar.selectbox("Weather", options=sorted(df['weather'].unique().astype(str)), index=0)

# Prepare single sample
try:
    date = pd.to_datetime(date_str)
    month = date.month
    day = date.day
except Exception:
    month = int(pd.to_datetime(df['date'].iloc[0]).month)
    day = int(pd.to_datetime(df['date'].iloc[0]).day)

input_df = pd.DataFrame([{
    "precipitation": precipitation,
    "temp_min": temp_min,
    "wind": wind,
    "month": month,
    "day": day,
    "weather": weather
}])

st.subheader("Manual input preview")
st.write(input_df)

if st.button("Predict temp_max"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted temp_max: {pred:.2f} °C")

st.subheader("Batch prediction (on uploaded or sample data)")
def prepare_df(df_in):
    df2 = df_in.copy()
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
    df2['month'] = df2['date'].dt.month
    df2['day'] = df2['date'].dt.day
    df2['weather'] = df2.get('weather', 'unknown').fillna('unknown')
    return df2[["precipitation","temp_min","wind","month","day","weather"]]

if st.button("Run batch predictions"):
    to_pred = prepare_df(df)
    preds = model.predict(to_pred)
    df_result = to_pred.copy()
    df_result['pred_temp_max'] = preds
    st.write(df_result.head(20))
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Trained model RMSE: 0.09 °C | R²: 1.000")
