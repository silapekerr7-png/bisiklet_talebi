import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bisiklet Talebi Tahmini", page_icon="🚲", layout="centered")
st.title("🚲 Bisiklet Kiralama Talebi Tahmini")
st.write("Eğitilmiş modelle saatlik bisiklet kiralama talebini tahmin et.")

try:
    model = joblib.load("bike_sharing_model.pkl")
    columns = joblib.load("model_columns.pkl")
except FileNotFoundError:
    st.error("Model dosyaları bulunamadı. Önce Bike_Sharing_Demand.ipynb dosyasını çalıştırıp modeli kaydet.")
    st.stop()

with st.form("tahmin_formu"):
    col1, col2 = st.columns(2)
    with col1:
        season = st.selectbox("Mevsim", [1, 2, 3, 4], format_func=lambda x: {1:"İlkbahar",2:"Yaz",3:"Sonbahar",4:"Kış"}[x])
        holiday = st.selectbox("Tatil günü mü?", [0, 1], format_func=lambda x: "Evet" if x else "Hayır")
        workingday = st.selectbox("İş günü mü?", [0, 1], format_func=lambda x: "Evet" if x else "Hayır")
        weather = st.selectbox("Hava durumu kodu", [1, 2, 3, 4])
        temp = st.slider("Sıcaklık (°C)", 0.0, 45.0, 20.0)
    with col2:
        atemp = st.slider("Hissedilen sıcaklık (°C)", 0.0, 50.0, 20.0)
        humidity = st.slider("Nem (%)", 0, 100, 50)
        windspeed = st.slider("Rüzgâr hızı", 0.0, 70.0, 10.0)
        year = st.selectbox("Yıl", [2011, 2012])
        month = st.slider("Ay", 1, 12, 6)
        day = st.slider("Ayın günü", 1, 19, 10)
        hour = st.slider("Saat", 0, 23, 8)
    submit = st.form_submit_button("Talebi tahmin et")

if submit:
    dt = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    dayofweek = dt.dayofweek
    row = {
        "season": season, "holiday": holiday, "workingday": workingday,
        "weather": weather, "temp": temp, "atemp": atemp,
        "humidity": humidity, "windspeed": windspeed,
        "year": year, "month": month, "day": day, "hour": hour,
        "dayofweek": dayofweek, "is_weekend": int(dayofweek >= 5)
    }
    input_df = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    prediction = max(0, float(model.predict(input_df)[0]))
    st.metric("Tahmini saatlik kiralama", f"{prediction:.0f} bisiklet")
    st.caption("Bu bir makine öğrenmesi tahminidir; gerçek talep farklı olabilir.")
