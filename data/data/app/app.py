import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import math

st.set_page_config(page_title="Bike Rescue AI", page_icon="🏍️", layout="centered")

st.markdown("""
    <h1 style='text-align:center;'>🏍️ Bike Rescue AI</h1>
    <p style='text-align:center; color:gray;'>Describe your bike problem → Get instant diagnosis + nearest mechanic</p>
    <hr>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    df = pd.read_csv("data/bike_faults.csv")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=5))
    ])
    model.fit(df["symptom"], df["fault_category"])
    return model

@st.cache_data
def load_mechanics():
    return pd.read_csv("data/mechanics.csv")

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = (lat2 - lat1) * math.pi / 180
    dLon = (lon2 - lon1) * math.pi / 180
    a = math.sin(dLat/2)**2 + math.cos(lat1*math.pi/180) * math.cos(lat2*math.pi/180) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def score_mechanic(row, fault, user_lat=17.3850, user_lon=78.4900):
    dist = get_distance(user_lat, user_lon, row["lat"], row["lon"])
    spec_match = 1.0 if fault in str(row["specialization"]) or "All" in str(row["specialization"]) else 0.3
    dist_score = max(0, 1 - dist/10)
    rating_score = (row["rating"] - 1) / 4
    exp_score = min(row["experience_years"] / 20, 1)
    avail_score = 1.0 if row["available"] == True else 0.2
    score = (spec_match*0.35 + dist_score*0.3 + rating_score*0.2 + exp_score*0.1 + avail_score*0.05) * 100
    return round(score, 1), round(dist, 1)

model = load_model()
mechanics_df = load_mechanics()

tips = {
    "Battery/Electrical": ["Check battery terminals for corrosion", "Try jump-starting the bike", "Check if fuses are blown", "Look for loose wiring connections"],
    "Fuel System": ["Check fuel level — may be empty", "Ensure fuel tap is ON", "Smell for petrol leaks near tank", "Check air filter — clean if clogged"],
    "Engine Mechanical": ["Stop riding immediately", "Check engine oil level", "Let engine cool before inspection", "Look for oil leaks underneath"],
    "Tyre/Brakes": ["Do NOT ride — safety critical", "Check tyre pressure visually", "Check brake fluid reservoir level", "Inspect brake pads for wear"],
    "Cooling System": ["Stop and let bike cool 30+ min", "Never open radiator cap when hot", "Check coolant overflow tank level", "Check if radiator fan is spinning"],
    "Transmission/Clutch": ["Check clutch cable tension", "Inspect chain slack", "Look for worn sprocket teeth", "Check clutch plates if slipping"],
}

tab1, tab2, tab3 = st.tabs(["🔍 Diagnose Fault", "🔧 Find Mechanic", "🚀 Full Rescue"])

with tab1:
    st.subheader("Describe your bike's problem")
    quick = st.selectbox("Quick symptoms:", ["-- Type your own --", "bike won't start", "engine overheating", "fuel leaking", "brakes not working", "gears slipping", "tyre puncture"])
    symptom = st.text_area("Or describe in detail:", value="" if quick == "-- Type your own --" else quick, height=100)
    if st.button("🔍 Diagnose", use_container_width=True):
        if symptom:
            fault = model.predict([symptom])[0]
            confidence = max(model.predict_proba([symptom])[0]) * 100
