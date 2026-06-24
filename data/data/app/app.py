import streamlit as st
import pandas as pd
import math
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Bike Rescue AI", page_icon="🏍️", layout="centered")
st.markdown("<h1 style='text-align:center;'>🏍️ Bike Rescue AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Describe your bike problem → Get instant diagnosis + nearest mechanic</p>", unsafe_allow_html=True)
st.divider()

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

tips = {
    "Battery/Electrical": ["Check battery terminals for corrosion", "Try jump-starting the bike", "Check if fuses are blown", "Look for loose wiring connections"],
    "Fuel System": ["Check fuel level — may be empty", "Ensure fuel tap is ON", "Smell for petrol leaks near tank", "Check air filter — clean if clogged"],
    "Engine Mechanical": ["Stop riding immediately", "Check engine oil level", "Let engine cool before inspection", "Look for oil leaks underneath"],
    "Tyre/Brakes": ["Do NOT ride — safety critical", "Check tyre pressure visually", "Check brake fluid reservoir level", "Inspect brake pads for wear"],
    "Cooling System": ["Stop and let bike cool 30+ min", "Never open radiator cap when hot", "Check coolant overflow tank level", "Check if radiator fan is spinning"],
    "Transmission/Clutch": ["Check clutch cable tension", "Inspect chain slack", "Look for worn sprocket teeth", "Check clutch plates if slipping"],
}

model = load_model()
mechanics_df = load_mechanics()

tab1, tab2, tab3 = st.tabs(["🔍 Diagnose Fault", "🔧 Find Mechanic", "🚀 Full Rescue"])

with tab1:
    st.subheader("Describe your bike's problem")
    quick = st.selectbox("Quick symptoms:", ["-- Type your own --", "bike won't start", "engine overheating", "fuel leaking", "brakes not working", "gears slipping", "tyre puncture"])
    symptom = st.text_area("Or describe in detail:", value="" if quick == "-- Type your own --" else quick, height=100)
    if st.button("🔍 Diagnose", use_container_width=True):
        if symptom:
            fault = model.predict([symptom])[0]
            confidence = max(model.predict_proba([symptom])[0]) * 100
            st.success(f"Fault Detected: {fault}")
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.0f}%")
            st.subheader("💡 What to do right now:")
            for tip in tips.get(fault, []):
                st.write(f"→ {tip}")
        else:
            st.warning("Please describe your problem first!")

with tab2:
    st.subheader("Find nearby mechanics")
    fault_filter = st.selectbox("Filter by fault type:", ["All", "Battery/Electrical", "Fuel System", "Engine Mechanical", "Tyre/Brakes", "Cooling System", "Transmission/Clutch"])
    scored = []
    for _, row in mechanics_df.iterrows():
        score, dist = score_mechanic(row, fault_filter)
        scored.append({**row.to_dict(), "score": score, "distance_km": dist})
    scored_df = pd.DataFrame(scored).sort_values("score", ascending=False).reset_index(drop=True)
    for i, row in scored_df.iterrows():
        avail = "✅ Available" if row["available"] == True else "❌ Busy"
        with st.expander(f"#{i+1} {row['name']} — ⭐ {row['rating']} | 📍 {row['distance_km']} km | Score: {row['score']}"):
            st.write(f"📞 Phone: {row['phone']}")
            st.write(f"💰 Price: {row['price_range']}")
            st.write(f"🏆 Experience: {row['experience_years']} years")
            st.write(f"🔧 Specialization: {row['specialization']}")
            st.write(f"Status: {avail}")

with tab3:
    st.subheader("Full Rescue Mode")
    rescue_symptom = st.text_area("What happened to your bike?", height=100)
    if st.button("🚀 Find Help Now", use_container_width=True):
        if rescue_symptom:
            fault = model.predict([rescue_symptom])[0]
            confidence = max(model.predict_proba([rescue_symptom])[0]) * 100
            st.success(f"Fault: {fault} ({confidence:.0f}% confidence)")
            st.subheader("💡 Immediate steps:")
            for tip in tips.get(fault, []):
                st.write(f"→ {tip}")
            st.subheader("🔧 Top 3 Mechanics for you:")
            scored = []
            for _, row in mechanics_df.iterrows():
                score, dist = score_mechanic(row, fault)
                scored.append({**row.to_dict(), "score": score, "distance_km": dist})
            top3 = pd.DataFrame(scored).sort_values("score", ascending=False).head(3).reset_index(drop=True)
            for i, row in top3.iterrows():
                avail = "✅ Available" if row["available"] == True else "❌ Busy"
                st.write(f"#{i+1} {row['name']} — ⭐ {row['rating']} | 📍 {row['distance_km']} km | {avail}")
                st.write(f"📞 {row['phone']} | 💰 {row['price_range']}")
                st.divider()
        else:
            st.warning("Please describe your problem first!")
