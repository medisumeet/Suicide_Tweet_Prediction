import streamlit as st
import joblib
import re

# -------------------------
# Load trained components
# -------------------------
model = joblib.load("suicide_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------
# Text Cleaning (same as training)
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Suicide Tweet Predictor",
    page_icon="🧠",
    layout="centered"
)

dark_mode = st.toggle("Dark mode",value = True)

# Base CSS
st.markdown(
    """
    <style>
    .stApp { background: #f5f5f5; }
    .header-card, .input-card, .result-card, .footer-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }
    .header-title {
        background: linear-gradient(135deg,#667eea,#764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .header-subtitle {
        color:#666;
        text-align:center;
    }
    .stButton button{
        background:linear-gradient(135deg,#667eea,#764ba2);
        color:white;
        border:none;
        padding:0.75rem 2rem;
        border-radius:25px;
        font-size:1.1rem;
        font-weight:600;
        width:100%;
        margin-top:1rem;
    }
    .result-emoji{
        font-size:3.5rem;
        text-align:center;
    }
    .result-message{
        font-size:1.4rem;
        font-weight:bold;
        text-align:center;
    }
    .result-probability{
        font-size:2rem;
        font-weight:bold;
        text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dark mode CSS
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background:#0f172a; }
        .header-card,.input-card,.result-card,.footer-card{
            background:#111827;
            color:#e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Header
st.markdown(
"""
<div class="header-card">
<div class="header-title">🧠 Suicide Tweet Predictor</div>
<div class="header-subtitle">
Enter a tweet and the model estimates suicide risk.
</div>
</div>
""",
unsafe_allow_html=True
)

# Input
tweet = st.text_area("Enter Tweet", height=150)

# Prediction
if st.button("Predict"):

    if tweet.strip()=="":
        st.warning("Please enter text.")

    else:
        try:

            # CLEAN
            tweet_clean = clean_text(tweet)

            # VECTORIZATION
            tweet_vec = vectorizer.transform([tweet_clean])

            # MODEL PREDICTION
            prob = model.predict_proba(tweet_vec)[0][1]

            prob_pct = round(prob*100,2)

            if prob_pct < 30:
                color="#4CAF50"
                emoji="✅"
                msg="Low Suicide Risk"

            elif prob_pct < 70:
                color="#FFC107"
                emoji="⚠️"
                msg="Moderate Risk – Monitor"

            else:
                color="#FF4C4C"
                emoji="🚨"
                msg="Potential Suicide Risk Detected"

            st.markdown(
            f"""
            <div class="result-card" style="border-left:6px solid {color};">
            <div class="result-emoji">{emoji}</div>
            <div class="result-message" style="color:{color};">{msg}</div>
            <div class="result-probability" style="color:{color};">
            Probability: {prob_pct}%
            </div>
            </div>
            """,
            unsafe_allow_html=True
            )

            st.progress(int(prob_pct))

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")

st.markdown(
"""
<div class="footer-card">
<div class="footer-text">
Developed by <strong>InsightMonk</strong> | Educational use only
</div>
</div>
""",
unsafe_allow_html=True
)