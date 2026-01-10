import streamlit as st
import joblib

# -------------------------
# Load the trained model
# -------------------------
model = joblib.load("bow_logreg_model.pkl")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Suicide Tweet Predictor",
    page_icon="🧠",
    layout="centered"
)

# THE NEW THING (calm down, it's just a toggle)
dark_mode = st.toggle("Dark mode")

# Simple Enhanced CSS (base light theme)
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-subtitle { color: #666; font-size: 1.1rem; text-align: center; }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .result-emoji { font-size: 3.5rem; text-align: center; margin-bottom: 1rem; }
    .result-message { font-size: 1.4rem; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .result-probability { font-size: 2rem; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .footer-text { color: #666; text-align: center; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# DARK THEME OVERRIDE (only applied when toggle is ON)
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background: #0f172a; }
        .header-card, .input-card, .result-card, .footer-card {
            background: #111827;
            color: #e5e7eb;
            box-shadow: 0 10px 30px rgba(255, 255, 255, 0.05);
        }
        label, .footer-text { color: #e5e7eb !important; }
        .header-subtitle { color: #9ca3af !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Header
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">🧠 Suicide Tweet Predictor</div>
        <div class="header-subtitle">Enter a tweet below and the model will predict if it's a potential suicidal post.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Input area
tweet = st.text_area("Enter Tweet:", height=150)

# Predict button
if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        try:
            prob = model.predict_proba([tweet])[0][1]
            pred = 1 if prob >= 0.5 else 0
            prob_pct = round(prob * 100, 2)

            if prob_pct < 30:
                color = "#4CAF50"
                emoji = "✅"
                msg = "Not a Suicide Post"
            elif prob_pct < 70:
                color = "#FFC107"
                emoji = "⚠️"
                msg = "Moderate Risk – Monitor"
            else:
                color = "#FF4C4C"
                emoji = "🚨"
                msg = "Potential Suicide Post Detected!"

            st.markdown(
                f"""
                <div class="result-card" style="border-left: 6px solid {color};">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-message" style="color: {color};">{msg}</div>
                    <div class="result-probability" style="color: {color};">Probability: {prob_pct}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(int(prob_pct))

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown(
    """
    <div class="footer-card">
        <div class="footer-text">
            Developed with ❤️ by <strong>InsightMonk</strong> | For educational purposes only.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
