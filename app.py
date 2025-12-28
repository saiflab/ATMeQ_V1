import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from pathlib import Path
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ======================================================
# 1. PAGE CONFIG (MUST BE FIRST)
# ======================================================
st.set_page_config(
    page_title="ATMeQ | Precision Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 2. REMOVE STREAMLIT HEADER / GITHUB / FORK / FOOTER
# ======================================================
st.markdown("""
<style>
/* Remove Streamlit top bar */
header {visibility: hidden;}
[data-testid="stHeader"] {display: none;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}

/* Remove menu (⋮) and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Remove blank space left by header */
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. GLOBAL MODERN DARK THEME
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background-color: #0b1121;
    background-image:
        linear-gradient(rgba(11,17,33,.92), rgba(11,17,33,.92)),
        linear-gradient(#1e293b 1px, transparent 1px),
        linear-gradient(90deg, #1e293b 1px, transparent 1px);
    background-size: 100% 100%, 40px 40px, 40px 40px;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #f8fafc !important;
    letter-spacing: -0.5px;
}

.gradient-text {
    background: linear-gradient(90deg, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}

/* Glass card */
.glass-card {
    background: rgba(30,41,59,.45);
    border: 1px solid rgba(148,163,184,.15);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(12px);
}

/* Input configuration panel */
.panel {
    background: rgba(15,23,42,.65);
    border: 1px solid rgba(148,163,184,.2);
    border-radius: 18px;
    padding: 18px;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 900;
    margin-bottom: 10px;
}

/* Pills */
.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.pill {
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    background: rgba(34,211,238,.15);
    color: #67e8f9;
    border: 1px solid rgba(34,211,238,.25);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(2,6,23,.4);
    border: 2px dashed rgba(56,189,248,.7);
    border-radius: 18px;
    padding: 20px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #22d3ee;
    box-shadow: 0 0 25px rgba(34,211,238,.25);
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(180deg, #ffffff, #e2e8f0) !important;
    color: #0b1121 !important;
    font-weight: 900 !important;
    border-radius: 12px !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg,#06b6d4,#3b82f6);
    color: white !important;
    border-radius: 12px;
    font-weight: 900;
    padding: 12px;
}
div.stButton > button:hover {
    box-shadow: 0 12px 30px rgba(6,182,212,.4);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 4. LOAD MODEL
# ======================================================
@st.cache_resource
def load_resources():
    model, scaler = None, None
    try:
        with open("ATMeQ.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        pass
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except:
        pass
    return model, scaler

model, saved_scaler = load_resources()

# ======================================================
# 5. SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## 🧬 **ATMeQ**")
    selected_page = st.radio(
        "",
        ["Home", "Run Diagnostics", "Research Team"],
        index=0
    )

# ======================================================
# 6. HOME PAGE
# ======================================================
if selected_page == "Home":
    st.markdown("""
    <h1>Welcome to <span class="gradient-text">ATMeQ</span></h1>
    <p>Machine-learning-based ALS prediction using RNA-seq expression profiles.</p>
    """, unsafe_allow_html=True)

# ======================================================
# 7. DIAGNOSTIC PAGE
# ======================================================
elif selected_page == "Run Diagnostics":

    st.markdown('<h1 class="gradient-text">Diagnostic Console</h1>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not found (ATMeQ.pkl missing).")
    else:
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">Input Configuration</div>
                <p><b>Required Gene Markers</b></p>
                <div class="pills">
                    <span class="pill">ACTA1</span>
                    <span class="pill">ABCA4</span>
                    <span class="pill">COL6A4P2</span>
                    <span class="pill">HERC2P2</span>
                    <span class="pill">KCNE4</span>
                    <span class="pill">LOC107987008</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Drop VST Data File Here",
                type=["csv"],
                help="CSV • max 200MB"
            )

            if uploaded_file:
                df = pd.read_csv(uploaded_file, index_col=0)
                st.success(f"Loaded {len(df)} samples")

                required = ["ACTA1","ABCA4","COL6A4P2","HERC2P2","KCNE4","LOC107987008"]
                if all(c in df.columns for c in required):
                    if st.button("⚡ INITIATE ANALYSIS", use_container_width=True):
                        X = df[required]
                        Xs = saved_scaler.transform(X) if saved_scaler else StandardScaler().fit_transform(X)
                        st.session_state["res"] = (X, model.predict(Xs), model.predict_proba(Xs))
                else:
                    st.error("Missing required gene columns")

        with col_right:
            if "res" in st.session_state:
                X, preds, probs = st.session_state["res"]
                is_als = preds[0] == 1
                color = "#ef4444" if is_als else "#22c55e"

                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <h3 style="color:{color}; font-size:3rem;">
                        {"POSITIVE" if is_als else "NEGATIVE"}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probs[0][1]*100,
                    number={'suffix': "%"},
                    gauge={'axis': {'range':[0,100]}, 'bar': {'color': color}}
                ))
                fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 8. TEAM PAGE
# ======================================================
elif selected_page == "Research Team":
    st.markdown('<h1 class="gradient-text">Research Team</h1>', unsafe_allow_html=True)

    cols = st.columns(2, gap="large")
    with cols[0]:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h3>Ahmed Saif, M.Pharm.</h3>
            <p>Graduate Student<br>University of Rajshahi</p>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h3>Md Tarikul Islam, MSc</h3>
            <p>Graduate Student<br>Jashore University of Science and Technology</p>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# 9. FOOTER (CUSTOM)
# ======================================================
st.markdown("""
<div style="text-align:center; margin-top:80px; padding:20px; border-top:1px solid #1e293b; color:#64748b; font-size:0.85em;">
    ATMeQ v2.0 Pro © 2025 Saif Lab
</div>
""", unsafe_allow_html=True)
