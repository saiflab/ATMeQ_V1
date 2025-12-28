import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from pathlib import Path
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Configuration
# -----------------------------
st.set_page_config(
    page_title="ATMeQ | Precision Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# ✅ ONLY ADDITION: Hide top Streamlit bar (Fork/GitHub/Deploy/Rerun)
# -----------------------------
st.markdown("""
<style>
/* Hide ONLY the top Streamlit chrome */
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Remove the empty space left behind */
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. Assets (Embedded SVGs)
# -----------------------------
icons = {
    "dna": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="M17 17c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="M2 9c6.667 6 13.333 0 20 6"/><path d="M7 7c1.798-1.998 2.518-3.995 2.807-5.993"/></svg>""",
    "step1": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>""",
    "step2": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>""",
    "step3": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/></svg>""",
    "step4": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/><path d="M21 5h-4a2 2 0 0 1-2-2v-4"/><polyline points="9 17 9 12 15 12 15 17"/></svg>""",
    "integration": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>""",
    "ml": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/><path d="M12 2a10 10 0 0 1 10 10h-10V2z"/><path d="M12 12 2.1 12.05"/><path d="M12 12 5.5 3.5"/></svg>""",
    "user": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>""",
    "rapid": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>""",
    "light": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/><line x1="16" x2="2" y1="8" y2="22"/><line x1="17.5" x2="9" y1="15" y2="15"/></svg>"""
}

# -----------------------------
# 3. Modern Dark Theme CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #0b1121;
        background-image: 
            linear-gradient(rgba(11, 17, 33, 0.9), rgba(11, 17, 33, 0.9)),
            linear-gradient(#1e293b 1px, transparent 1px), 
            linear-gradient(90deg, #1e293b 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        color: #e2e8f0;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }
    p, li {
        color: #cbd5e1;
        line-height: 1.6;
    }
    .gradient-text {
        background: linear-gradient(90deg, #22d3ee, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* --- CARDS --- */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        backdrop-filter: blur(10px);
    }
    .feature-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .feature-card:hover {
        border-color: #818cf8;
        transform: translateY(-5px);
    }
    .icon-box {
        background: rgba(34, 211, 238, 0.1);
        width: 60px;
        height: 60px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
        border: 1px solid rgba(34, 211, 238, 0.2);
    }

    /* --- ADVANCED UPLOAD ZONE STYLING --- */
    [data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.8);
        border: 2px dashed #3b82f6;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #22d3ee;
        background-color: rgba(30, 41, 59, 0.9);
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.15);
    }
    [data-testid="stFileUploader"] section {
        padding: 0;
    }
    
    /* Browse Button Styling */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(to bottom, #ffffff, #e2e8f0);
        color: #0f172a !important; /* Strong Black Text */
        border: 1px solid #cbd5e1;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    [data-testid="stFileUploader"] button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
        border-color: #ffffff;
    }
    
    /* Upload Text */
    [data-testid="stFileUploader"] div[role="button"] {
        color: #94a3b8;
    }

    /* --- MAIN BUTTONS --- */
    div.stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white !important;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 4. Resources
# -----------------------------
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    try:
        with open("ATMeQ.pkl", "rb") as f: model = pickle.load(f)
    except: pass
    try:
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
    except: pass
    return model, scaler

def get_img_as_base64(file_path):
    if not Path(file_path).exists(): return ""
    with open(file_path, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

model, saved_scaler = load_resources()

# -----------------------------
# 5. Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:25px;">
        {icons['dna']}
        <h2 style="margin:0; font-size: 26px; font-weight:700; color:white;">ATMeQ</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 11px; color: #ffffff; font-weight:600; letter-spacing: 1.2px; margin-bottom:10px;'>MENU</p>", unsafe_allow_html=True)
    
    selected_page = st.radio(
        "Page Navigation",
        ["Home", "Run Diagnostics", "Research Team"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<p style='font-size: 11px; color: #ffffff; font-weight:600; letter-spacing: 1.2px; margin-bottom:15px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.caption("AI Model")
        st.markdown(f"<span style='color:{'#4ade80' if model else '#ef4444'} !important; font-weight:bold;'>● {'Active' if model else 'Offline'}</span>", unsafe_allow_html=True)
    with col_stat2:
        st.caption("Version")
        st.markdown("<span style='color:#ffffff !important; font-weight:bold;'>v2.0 Pro</span>", unsafe_allow_html=True)

# -----------------------------
# 6. Page Content
# -----------------------------

# === HOME PAGE ===
if selected_page == "Home":
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # 1. HEADER
    col_intro, col_logo = st.columns([2, 1])
    with col_intro:
        st.markdown('<h1 style="font-size: 3rem; line-height: 1.1;">Welcome to <span class="gradient-text">ATMeQ</span></h1>', unsafe_allow_html=True)
        st.markdown("### ALS Prediction Tool using Machine Learning and RNA-Seq")
        st.markdown("""
        **ATMeQ** is a state-of-the-art tool designed to predict Amyotrophic Lateral Sclerosis (ALS) with unmatched precision. 
        By combining the power of machine learning with RNA-Seq data, ATMeQ provides accurate ALS predictions based on key gene expressions.
        """)
    with col_logo:
        logo_path = "logo.png"
        if Path(logo_path).exists():
            st.image(logo_path, use_container_width=True)
        else:
            st.image("https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&w=800&q=80", use_container_width=True)

    st.markdown("---")
    
    # 2. MOTIVATION
    col_motiv_img, col_motiv_text = st.columns([1, 1.5], gap="large")
    with col_motiv_img:
        st.image("https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=800&q=80", caption="Neurodegenerative Research", use_container_width=True)
    with col_motiv_text:
        st.markdown('<h2 class="gradient-text">Motivation</h2>', unsafe_allow_html=True)
        st.markdown("""
        **Amyotrophic Lateral Sclerosis (ALS)** is a devastating neurodegenerative disease characterized by progressive motor neuron degeneration. 
        Accurate and early diagnosis is paramount for facilitating timely therapeutic interventions.
        
        **ATMeQ** integrates machine learning with high-throughput RNA-Seq data to identify robust gene signatures, 
        empowering researchers with precise molecular insights for early detection.
        """)

    st.markdown("---")

    # 3. FEATURES
    st.subheader("Key Features")
    f1, f2, f3, f4, f5 = st.columns(5)
    features = [
        {"icon": icons["integration"], "title": "Integration", "desc": "Simple API connection."},
        {"icon": icons["ml"], "title": "Machine Learning", "desc": "Data-driven accuracy."},
        {"icon": icons["user"], "title": "User-Friendly", "desc": "Intuitive design."},
        {"icon": icons["rapid"], "title": "Rapid Prediction", "desc": "Instant outputs."},
        {"icon": icons["light"], "title": "Lightweight", "desc": "High efficiency."},
    ]
    for col, feat in zip([f1, f2, f3, f4, f5], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div style="display:flex; justify-content:center; margin-bottom:10px;">{feat['icon']}</div>
                <h4 style="font-size:1.1em; color:white;">{feat['title']}</h4>
                <p style="font-size:0.85em; margin:0;">{feat['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. STEPS
    st.subheader("Steps to Use ATMeQ")
    s1, s2, s3, s4 = st.columns(4)
    steps_data = [
        {"icon": icons["step1"], "title": "Step 1: Launch", "desc": "Launch the <a href='#' style='color:#22d3ee'>ATMeQ Application</a>."},
        {"icon": icons["step2"], "title": "Step 2: Upload", "desc": "Upload your VST file. <a href='https://github.com/saiflab/ATMeQ' style='color:#22d3ee'>Example File</a>."},
        {"icon": icons["step3"], "title": "Step 3: Compute", "desc": "Run predictions instantly."},
        {"icon": icons["step4"], "title": "Step 4: Result", "desc": "Download diagnostics report."},
    ]
    for col, step in zip([s1, s2, s3, s4], steps_data):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="icon-box" style="margin:0 auto 15px auto;">{step['icon']}</div>
                <h4 style="margin:0 0 10px 0;">{step['title']}</h4>
                <p style="font-size:0.9em;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # 5. FOOTER
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    col_contact, col_project = st.columns(2)
    with col_contact:
        st.markdown("### 📩 Contact")
        st.markdown("If you have any questions, feedback, or issues, please don't hesitate to contact us at:")
        st.markdown("📧 **tamim.ahmedsaif@gmail.com**")
        st.markdown("### 📜 License")
        st.markdown("This project is licensed under the **GPL-3.0 license**.")
    with col_project:
        st.markdown("### 🔬 Ongoing Project")
        st.markdown("""
        <div class="glass-card">
            <p style="color:white; font-style:italic;">
                "ATMeQ: A Machine Learning-Based Framework for Amyotrophic Lateral Sclerosis Disease using RNA-seq Meta-Analysis."
            </p>
            <p style="color:#22d3ee; margin-top:10px;">— Ahmed Saif</p>
        </div>
        """, unsafe_allow_html=True)


# === DIAGNOSTICS PAGE ===
elif selected_page == "Run Diagnostics":
    st.markdown('<h1 class="gradient-text">Diagnostic Console</h1>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ SYSTEM ALERT: Model file (ATMeQ.pkl) not detected on server.")
    else:
        col_ctrl, col_viz = st.columns([1, 2], gap="large")
        with col_ctrl:
            st.markdown("""
            <div class="glass-card" style="border-left: 4px solid #06b6d4; margin-bottom: 20px;">
                <h4 style="color:#fff; margin-top:0;">Input Configuration</h4>
                <p style="font-size:0.9em;">Required Gene Markers: <code style="color:#22d3ee">ACTA1, ABCA4, COL6A4P2...</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            # ADVANCED UPLOAD ZONE
            uploaded_file = st.file_uploader("Drop VST Data File Here", type=["csv"], help="Limit 200MB")
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file, index_col=0)
                st.markdown(f"<div style='margin:10px 0; text-align:center; color:#4ade80; background:rgba(74, 222, 128, 0.1); padding:10px; border-radius:8px;'>✅ Loaded {len(df)} samples</div>", unsafe_allow_html=True)
                required = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
                if all(col in df.columns for col in required):
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("⚡ INITIATE ANALYSIS", use_container_width=True):
                        X = df[required].copy()
                        if saved_scaler: X_scaled = saved_scaler.transform(X)
                        else: 
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                        st.session_state['res'] = (X, model.predict(X_scaled), model.predict_proba(X_scaled))
                else:
                    st.error(f"Missing columns: {[c for c in required if c not in df.columns]}")

        with col_viz:
            if 'res' in st.session_state:
                X_res, preds, probs = st.session_state['res']
                is_als = preds[0] == 1
                color = "#ef4444" if is_als else "#22c55e"
                
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; border: 1px solid {color}; box-shadow: 0 0 30px {color}20;">
                    <h5 style="color:#94a3b8; font-size:12px; letter-spacing:2px;">PRIMARY DETECTION</h5>
                    <h1 style="font-size: 3.5rem; color: {color}; margin: 10px 0;">{'POSITIVE' if is_als else 'NEGATIVE'}</h1>
                    <p style="color:#fff;">Sample: <b>{X_res.index[0]}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=probs[0][1]*100,
                    number={'suffix': "%", 'font': {'color': "#f8fafc", 'size':40}},
                    title={'text': "CONFIDENCE", 'font': {'color': "#94a3b8"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"}, 
                        'bar': {'color': color}, 
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'bordercolor': "#334155"
                    }
                ))
                fig.update_layout(height=280, margin=dict(t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
                res_df = pd.DataFrame({"Sample": X_res.index, "Result": np.where(preds==1, "ALS", "Control"), "Conf": probs[:,1]})
                st.dataframe(res_df, use_container_width=True, hide_index=True)
            else:
                 st.markdown("""
                 <div class='glass-card' style='text-align:center; padding:60px; border:2px dashed #334155;'>
                    <div style='font-size:50px; opacity:0.3; margin-bottom:20px;'>📊</div>
                    <h3 style='color:#64748b'>Awaiting Data Input</h3>
                    <p>Upload a VST CSV file to view diagnostic results</p>
                 </div>
                 """, unsafe_allow_html=True)

# === TEAM PAGE ===
elif selected_page == "Research Team":
    st.markdown('<h1 class="gradient-text">Research Team</h1>', unsafe_allow_html=True)
    st.markdown("---")
    team = [
        {"name": "Ahmed Saif, M.Pharm.", "role": "Graduate Student", "uni": "University of Rajshahi, Bangladesh", "img": "Ahmed_Saif.png"},
        {"name": "Md Tarikul Islam, MSc", "role": "Graduate Student", "uni": "Jashore University of Science and Technology, Bangladesh", "img": "Tarikul_Islam.png"}
    ]
    cols = st.columns(len(team))
    for idx, mem in enumerate(team):
        with cols[idx]:
            src = f"data:image/png;base64,{get_img_as_base64(mem['img'])}" if Path(mem['img']).exists() else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <img src="{src}" style="width:120px; height:120px; border-radius:50%; border:3px solid #06b6d4; margin-bottom:15px; box-shadow:0 0 20px rgba(6,182,212,0.3);">
                <h3 style="margin-bottom:5px; color:#fff;">{mem['name']}</h3>
                <p style="color:#22d3ee; font-weight:700; font-size:0.85em; text-transform:uppercase;">{mem['role']}</p>
                <p style="font-size:0.9em; color:#94a3b8;">{mem['uni']}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# 7. Footer
# -----------------------------
st.markdown("""
<div style="text-align:center; margin-top:80px; padding:20px; border-top:1px solid #1e293b; color:#64748b; font-size:0.85em;">
    ATMeQ v2.0 Pro | © 2025 Saif Lab | Powered by Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
