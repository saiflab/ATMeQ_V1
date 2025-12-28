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
# 2. Assets (Embedded SVGs)
# -----------------------------
icons = {
    "prepare": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 7 6.82 21.18a2.83 2.83 0 0 1-3.99-.01v0a2.83 2.83 0 0 1 0-4L17 3z"/><path d="m16 2 6 6"/><path d="M12 16H4"/></svg>""",
    "upload": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>""",
    "compute": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/><path d="m15 9 3 3-3 3"/><path d="m9 9-3 3 3 3"/></svg>""",
    "result": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>""",
    "dna": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="M17 17c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="M2 9c6.667 6 13.333 0 20 6"/><path d="M7 7c1.798-1.998 2.518-3.995 2.807-5.993"/></svg>"""
}

# -----------------------------
# 3. Modern Dark Theme CSS (Updated)
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700&display=swap');

    /* --- GLOBAL THEME --- */
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

    /* --- SIDEBAR CUSTOMIZATION (Updated for White Font) --- */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Force Sidebar Radio Options (Home, etc) to White */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Force Sidebar Captions (AI Model, Version) to White */
    [data-testid="stSidebar"] .stCaption {
        color: #ffffff !important;
        opacity: 0.9;
    }
    
    /* Ensure other sidebar text is white */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* --- FILE UPLOADER (Updated for Black Button Text) --- */
    [data-testid="stFileUploader"] button {
        color: #000000 !important; /* Force Black Text */
        font-weight: 600;
        background-color: #f1f5f9; /* Light background for contrast */
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #22d3ee, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* --- CARDS (GLASSMORPHISM) --- */
    .tech-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        height: 100%;
        position: relative;
    }
    
    .tech-card:hover {
        transform: translateY(-5px);
        border-color: #22d3ee;
        box-shadow: 0 10px 30px -10px rgba(34, 211, 238, 0.2);
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

    /* --- BUTTONS --- */
    div.stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white !important;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.4);
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
        color: white !important;
    }

    /* --- TEAM --- */
    .avatar-glow {
        border-radius: 50%;
        border: 3px solid #06b6d4;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        width: 140px;
        height: 140px;
        object-fit: cover;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 4. Logic & Resources
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
# 5. Advanced Sidebar
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
    
    # System Status
    st.markdown("<p style='font-size: 11px; color: #ffffff; font-weight:600; letter-spacing: 1.2px; margin-bottom:15px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.caption("AI Model")
        if model:
            st.markdown("<span style='color:#4ade80 !important; font-weight:bold; font-size:14px;'>● Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#ef4444 !important; font-weight:bold; font-size:14px;'>● Inactive</span>", unsafe_allow_html=True)
            
    with col_stat2:
        st.caption("Version")
        st.markdown("<span style='color:#ffffff !important; font-weight:bold; font-size:14px;'>v2.0 Pro</span>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
    st.info("💡 **Requirement:** Upload DESeq2 VST normalized data only.")

# -----------------------------
# 6. Page Content
# -----------------------------

# === HOME ===
if selected_page == "Home":
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Hero Section
    col_text, col_img = st.columns([1.8, 1])
    
    with col_text:
        st.markdown('<h1 style="font-size: 3.5rem; line-height: 1.2;">Next-Gen <br><span class="gradient-text">ALS Diagnostics</span></h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.25rem; line-height: 1.7; color: #cbd5e1; margin: 20px 0 30px 0;">
            <b>ATMeQ</b> leverages high-dimensional transcriptomic data to detect Amyotrophic Lateral Sclerosis biomarkers with clinical-grade precision. 
            Powered by advanced Support Vector Machine learning kernels.
        </p>
        """, unsafe_allow_html=True)
        
        # High contrast button
        if st.button("🚀 Launch Analysis Engine"):
            st.toast("Switching to Diagnostics tab...", icon="⚡")

    with col_img:
        # LOGO LOGIC
        logo_path = "logo.png"
        if Path(logo_path).exists():
            st.image(logo_path, use_container_width=True)
        else:
            # Fallback
            st.markdown("""
            <div style="background: radial-gradient(circle at center, rgba(6,182,212,0.2) 0%, transparent 70%); padding:40px; border-radius:50%; text-align:center;">
                <h1 style="font-size:100px;">🧬</h1>
                <p style="color:#22d3ee">ATMeQ Lab</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
    
    st.subheader("Workflow Architecture")
    st.markdown("---")
    
    # Step Cards
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        {"icon": icons['prepare'], "title": "Data Prep", "desc": "Normalize raw counts via DESeq2 (VST)."},
        {"icon": icons['upload'], "title": "Secure Upload", "desc": "Drag & drop CSV. Local processing only."},
        {"icon": icons['compute'], "title": "Inference", "desc": "SVM Kernel analyzes gene signatures."},
        {"icon": icons['result'], "title": "Diagnostics", "desc": "Probabilistic scoring & classification."},
    ]
    
    for col, step in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="tech-card">
                <div class="icon-box">{step['icon']}</div>
                <h4 style="color:#f1f5f9; margin:0 0 10px 0;">{step['title']}</h4>
                <p style="font-size:0.95em; color:#94a3b8; line-height:1.5;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# === DIAGNOSTICS ===
elif selected_page == "Run Diagnostics":
    st.markdown('<h1 class="gradient-text">Diagnostic Console</h1>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ SYSTEM ALERT: Model file (ATMeQ.pkl) not detected on server.")
    else:
        col_ctrl, col_viz = st.columns([1, 2], gap="large")
        
        with col_ctrl:
            st.markdown("""
            <div class="tech-card" style="border-left: 4px solid #06b6d4;">
                <h4 style="color:#fff; margin-top:0;">Input Configuration</h4>
                <p style="font-size:0.9em; margin-bottom:10px;">Required Gene Markers:</p>
                <code style="color:#22d3ee; background:rgba(0,0,0,0.3)">ACTA1, ABCA4, COL6A4P2...</code>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=["csv"], help="Limit 200MB")
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file, index_col=0)
                st.markdown(f"<div style='margin:10px 0; color:#4ade80'>✅ Loaded {len(df)} samples successfully</div>", unsafe_allow_html=True)
                
                required_cols = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("⚡ INITIATE ANALYSIS", use_container_width=True):
                        X = df[required_cols].copy()
                        
                        if saved_scaler:
                            X_scaled = saved_scaler.transform(X)
                        else:
                            st.warning("⚠️ Auto-scaling (experimental)")
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                        preds = model.predict(X_scaled)
                        probs = model.predict_proba(X_scaled)
                        st.session_state['res'] = (X, preds, probs)

        with col_viz:
            if 'res' in st.session_state:
                X_res, preds_res, probs_res = st.session_state['res']
                
                top_prob = probs_res[0][1] * 100
                is_als = preds_res[0] == 1
                color = "#ef4444" if is_als else "#22c55e"
                status_text = "POSITIVE" if is_als else "NEGATIVE"
                
                # Result Card
                st.markdown(f"""
                <div class="tech-card" style="text-align:center; border: 1px solid {color}; box-shadow: 0 0 30px {color}20;">
                    <h5 style="margin:0; color:#94a3b8; letter-spacing:2px; font-size:12px;">PRIMARY DETECTION</h5>
                    <h1 style="font-size: 4rem; color: {color}; margin: 15px 0; text-shadow: 0 0 20px {color}40;">{status_text}</h1>
                    <p style="color:#fff;">Sample ID: <b>{X_res.index[0]}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = top_prob,
                    number = {'suffix': "%", 'font': {'color': "#f8fafc", 'size': 50}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "CONFIDENCE INTERVAL", 'font': {'size': 14, 'color': "#94a3b8", 'family': "sans-serif"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                        'bar': {'color': color, 'thickness': 1},
                        'bgcolor': "rgba(30, 41, 59, 0)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 100], 'color': "rgba(148, 163, 184, 0.1)"}],
                    }
                ))
                fig.update_layout(height=280, margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                res_df = pd.DataFrame({
                    "Sample ID": X_res.index,
                    "Diagnosis": np.where(preds_res == 1, "ALS Positive", "Healthy Control"),
                    "Confidence": np.round(probs_res[:, 1], 4)
                })
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                
            else:
                st.markdown("""
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:450px; border: 2px dashed #334155; border-radius:16px; background:rgba(30,41,59,0.3);">
                    <div style="font-size:60px; margin-bottom:20px; opacity:0.5;">📊</div>
                    <h3 style="color:#94a3b8; margin:0;">Awaiting Data Input</h3>
                    <p style="color:#64748b">Upload CSV to visualize biomarkers</p>
                </div>
                """, unsafe_allow_html=True)

# === TEAM ===
elif selected_page == "Research Team":
    st.markdown('<h1 class="gradient-text">Research Team</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1em;'>The minds behind the ATMeQ algorithm.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    team_members = [
        {"name": "Ahmed Saif, B.Pharm.", "role": "Graduate Researcher", "uni": "UNC Charlotte", "img": "Ahmed_Saif.png"},
        {"name": "Md Obayed Raihan, Ph.D", "role": "Assistant Professor", "uni": "Chicago State University", "img": "Obayed_Raihan.png"},
        {"name": "Bioinformatics Lead", "role": "Data Analyst", "uni": "Research Lab", "img": "ast.jpg"}
    ]
    
    cols = st.columns(len(team_members))
    
    for idx, member in enumerate(team_members):
        with cols[idx]:
            if Path(member['img']).exists():
                src = f"data:image/png;base64,{get_img_as_base64(member['img'])}"
            else:
                src = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

            st.markdown(f"""
            <div class="tech-card" style="text-align:center;">
                <img src="{src}" class="avatar-glow">
                <h3 style="margin-bottom:5px; color:#fff;">{member['name']}</h3>
                <p style="color:#22d3ee; font-weight:700; margin:0; text-transform:uppercase; font-size:0.85em; letter-spacing:1px;">{member['role']}</p>
                <p style="font-size:0.9em; margin-top:5px; color:#94a3b8;">{member['uni']}</p>
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
