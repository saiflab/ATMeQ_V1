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
#    These will ALWAYS load because they are code, not images.
# -----------------------------
icons = {
    "prepare": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 7 6.82 21.18a2.83 2.83 0 0 1-3.99-.01v0a2.83 2.83 0 0 1 0-4L17 3z"/><path d="m16 2 6 6"/><path d="M12 16H4"/></svg>""",
    "upload": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>""",
    "compute": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/><path d="m15 9 3 3-3 3"/><path d="m9 9-3 3 3 3"/></svg>""",
    "result": """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>""",
    "dna": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="M17 17c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="M2 9c6.667 6 13.333 0 20 6"/><path d="M7 7c1.798-1.998 2.518-3.995 2.807-5.993"/></svg>"""
}

# -----------------------------
# 3. "Bio-Tech" Dark Theme CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Dark Background with Subtle Glow */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.15) 0px, transparent 50%);
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Metrics / Text */
    p, label, .stMarkdown {
        color: #94a3b8;
    }

    /* --- COMPONENT: TECH CARDS --- */
    .tech-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    /* Hover Glow Effect */
    .tech-card:hover {
        transform: translateY(-5px);
        border-color: #06b6d4;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
    }
    
    /* Icon Container */
    .icon-box {
        background: rgba(6, 182, 212, 0.1);
        width: 64px;
        height: 64px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
        border: 1px solid rgba(6, 182, 212, 0.2);
    }

    /* Primary Button Override */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.6);
        transform: scale(1.02);
    }

    /* Table Styling */
    [data-testid="stDataFrame"] {
        background: #1e293b;
        border-radius: 12px;
        padding: 10px;
    }
    
    /* Team Images */
    .avatar-glow {
        border-radius: 50%;
        border: 3px solid #06b6d4;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
        width: 120px;
        height: 120px;
        object-fit: cover;
        margin-bottom: 15px;
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
        with open("ATMeQ.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass
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
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
        {icons['dna']}
        <h2 style="margin:0; font-size: 24px;">ATMeQ</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 12px; color: #64748b; letter-spacing: 1px;'>NAVIGATION</p>", unsafe_allow_html=True)
    
    selected_page = st.radio(
        "Page Navigation",
        ["Home", "Run Diagnostics", "Research Team"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System Status Mockup
    st.markdown("<p style='font-size: 12px; color: #64748b; letter-spacing: 1px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("**Model**")
        if model:
            st.markdown("🟢 <span style='color:#4ade80'>Online</span>", unsafe_allow_html=True)
        else:
            st.markdown("🔴 <span style='color:#ef4444'>Offline</span>", unsafe_allow_html=True)
            
    with col_stat2:
        st.markdown("**Version**")
        st.caption("v1.2.0")

    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    st.info("💡 **Note:** Input must be VST normalized RNA-Seq data.")

# -----------------------------
# 6. Page Content
# -----------------------------

# === HOME ===
if selected_page == "Home":
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Hero
    col_text, col_img = st.columns([2, 1])
    with col_text:
        st.title("Next-Gen ALS Diagnostics")
        st.markdown("""
        <p style="font-size: 1.2rem; line-height: 1.6; color: #cbd5e1;">
            ATMeQ leverages high-dimensional transcriptomic data to detect Amyotrophic Lateral Sclerosis biomarkers with clinical-grade precision.
            Powered by advanced machine learning and variance-stabilized gene expression analysis.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Analysis Engine ➔"):
            st.toast("Switching to Diagnostics tab...", icon="⚡")

    with col_img:
        # Abstract sci-fi UI element as image
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=800&q=80", use_container_width=True)

    st.markdown("---")
    st.subheader("Workflow Architecture")
    
    # Custom Step Cards with SVG Icons
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
                <div class="icon-box">
                    {step['icon']}
                </div>
                <h4 style="color:white; margin:0 0 10px 0;">{step['title']}</h4>
                <p style="font-size:0.9em; margin:0;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# === DIAGNOSTICS ===
elif selected_page == "Run Diagnostics":
    st.title("🧬 Diagnostic Console")
    
    if model is None:
        st.error("⚠️ SYSTEM ALERT: Model file (ATMeQ.pkl) not detected.")
    else:
        # Layout: Control Panel (Left) vs Visualization (Right)
        col_ctrl, col_viz = st.columns([1, 2], gap="large")
        
        with col_ctrl:
            st.markdown("""
            <div class="tech-card">
                <h4 style="color:#06b6d4">Input Configuration</h4>
                <p style="font-size:0.85em">Require VST CSV with genes: <br>
                <code>ACTA1, ABCA4, COL6A4P2...</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=["csv"])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file, index_col=0)
                st.markdown(f"✅ **Loaded:** {len(df)} samples")
                
                required_cols = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.error(f"Missing: {missing}")
                else:
                    if st.button("INITIATE SEQUENCE", use_container_width=True):
                        X = df[required_cols].copy()
                        
                        # Scaling
                        if saved_scaler:
                            X_scaled = saved_scaler.transform(X)
                        else:
                            st.warning("⚠️ Auto-scaling active (No preset found)")
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                        # Predict
                        preds = model.predict(X_scaled)
                        probs = model.predict_proba(X_scaled)
                        st.session_state['res'] = (X, preds, probs)

        with col_viz:
            if 'res' in st.session_state:
                X_res, preds_res, probs_res = st.session_state['res']
                
                # Visuals
                top_prob = probs_res[0][1] * 100
                is_als = preds_res[0] == 1
                color = "#ef4444" if is_als else "#22c55e" # Red vs Green
                status_text = "POSITIVE" if is_als else "NEGATIVE"
                
                # Main Result Card
                st.markdown(f"""
                <div class="tech-card" style="text-align:center; border-color: {color};">
                    <h5 style="margin:0; color:#94a3b8">PRIMARY SAMPLE ANALYSIS</h5>
                    <h1 style="font-size: 3.5rem; color: {color}; margin: 10px 0;">{status_text}</h1>
                    <p>Sample ID: {X_res.index[0]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = top_prob,
                    number = {'suffix': "%", 'font': {'color': "#e2e8f0"}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "CONFIDENCE SCORE", 'font': {'size': 14, 'color': "#94a3b8"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(30, 41, 59, 0)",
                        'borderwidth': 2,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 100], 'color': "rgba(30, 41, 59, 0.5)"}],
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Data Table
                res_df = pd.DataFrame({
                    "Sample ID": X_res.index,
                    "Prediction": np.where(preds_res == 1, "ALS Positive", "Control"),
                    "Confidence": probs_res[:, 1]
                })
                st.dataframe(res_df, use_container_width=True)
                
            else:
                # Empty State
                st.markdown("""
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:400px; border: 2px dashed #334155; border-radius:16px;">
                    <h3 style="color:#475569">Awaiting Data Input</h3>
                    <p>Upload CSV to visualize results</p>
                </div>
                """, unsafe_allow_html=True)

# === TEAM ===
elif selected_page == "Research Team":
    st.title("👥 Lab Members")
    
    team_members = [
        {"name": "Ahmed Saif, B.Pharm.", "role": "Graduate Researcher", "uni": "UNC Charlotte", "img": "Ahmed_Saif.png"},
        {"name": "Md Obayed Raihan, Ph.D", "role": "Assistant Professor", "uni": "Chicago State University", "img": "Obayed_Raihan.png"},
        {"name": "Bioinformatics Lead", "role": "Data Analyst", "uni": "Research Lab", "img": "ast.jpg"}
    ]
    
    cols = st.columns(len(team_members))
    
    for idx, member in enumerate(team_members):
        with cols[idx]:
            # Image logic
            if Path(member['img']).exists():
                src = f"data:image/png;base64,{get_img_as_base64(member['img'])}"
            else:
                src = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

            st.markdown(f"""
            <div class="tech-card" style="text-align:center;">
                <img src="{src}" class="avatar-glow">
                <h3 style="margin-bottom:5px;">{member['name']}</h3>
                <p style="color:#06b6d4; font-weight:bold; margin:0;">{member['role']}</p>
                <p style="font-size:0.85em; margin-top:5px;">{member['uni']}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# 7. Footer
# -----------------------------
st.markdown("""
<div style="text-align:center; margin-top:80px; padding:20px; border-top:1px solid #1e293b; color:#475569;">
    ATMeQ v1.2 | © 2025 Saif Lab | Powered by Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
