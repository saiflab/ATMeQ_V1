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
    page_title="ATMeQ | ALS Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. Advanced Modern CSS
# -----------------------------
st.markdown("""
<style>
    /* --- ANIMATED BACKGROUND --- */
    .stApp {
        background: linear-gradient(-45deg, #f3f4f6, #dbeafe, #e0e7ff, #f3e8ff);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-right: 1px solid #e5e7eb;
    }

    /* --- GLASSMORPHISM CARDS --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* --- HOME STEP CARDS --- */
    .step-card {
        background: white;
        border-radius: 15px;
        padding: 25px 15px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #f0f0f0;
        height: 100%;
    }
    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-color: #3b82f6;
    }
    .step-icon {
        font-size: 40px;
        margin-bottom: 15px;
        display: inline-block;
        background: #eff6ff;
        padding: 15px;
        border-radius: 50%;
        line-height: 1;
    }

    /* --- TEAM CARDS --- */
    .team-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    .team-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.12);
    }
    .team-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 15px;
        border: 4px solid #f0f2f6;
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {
        color: #1e293b;
        font-family: sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Helpers & Resources
# -----------------------------
@st.cache_resource
def load_resources():
    """Load the model and scaler. If scaler is missing, return None."""
    model = None
    scaler = None
    
    # Load Model
    try:
        with open("ATMeQ.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        pass
    
    # Load Scaler
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass
        
    return model, scaler

def get_img_as_base64(file_path):
    """Convert local image to base64 for HTML embedding."""
    if not Path(file_path).exists():
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load resources once
model, saved_scaler = load_resources()

# -----------------------------
# 4. Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/dusk/64/000000/dna-helix.png", width=60)
    st.title("ATMeQ")
    st.caption("v1.0 • ALS Transcriptomic Model")
    
    st.markdown("---")
    
    selected_page = st.radio(
        "Menu",
        ["Home", "Prediction Analysis", "Research Team"],
        index=0,
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** This tool requires VST normalized RNA-Seq data.")
    st.markdown("Developed by **Ahmed Saif**")

# -----------------------------
# 5. Page Logic
# -----------------------------

# === HOME PAGE ===
if selected_page == "Home":
    # Hero Section
    st.markdown("<div style='text-align: center; padding: 20px 0;'>", unsafe_allow_html=True)
    st.title("ATMeQ: Precision ALS Diagnostics")
    st.markdown("<h4 style='color: #64748b; font-weight: normal;'>Advanced Machine Learning for Transcriptomic Biomarker Detection</h4>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_hero1, col_hero2 = st.columns([1.5, 1])
    with col_hero1:
        st.markdown("""
        <div class="glass-card">
            <h3>🧬 About the Tool</h3>
            <p style="font-size: 1.1em; line-height: 1.6; color: #334155;">
                Welcome to <b>ATMeQ</b>. This application utilizes a Support Vector Machine (SVM) model trained on high-throughput RNA-Seq data to distinguish between Amyotrophic Lateral Sclerosis (ALS) samples and controls.
            </p>
            <ul>
                <li><b>Input:</b> Variance Stabilized Transformed (VST) counts.</li>
                <li><b>Targets:</b> Key genes including ACTA1, ABCA4, and COL6A4P2.</li>
                <li><b>Output:</b> Clinical-grade probability scores.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_hero2:
        logo_path = "logo.png"
        if Path(logo_path).exists():
            st.image(logo_path, use_container_width=True)
        else:
            # Fallback online illustration
            st.image("https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=800&q=80", 
                     caption="Genomic Analysis", use_container_width=True)

    st.markdown("---")
    
    # "How It Works" Section
    st.subheader("⚙️ How It Works")
    
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    
    steps = [
        {"icon": "🧪", "title": "1. Prepare", "desc": "Normalize your raw counts using DESeq2 (VST)."},
        {"icon": "☁️", "title": "2. Upload", "desc": "Upload your .csv file to the secure dashboard."},
        {"icon": "⚡", "title": "3. Compute", "desc": "Our ML model analyzes gene signatures instantly."},
        {"icon": "📊", "title": "4. Result", "desc": "Get Probability scores & Diagnostic status."},
    ]
    
    # Render Steps
    for col, step in zip([step_col1, step_col2, step_col3, step_col4], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-icon">{step['icon']}</div>
                <h4 style="margin:0;">{step['title']}</h4>
                <p style="color: #64748b; font-size: 0.9em; margin-top: 10px;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# === PREDICTION PAGE ===
elif selected_page == "Prediction Analysis":
    
    # Header Image (Fixed: removed 'height' parameter to prevent crash)
    st.image(
        "https://images.unsplash.com/photo-1579165466741-7f35a4755657?auto=format&fit=crop&w=1200&h=400&q=80",
        use_container_width=True
    )
    
    st.title("🔬 Diagnostics Interface")
    
    if model is None:
        st.error("⚠️ Model file (`ATMeQ.pkl`) missing. Please upload it to the app directory.")
    else:
        st.markdown("""
        <div class="glass-card">
            <b>Instructions:</b> Please ensure your CSV file follows the exact column structure required by the model. 
            The file must contain the following genes: <code>ACTA1, ABCA4, COL6A4P2, HERC2P2, KCNE4, LOC107987008</code>.
        </div>
        """, unsafe_allow_html=True)
        
        col_input, col_viz = st.columns([1, 1.5], gap="large")
        
        with col_input:
            st.subheader("1. Import Data")
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Limit 200MB per file")
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file, index_col=0)
                st.success(f"Loaded {len(df)} samples")
                
                # Validation
                required_cols = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                else:
                    X = df[required_cols].copy()
                    
                    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                        # Scaling
                        if saved_scaler:
                            X_scaled = saved_scaler.transform(X)
                        else:
                            st.warning("⚠️ 'scaler.pkl' not found. Fitting scaler on uploaded data (experimental).")
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                        # Predict
                        preds = model.predict(X_scaled)
                        probs = model.predict_proba(X_scaled)
                        
                        # Store in session state to persist visuals
                        st.session_state['results'] = (X, preds, probs)

        with col_viz:
            st.subheader("2. Analysis Results")
            
            if 'results' in st.session_state:
                X_res, preds_res, probs_res = st.session_state['results']
                
                # Creating a clean results table
                res_df = pd.DataFrame({
                    "Sample ID": X_res.index,
                    "Status": np.where(preds_res == 1, "ALS Positive", "Healthy Control"),
                    "Confidence (%)": np.round(probs_res[:, 1] * 100, 2)
                })
                
                # Show top result visually (Gauge Chart)
                top_prob = probs_res[0][1] * 100
                is_positive = preds_res[0] == 1
                color_hex = "#ef4444" if is_positive else "#22c55e"
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = top_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Sample: {X_res.index[0]}<br><span style='font-size:0.8em;color:gray'>Probability of ALS</span>"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color_hex},
                        'steps': [
                            {'range': [0, 50], 'color': "#f0fdf4"},
                            {'range': [50, 100], 'color': "#fef2f2"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=0), paper_bgcolor="rgba(0,0,0,0)", font={'family': "sans-serif"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Table below
                st.dataframe(
                    res_df.style.background_gradient(cmap="Reds", subset=["Confidence (%)"]), 
                    use_container_width=True,
                    height=200
                )
                
                # Download Button
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Report",
                    data=csv,
                    file_name="ATMeQ_Results.csv",
                    mime="text/csv"
                )
            else:
                st.info("Waiting for data submission...")
                # Placeholder image
                st.image("https://cdn.dribbble.com/users/2008861/screenshots/12558571/media/2529241b272f7dfc0903328229f3d67f.png?compress=1&resize=800x600", 
                         caption="Ready for Analysis", width=300)

# === TEAM PAGE ===
elif selected_page == "Research Team":
    st.title("👥 The Team")
    st.markdown("Meet the researchers behind the ATMeQ project.")
    
    team_data = [
        {"name": "Ahmed Saif, B.Pharm.", "role": "Graduate Student | UNC Charlotte", "uni": "University of Rajshahi (Alumni)", "img": "Ahmed_Saif.png"},
        {"name": "Md Obayed Raihan, Ph.D", "role": "Assistant Professor", "uni": "Chicago State University", "img": "Obayed_Raihan.png"},
        {"name": "Research Analyst", "role": "Bioinformatics Lead", "uni": "Research Lab", "img": "ast.jpg"}
    ]
    
    cols = st.columns(len(team_data))
    for idx, member in enumerate(team_data):
        with cols[idx]:
            # Load local image if exists, else fallback icon
            if Path(member['img']).exists():
                img_src = f"data:image/png;base64,{get_img_as_base64(member['img'])}"
            else:
                img_src = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            
            st.markdown(f"""
            <div class="team-card">
                <img src="{img_src}" class="team-img">
                <h3>{member['name']}</h3>
                <div style="color:#3b82f6; font-weight:bold; margin-bottom:5px;">{member['role']}</div>
                <div style="color:#64748b; font-size:0.9em;">{member['uni']}</div>
            </div>
            """, unsafe_allow_html=True)
