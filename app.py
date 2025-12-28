import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from pathlib import Path
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Configuration & Global Settings
# -----------------------------
st.set_page_config(
    page_title="ATMeQ | ALS Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. Modern CSS (Glassmorphism & Clean UI)
# -----------------------------
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* Cards / Containers (Glassmorphism) */
    .css-1r6slb0, .stDataFrame, .stPlotlyChart {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    /* Custom Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(45deg, #2193b0, #6dd5ed);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33, 147, 176, 0.3);
    }

    /* Team Member Card */
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
        transform: translateY(-10px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .team-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 15px;
        border: 4px solid #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Helper Functions & Caching
# -----------------------------
@st.cache_resource
def load_resources():
    """Load Model and Scaler efficiently with caching."""
    model = None
    scaler = None
    
    # Load Model
    try:
        with open("ATMeQ.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        pass # Handle in UI
        
    # Load Scaler (Best Practice: Use the same scaler from training)
    # If not found, we will fit_transform (fallback)
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass
        
    return model, scaler

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Initialize Resources
model, saved_scaler = load_resources()

# -----------------------------
# 4. Navigation System
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/dusk/64/000000/dna-helix.png", width=50) 
    st.title("ATMeQ")
    st.caption("ALS Transcriptomic Model")
    
    selected_page = st.radio(
        "Navigate",
        ["Home", "Prediction Analysis", "Research Team"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Ensure your CSV input is VST normalized.")
    st.markdown("Created by **Ahmed Saif**")

# -----------------------------
# 5. Page Logic
# -----------------------------

# === HOME PAGE ===
if selected_page == "Home":
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.title("Predicting ALS with Precision")
        st.markdown("""
        ### Welcome to ATMeQ v1.0
        
        **ATMeQ** (ALS Prediction Tool using Machine Learning and RNA-Seq) leverages advanced gene expression patterns to identify Amyotrophic Lateral Sclerosis biomarkers.
        
        #### 🚀 Key Features
        * **High Accuracy:** Trained on extensive RNA-Seq datasets.
        * **Fast Analysis:** Instant processing of VST normalized data.
        * **Secure:** All processing happens locally in your session.
        """)
        
        if st.button("Start Analysis ➔"):
            st.toast("Please switch to the 'Prediction Analysis' tab!")
            
    with col2:
        # Placeholder for a hero image or logo
        logo_path = "logo.png"
        if Path(logo_path).exists():
            st.image(logo_path, use_container_width=True)
        else:
            # Fallback visuals if no logo
            st.markdown("""
            <div style="background-color:white; padding:40px; border-radius:20px; text-align:center;">
                <h1 style="font-size: 80px;">🧬</h1>
                <p style="color:gray;">Upload. Analyze. Predict.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Steps Container
    st.subheader("How it works")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("#### 1. Prepare")
    s1.caption("Get your VST data from DESeq2.")
    s2.markdown("#### 2. Upload")
    s2.caption("Upload CSV to the app.")
    s3.markdown("#### 3. Compute")
    s3.caption("ML model analyzes signatures.")
    s4.markdown("#### 4. Result")
    s4.caption("Get Probability & Status.")

# === PREDICTION PAGE ===
elif selected_page == "Prediction Analysis":
    st.title("🧬 Analysis Interface")
    
    if model is None:
        st.error("⚠️ Model file (`ATMeQ.pkl`) not found. Please upload it to the directory.")
    else:
        col_upload, col_preview = st.columns([1, 2], gap="medium")
        
        with col_upload:
            st.markdown("### Upload Data")
            uploaded_file = st.file_uploader("Drag and drop CSV file", type=["csv"])
            st.caption("[Download Example CSV](https://github.com/saiflab/ATMeQ)")

        if uploaded_file:
            df = pd.read_csv(uploaded_file, index_col=0)
            
            with col_preview:
                st.markdown("### Data Preview")
                st.dataframe(df.head(3), use_container_width=True, height=150)

            # Validation
            required_cols = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
            missing = [c for c in required_cols if c not in df.columns]
            
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                X = df[required_cols].copy()
                
                # Scaling Logic
                if saved_scaler:
                    X_scaled = saved_scaler.transform(X)
                else:
                    # Fallback warning
                    st.warning("⚠️ No 'scaler.pkl' found. Fitting scaler on uploaded data (experimental).")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                # Action Button
                if st.button("🚀 Run Diagnostics", type="primary", use_container_width=True):
                    with st.spinner("Analyzing Gene Signatures..."):
                        preds = model.predict(X_scaled)
                        probs = model.predict_proba(X_scaled)

                    # Results Dashboard
                    st.markdown("### 📊 Diagnostic Results")
                    
                    # Create Tabs for different views
                    tab1, tab2 = st.tabs(["Summary View", "Detailed Table"])
                    
                    with tab1:
                        # We will show the result for the first sample as a highlight, 
                        # or a summary if multiple samples.
                        for i, (idx, row) in enumerate(X.iterrows()):
                            p_score = probs[i][1]
                            is_als = preds[i] == 1
                            
                            c1, c2 = st.columns([1, 2])
                            
                            with c1:
                                status_color = "red" if is_als else "green"
                                status_text = "ALS DETECTED" if is_als else "NEGATIVE"
                                st.markdown(f"""
                                <div style="text-align:center; padding:20px; background:white; border-radius:10px; border-left: 10px solid {status_color};">
                                    <h4 style="margin:0;">Sample: {idx}</h4>
                                    <h2 style="color:{status_color}; margin:10px 0;">{status_text}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            with c2:
                                # Plotly Gauge
                                fig = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = p_score * 100,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "ALS Probability (%)"},
                                    gauge = {
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "#ff4b4b" if is_als else "#00c853"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "rgba(0, 200, 83, 0.2)"},
                                            {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.2)"}],
                                    }
                                ))
                                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()

                    with tab2:
                        results_df = pd.DataFrame({
                            "Sample ID": X.index,
                            "Prediction": np.where(preds == 1, "ALS", "Control"),
                            "Confidence Score": np.round(probs[:, 1], 4)
                        })
                        
                        # Apply highlighting
                        def highlight_als(val):
                            return 'background-color: #ffcccc' if val == "ALS" else 'background-color: #ccffcc'
                        
                        st.dataframe(results_df.style.applymap(highlight_als, subset=['Prediction']), use_container_width=True)
                        
                        # Download
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download Report", csv, "ATMeQ_Results.csv", "text/csv")


# === TEAM PAGE ===
elif selected_page == "Research Team":
    st.title("👥 Our Team")
    st.markdown("Meet the minds behind ATMeQ.")
    
    # Modern Responsive Grid using Columns
    team_data = [
        {
            "name": "Ahmed Saif, B.Pharm.",
            "role": "Graduate Student | UNC Charlotte",
            "uni": "University of Rajshahi (Alumni)",
            "img": "Ahmed_Saif.png"
        },
        {
            "name": "Md Obayed Raihan, Ph.D",
            "role": "Assistant Professor",
            "uni": "Chicago State University",
            "img": "Obayed_Raihan.png"
        },
        {
            "name": "Research Analyst",
            "role": "Data Science Lead",
            "uni": "Bioinformatics Lab",
            "img": "ast.jpg"
        }
    ]
    
    cols = st.columns(len(team_data))
    
    for idx, member in enumerate(team_data):
        with cols[idx]:
            # Fallback image logic
            if Path(member['img']).exists():
                img_src = f"data:image/png;base64,{get_img_as_base64(member['img'])}"
            else:
                img_src = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

            st.markdown(f"""
            <div class="team-card">
                <img src="{img_src}" class="team-img">
                <h3>{member['name']}</h3>
                <p style="color:#2193b0; font-weight:bold;">{member['role']}</p>
                <p style="color:#7f8c8d; font-size:0.9em;">{member['uni']}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div style="text-align:center; margin-top:50px; color:#bdc3c7; font-size:0.8em;">
    &copy; 2025 ATMeQ Lab. All rights reserved. <br>
    Built with Streamlit & Python.
</div>
""", unsafe_allow_html=True)
