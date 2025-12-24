import streamlit as st
import pandas as pd
import numpy as np
import time
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="ATMeQ.ai - ALS Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (The "Modern" Look) ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }

    /* Navbar-like Container for Top Buttons */
    div.row-widget.stButton {
        text-align: center;
    }
    
    /* Modernize Buttons */
    .stButton > button {
        background-color: white;
        color: #475569;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        width: 100%;
    }
    
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        background-color: #eff6ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Active Button Styling (We can't target state directly in CSS easily, but we design for hover) */

    /* Card Styling */
    .css-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        border: 1px solid #f1f5f9;
    }
    
    /* Headlines */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #2563eb, #0d9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Custom Info Box */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2563eb;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Footer */
    footer {visibility: hidden;}
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem;
        margin-top: 4rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Team Images */
    .team-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 1rem auto;
        display: block;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 3px solid white;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

# --- State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# --- Navigation ---
def navbar():
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
        with col2:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = 'Home'
        with col3:
            if st.button("📊 Prediction", use_container_width=True):
                st.session_state.page = 'Prediction'
        with col4:
            if st.button("👥 Team", use_container_width=True):
                st.session_state.page = 'Team'
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# --- Page: Home ---
def home_page():
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; max-width: 800px; margin: 0 auto;">
        <span style="background-color: #dbeafe; color: #1e40af; padding: 0.5rem 1rem; border-radius: 9999px; font-weight: 600; font-size: 0.875rem;">
            Version 1.0 Now Available
        </span>
        <h1 style="font-size: 3.5rem; margin-top: 1.5rem; line-height: 1.2;">
            Advanced ALS Prediction via <br> RNA-Seq Analysis
        </h1>
        <p style="font-size: 1.25rem; color: #64748b; margin-top: 1.5rem; line-height: 1.6;">
            ATMeQ combines state-of-the-art machine learning with high-throughput genetic data to identify key gene signatures associated with Amyotrophic Lateral Sclerosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="css-card">
            <h3 style="display:flex; align-items:center; gap: 10px;">🧬 The Science</h3>
            <p style="color: #64748b; margin-top: 10px;">
                Our model targets a specific gene signature including <strong>ACTA1, ABCA4, COL6A4P2</strong>, and others. These biomarkers were selected via recursive feature elimination for maximum predictive power.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="css-card">
            <h3 style="display:flex; align-items:center; gap: 10px;">🤖 The Technology</h3>
            <p style="color: #64748b; margin-top: 10px;">
                Built on robust Random Forest algorithms optimized for high-dimensional genomic data. ATMeQ bridges the gap between complex bioinformatics and actionable clinical insights.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Steps
    st.markdown("### 🚀 How to use ATMeQ")
    st.markdown("""
    <div class="info-box">
        <strong>Step 1:</strong> Prepare your data (VST normalized CSV from DESeq2).<br>
        <strong>Step 2:</strong> Navigate to the <b>Prediction</b> tab.<br>
        <strong>Step 3:</strong> Upload your CSV file.<br>
        <strong>Step 4:</strong> Click 'Run Prediction' to get instant classification.
    </div>
    """, unsafe_allow_html=True)

# --- Page: Prediction ---
def prediction_page():
    st.markdown("<h2 style='text-align: center;'>ALS Prediction Interface</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Upload your VST normalized RNA-Seq data below.</p>", unsafe_allow_html=True)
    
    # Layout for upload
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        st.markdown("""
            <small style="color: #94a3b8;">
            Required columns: ACTA1, ABCA4, COL6A4P2, HERC2P2, KCNE4, LOC107987008
            <br>
            <a href="https://github.com/saiflab/ATMeQ/blob/main/VST%20File%20(example).csv" target="_blank">Download Example Data</a>
            </small>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, index_col=0)
            
            # Validation
            required_cols = ['ACTA1', 'ABCA4', 'COL6A4P2', 'HERC2P2', 'KCNE4', 'LOC107987008']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ File validated successfully!")
                
                with st.expander("📄 View Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)

                # Prediction Logic
                center_col1, center_col2, center_col3 = st.columns([1, 1, 1])
                with center_col2:
                    predict_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

                if predict_btn:
                    with st.spinner("Analyzing gene expression signatures..."):
                        # --- MODEL LOADING LOGIC ---
                        # In a real scenario, we load the pickle. 
                        # For this demo to be runnable without the file, we add a fallback.
                        try:
                            import pickle
                            from sklearn.preprocessing import StandardScaler
                            
                            with open('ATMeQ.pkl', 'rb') as f:
                                model = pickle.load(f)
                                
                            # Preprocess
                            X = df[required_cols]
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            predictions = model.predict(X_scaled)
                            probas = model.predict_proba(X_scaled)
                            
                            results = pd.DataFrame({
                                'Sample': X.index,
                                'Prediction': np.where(predictions == 1, 'ALS', 'Non-ALS'),
                                'ALS Probability': probas[:, 1].round(4)
                            })
                            
                        except FileNotFoundError:
                            st.warning("⚠️ Model file 'ATMeQ.pkl' not found. Running in DEMO MODE with simulated results.")
                            time.sleep(1.5) # Simulate processing time
                            
                            # Simulated Results for Demo
                            sim_preds = np.random.choice(['ALS', 'Non-ALS'], size=len(df))
                            sim_probs = np.random.uniform(0, 1, size=len(df)).round(4)
                            
                            # Adjust probs based on prediction for realism
                            final_probs = []
                            for p, pred in zip(sim_probs, sim_preds):
                                if pred == 'ALS' and p < 0.5: final_probs.append(1-p)
                                elif pred == 'Non-ALS' and p > 0.5: final_probs.append(1-p)
                                else: final_probs.append(p)

                            results = pd.DataFrame({
                                'Sample': df.index,
                                'Prediction': sim_preds,
                                'ALS Probability': final_probs
                            })

                    # --- RESULTS DISPLAY ---
                    st.markdown("### 🎯 Analysis Results")
                    
                    # Metrics Summary
                    m_col1, m_col2, m_col3 = st.columns(3)
                    als_count = results[results['Prediction'] == 'ALS'].shape[0]
                    total = results.shape[0]
                    
                    with m_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total}</div>
                            <div class="metric-label">Total Samples</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #ef4444;">{als_count}</div>
                            <div class="metric-label">Predicted ALS</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #10b981;">{total - als_count}</div>
                            <div class="metric-label">Predicted Healthy</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Detailed Dataframe with styling
                    def color_survived(val):
                        color = '#fee2e2' if val == 'ALS' else '#dcfce7'
                        return f'background-color: {color}; color: black; font-weight: bold;'
                        
                    st.dataframe(
                        results.style.applymap(color_survived, subset=['Prediction']),
                        use_container_width=True
                    )
                    
                    # Download
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Report",
                        data=csv,
                        file_name="ATMeQ_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- Page: Team ---
def team_page():
    st.markdown("<h2 style='text-align: center;'>Meet Our Team</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; margin-bottom: 3rem;'>The researchers and data scientists behind ATMeQ</p>", unsafe_allow_html=True)
    
    # Placeholder images setup
    # In a real app, ensure these files exist or use URLs
    
    team_members = [
        {"name": "Ahmed Saif, B.Pharm.", "role": "Graduate Student", "inst": "University of Rajshahi", "initials": "AS"},
        {"name": "Md Obayed Raihan, Ph.D", "role": "Assistant Professor", "inst": "Chicago State University", "initials": "OR"},
        {"name": "Research Associate", "role": "Data Scientist", "inst": "ATMeQ Lab", "initials": "RA"}
    ]
    
    cols = st.columns(3)
    
    for idx, col in enumerate(cols):
        member = team_members[idx]
        with col:
            # We use an Avatar placeholder API if local image fails
            avatar_url = f"https://ui-avatars.com/api/?name={member['initials']}&background=random&size=128"
            
            st.markdown(f"""
            <div class="css-card" style="text-align: center;">
                <img src="{avatar_url}" class="team-img" alt="{member['name']}">
                <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">{member['name']}</h3>
                <p style="color: #3b82f6; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">{member['role']}</p>
                <p style="color: #64748b; font-size: 0.85rem;">{member['inst']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- Main App Execution ---
navbar()

if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Prediction':
    prediction_page()
elif st.session_state.page == 'Team':
    team_page()

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 ATMeQ Lab. All rights reserved.</p>
    <small>For support, contact: tamim.ahmedsaif@gmail.com</small>
</div>
""", unsafe_allow_html=True)
