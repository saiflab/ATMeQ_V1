import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import base64
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="ALS Prediction App", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def get_base64_image(image_path: str) -> str:
    """Return base64 string for an image file. Raise FileNotFoundError if missing."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def safe_base64_image(image_path: str, fallback_b64: str) -> str:
    """Return base64 image src if file exists, else fallback data URI."""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        image_b64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{image_b64}"
    except FileNotFoundError:
        return fallback_b64

# -----------------------------
# Global CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #fdfbfb, #ebedee);
  background-attachment: fixed;
  background-size: cover;
}
.stButton button {
  font-size: 24px !important;
  padding: 20px !important;
  width: 100% !important;
  border-radius: 10px !important;
  transition: transform 0.2s, box-shadow 0.2s;
}
.stButton button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.header-container {
  text-align: center;
  margin: 0 auto;
  padding: 10px 0;
}
.app-description {
  text-align: center;
  margin: 40px auto;
  max-width: 1000px;
  line-height: 1.8;
  color: #4a4a4a;
  padding: 30px;
  background-color: rgba(245, 245, 245, 0.85);
  border-radius: 15px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.team-member {
  margin: 20px;
  padding: 20px;
  text-align: center;
  transition: transform 0.3s;
  border-radius: 10px;
  background-color: rgba(255, 255, 255, 0.9);
}
.team-member:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}
.team-image {
  border-radius: 50%;
  width: 180px;
  height: 180px;
  object-fit: cover;
  margin-bottom: 15px;
  border: 3px solid #fff;
  box-shadow: 0 3px 10px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize session state early
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# -----------------------------
# Navigation bar
# -----------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🏠 Home", key="home_btn", use_container_width=True):
        st.session_state.page = "Home"
with col2:
    if st.button("📊 Prediction", key="prediction_btn", use_container_width=True):
        st.session_state.page = "Prediction"
with col3:
    if st.button("👥 Team", key="team_btn", use_container_width=True):
        st.session_state.page = "Team"

st.markdown('<hr style="border: 2px solid #0078ff; border-radius: 5px;">', unsafe_allow_html=True)

# -----------------------------
# HOME
# -----------------------------
if st.session_state.page == "Home":
    # Safer: don't crash if logo is missing
    logo_path = "logo.png"
    if Path(logo_path).exists():
        logo_base64 = get_base64_image(logo_path)
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="400" style="display: block; margin: 0 auto;">'
    else:
        st.warning("logo.png not found in the app folder. (Home page will still run.)")
        logo_html = ""

    st.markdown("""
    <style>
    .header-container {
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 20px;
    }
    .app-description {
      text-align: center;
      max-width: 600px;
      margin: 20px auto;
    }
    .app-description ul {
      list-style-type: none;
      padding-left: 0;
    }
    .app-description li {
      text-align: left;
      margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="header-container">
      <h2 style="margin: 15px 0 0 0; padding: 0; text-align: center;">
        ATMeQ (ALS Prediction Tool using Machine Learning and RNA-Seq) version 1.0
      </h2>
      {logo_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-description">
      <p>Welcome to ATMeQ, an ALS Prediction Application. This tool utilizes a machine learning model trained on RNA-Seq data to accurately predict ALS status.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
---
## Introduction

Welcome to ATMeQ (ALS Prediction Tool using Machine Learning and RNA-Seq), a tool designed to predict Amyotrophic Lateral Sclerosis (ALS) using a machine learning model trained on RNA-Seq gene expression features.

Upload a .csv file containing variance-stabilized transformation (VST) output from DESeq2, and ATMeQ will generate predictions and probabilities.

## Steps to Use ATMeQ

1. **Data Preparation:** Generate a CSV file with VST data from DESeq2.  
2. **Accessing the Application:** Open the app in a browser.  
3. **File Upload:** Upload your prepared CSV file.  
4. **Make Prediction:** Click “Run Prediction”.  
5. **Review Results:** View predictions and download as CSV.

## Contact
If you have questions, email: tamim.ahmedsaif@gmail.com
---
""")

# -----------------------------
# PREDICTION
# -----------------------------
elif st.session_state.page == "Prediction":
    st.title("ALS Prediction Interface")
    st.markdown("""Here the [example file](https://github.com/saiflab/ATMeQ/blob/main/VST%20File%20(example).csv)""")

    # Load model
    try:
        with open("ATMeQ.pkl", "rb") as f:
            ATMeQ_model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found! Please ensure ATMeQ.pkl is in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload RNA-Seq CSV file",
        type=["csv"],
        help="Upload your dataset in CSV format containing gene expression data",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, index_col=0)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.write("Data preview:")
        st.dataframe(df.head(), use_container_width=True)

        required_cols = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        X = df[required_cols].copy()

        # IMPORTANT NOTE:
        # Ideally you should load the same scaler used during training (pickle it),
        # not fit a new one on user data. Keeping your behavior, but safer code below.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if st.button("🚀 Run Prediction", use_container_width=True):
            with st.spinner("Running predictions..."):
                try:
                    predictions = ATMeQ_model.predict(X_scaled)
                    probas = ATMeQ_model.predict_proba(X_scaled)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            results = pd.DataFrame({
                "Sample": X.index,
                "Prediction": np.where(predictions == 1, "ALS", "Non-ALS"),
                "ALS Probability": np.round(probas[:, 1], 4),
            })

            st.success(f"Predictions generated for {len(results)} samples!")
            st.dataframe(results, use_container_width=True)

            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="ATMeQ_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# -----------------------------
# TEAM
# -----------------------------
elif st.session_state.page == "Team":
    st.title("Our Expert Team")

    st.markdown("""
    <style>
    .team-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 2rem;
      padding: 2rem 0;
    }
    .team-member {
      text-align: center;
      padding: 1.5rem;
      border-radius: 12px;
      background: #f8f9fa;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .team-member:hover {
      transform: translateY(-5px);
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .team-member img {
      border-radius: 50%;
      width: 180px;
      height: 180px;
      object-fit: cover;
      border: 3px solid #fff;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      margin-bottom: 1rem;
    }
    .team-member h3 {
      margin: 1rem 0 0.5rem;
      color: #333;
    }
    .team-member p {
      margin: 0;
      color: #666;
      font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    team_members = [
        {"name": "Ahmed Saif, B.Pharm.", "image_path": "Ahmed_Saif.png",
         "role": "Graduate Student, Department of Pharmacy, University of Rajshahi"},
        {"name": "Md Obayed Raihan, Ph.D", "image_path": "Obayed_Raihan.png",
         "role": "Assistant Professor, Department of Pharmaceutical Science, Chicago State University"},
        {"name": "Other Member", "image_path": "ast.jpg", "role": "Research Analyst"},
    ]

    placeholder_b64 = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAMAAADQmPP/AAAAA1BMVEX///+nxBvIAAAAR0lEQVR4nO3BAQ0AAADCoPdPbQ43oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIBLcQ8AAa0jZQAAAABJRU5ErkJggg=="
    )

    st.markdown('<div class="team-grid">', unsafe_allow_html=True)

    for member in team_members:
        image_src = safe_base64_image(member["image_path"], placeholder_b64)

        st.markdown(f"""
        <div class="team-member">
          <img src="{image_src}" alt="{member['name']}">
          <h3>{member['name']}</h3>
          <p>{member['role']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
