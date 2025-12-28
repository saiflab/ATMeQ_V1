import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import base64
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(
    page_title="ATMeQ • ALS Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# Helpers
# ==========================================================
def b64_img(path: str) -> str:
    """Return base64 data URI for an image file; return empty string if missing."""
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("utf-8")

# ==========================================================
# Modern CSS (Full)
# ==========================================================
st.markdown(
    """
<style>
/* ---------- Global theme ---------- */
:root{
  --bg1:#0b1220;
  --bg2:#0d1b2a;
  --card: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --muted2: rgba(255,255,255,0.55);
  --accent:#6ee7ff;
  --accent2:#a78bfa;
}

.stApp{
  background: radial-gradient(1200px 600px at 10% 10%, rgba(110,231,255,0.20), transparent 55%),
              radial-gradient(1000px 600px at 90% 20%, rgba(167,139,250,0.18), transparent 60%),
              linear-gradient(160deg, var(--bg1), var(--bg2));
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.block-container{
  padding-top: 5.0rem;
  padding-bottom: 2.5rem;
  max-width: 1200px;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Headings */
h1, h2, h3, h4{
  letter-spacing: -0.25px !important;
}

/* Links */
a{ color: var(--accent); text-decoration: none; }
a:hover{ text-decoration: underline; }

/* ---------- Top Nav ---------- */
.topnav{
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 999;
  backdrop-filter: blur(14px);
  background: rgba(15, 23, 42, 0.55);
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.nav-inner{
  max-width: 1200px;
  margin: 0 auto;
  padding: 0.85rem 1.2rem;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 1rem;
}
.brand{
  display:flex;
  align-items:center;
  gap: 0.7rem;
}
.brand .logo{
  width: 36px; height: 36px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(110,231,255,0.9), rgba(167,139,250,0.9));
  box-shadow: 0 10px 30px rgba(110,231,255,0.12);
}
.brand .title{
  font-weight: 800;
  letter-spacing: -0.2px;
  color: var(--text);
  font-size: 1.05rem;
}
.brand .subtitle{
  color: var(--muted2);
  font-size: 0.85rem;
  margin-top: -2px;
}

/* ---------- Buttons ---------- */
.stButton > button{
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.08) !important;
  color: var(--text) !important;
  border-radius: 14px !important;
  padding: 0.85rem 1.1rem !important;
  font-weight: 700 !important;
  transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 10px 28px rgba(0,0,0,0.35);
  background: rgba(255,255,255,0.11) !important;
}
.stButton > button:focus{
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(110,231,255,0.20) !important;
}

/* ---------- Cards ---------- */
.card{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.06);
  backdrop-filter: blur(12px);
  border-radius: 18px;
  padding: 1.25rem 1.25rem;
  box-shadow: 0 18px 60px rgba(0,0,0,0.25);
}
.small{
  color: var(--muted2);
  font-size: 0.95rem;
  line-height: 1.6;
}
.section-title{
  font-size: 1.05rem;
  font-weight: 900;
  letter-spacing: -0.3px;
  margin-bottom: 0.25rem;
}
.section-sub{
  color: rgba(255,255,255,0.72);
  font-size: 0.95rem;
  line-height: 1.5;
  margin-bottom: 0.2rem;
}

/* ---------- KPI ---------- */
.kpi{
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.07);
  border-radius: 16px;
  padding: 1rem 1rem;
}
.kpi .label{ color: var(--muted2); font-size: 0.85rem; }
.kpi .value{ color: var(--text); font-weight: 900; font-size: 1.4rem; margin-top: 0.25rem; }
.kpi .hint{ color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }

/* ---------- DataFrame ---------- */
[data-testid="stDataFrame"]{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
}

/* ---------- File uploader typography ---------- */
div[data-testid="stFileUploader"]{
  border-radius: 18px !important;
}
div[data-testid="stFileUploader"] section{
  border-radius: 18px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.06) !important;
  padding: 14px 14px 10px 14px !important;
}
/* Label: "Drop VST Data File Here" */
div[data-testid="stFileUploader"] label{
  font-size: 1.02rem !important;
  font-weight: 900 !important;
  letter-spacing: -0.2px !important;
  color: rgba(255,255,255,0.92) !important;
  margin-bottom: 6px !important;
}
/* Help text under label */
div[data-testid="stFileUploader"] small{
  font-size: 0.92rem !important;
  color: rgba(255,255,255,0.70) !important;
}
/* Inner drop zone */
div[data-testid="stFileUploader"] div[role="button"]{
  border-radius: 16px !important;
}
div[data-testid="stFileUploader"] div[role="button"] p{
  font-size: 0.95rem !important;
  color: rgba(255,255,255,0.78) !important;
  font-weight: 650 !important;
}
/* “No file chosen” + limit text */
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] div{
  font-size: 0.95rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# Session state
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==========================================================
# Sticky top nav
# ==========================================================
st.markdown(
    """
<div class="topnav">
  <div class="nav-inner">
    <div class="brand">
      <div class="logo"></div>
      <div>
        <div class="title">ATMeQ</div>
        <div class="subtitle">ALS Prediction • RNA-seq + ML</div>
      </div>
    </div>
    <div class="small" style="color: rgba(255,255,255,0.65);">v1.0</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# Navigation buttons
# ==========================================================
c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
with c1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
with c2:
    if st.button("📊 Prediction", use_container_width=True):
        st.session_state.page = "Prediction"
with c3:
    if st.button("👥 Team", use_container_width=True):
        st.session_state.page = "Team"

st.write("")

# ==========================================================
# HOME
# ==========================================================
if st.session_state.page == "Home":
    logo_src = b64_img("logo.png")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
<div class="card">
  <h2>ATMeQ — ALS Prediction Tool</h2>
  <div class="small">
    ATMeQ (ALS Prediction Tool using Machine Learning and RNA-Seq) predicts ALS status
    from DESeq2 VST gene expression values using a trained machine learning model.
  </div>
  <br/>
  <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
    <span class="small" style="padding:0.35rem 0.65rem; border-radius:999px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.07);">RNA-seq (DESeq2 VST)</span>
    <span class="small" style="padding:0.35rem 0.65rem; border-radius:999px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.07);">6-gene signature</span>
    <span class="small" style="padding:0.35rem 0.65rem; border-radius:999px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.07);">Probability output</span>
    <span class="small" style="padding:0.35rem 0.65rem; border-radius:999px; border:1px solid rgba(255,255,255,0.15); background:rgba(255,255,255,0.07);">CSV export</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            """
<div class="card">
  <h3>How to use</h3>
  <ol class="small">
    <li><b>Prepare data:</b> Export a CSV containing DESeq2 VST values (rows = samples, columns = genes).</li>
    <li><b>Go to Prediction:</b> Upload your CSV.</li>
    <li><b>Run:</b> Click <b>Run Prediction</b>.</li>
    <li><b>Review:</b> See ALS / Non-ALS + probability.</li>
    <li><b>Download:</b> Export results as CSV.</li>
  </ol>
  <div class="small">
    Example file:
    <a href="https://github.com/saiflab/ATMeQ/blob/main/VST%20File%20(example).csv" target="_blank">
      GitHub example CSV
    </a>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Quick overview</h3>", unsafe_allow_html=True)

        if logo_src:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-top:0.5rem;">
                  <img src="{logo_src}" style="width:100%; max-width:420px; border-radius:18px; border:1px solid rgba(255,255,255,0.12);" />
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("logo.png not found (optional). Add it to the app folder for branding.")

        st.markdown(
            """
<div class="small" style="margin-top:0.9rem;">
  <b>Required genes:</b><br/>
  ACTA1, ABCA4, COL6A4P2, HERC2P2, KCNE4, LOC107987008
</div>
<hr style="border:0; border-top:1px solid rgba(255,255,255,0.12); margin:1rem 0;">
<div class="small">
  <b>Contact</b><br/>
  tamim.ahmedsaif@gmail.com
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICTION
# ==========================================================
elif st.session_state.page == "Prediction":
    st.markdown(
        """
<div class="card">
  <h2>Prediction</h2>
  <div class="small">
    Upload your <b>DESeq2 VST CSV</b> and run ATMeQ to get ALS / Non-ALS predictions and probabilities.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    # Load model
    try:
        with open("ATMeQ.pkl", "rb") as f:
            ATMeQ_model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please place ATMeQ.pkl in the same folder as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Modern Input Configuration header
    st.markdown(
        """
<div class="card">
  <div class="section-title">Input Configuration</div>
  <div class="section-sub">
    <b>Required Gene Markers:</b> ACTA1, ABCA4, COL6A4P2, HERC2P2, KCNE4, LOC107987008
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader(
        "Drop VST Data File Here",
        type=["csv"],
        help="Limit 200MB per file • CSV",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, index_col=0)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        required_cols = ["ACTA1", "ABCA4", "COL6A4P2", "HERC2P2", "KCNE4", "LOC107987008"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        a, b = st.columns([1.35, 0.65], gap="large")

        with a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>Data preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>Checks</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'><b>Samples:</b> {df.shape[0]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'><b>Columns:</b> {df.shape[1]}</div>", unsafe_allow_html=True)
            st.write("")
            if missing_cols:
                st.error("Missing required genes:")
                st.write(missing_cols)
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()
            else:
                st.success("All required genes found ✅")
                st.markdown("<div class='small' style='color: rgba(255,255,255,0.70);'>Ready to run prediction.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        X = df[required_cols].copy()

        # NOTE: Best practice is to load the scaler used during training.
        # Keeping your behavior for compatibility.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.write("")
        run = st.button("🚀 Run Prediction", use_container_width=True)

        if run:
            with st.spinner("Running predictions..."):
                try:
                    preds = ATMeQ_model.predict(X_scaled)
                    probas = ATMeQ_model.predict_proba(X_scaled)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            results = pd.DataFrame({
                "Sample": X.index,
                "Prediction": np.where(preds == 1, "ALS", "Non-ALS"),
                "ALS Probability": np.round(probas[:, 1], 4),
            })

            st.write("")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>Results</h3>", unsafe_allow_html=True)

            n_samples = len(results)
            als_count = int((results["Prediction"] == "ALS").sum())
            non_count = n_samples - als_count
            mean_p = float(results["ALS Probability"].mean()) if n_samples > 0 else 0.0

            k1, k2, k3, k4 = st.columns(4, gap="medium")
            with k1:
                st.markdown(f"<div class='kpi'><div class='label'>Samples</div><div class='value'>{n_samples}</div><div class='hint'>Total predictions</div></div>", unsafe_allow_html=True)
            with k2:
                st.markdown(f"<div class='kpi'><div class='label'>ALS</div><div class='value'>{als_count}</div><div class='hint'>Predicted ALS</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='kpi'><div class='label'>Non-ALS</div><div class='value'>{non_count}</div><div class='hint'>Predicted Non-ALS</div></div>", unsafe_allow_html=True)
            with k4:
                st.markdown(f"<div class='kpi'><div class='label'>Mean ALS Prob.</div><div class='value'>{mean_p:.3f}</div><div class='hint'>Across samples</div></div>", unsafe_allow_html=True)

            st.write("")
            st.dataframe(results, use_container_width=True)

            st.write("")
            csv = results.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name="ATMeQ_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# TEAM
# ==========================================================
elif st.session_state.page == "Team":
    st.markdown(
        """
<div class="card">
  <h2>Research Team</h2>
  <div class="small">The ATMeQ project team and contributors.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    team_members = [
        {
            "name": "Ahmed Saif, B.Pharm.",
            "role": "Graduate Student, Department of Pharmacy, University of Rajshahi",
            "image": "Ahmed_Saif.png",
            "links": {}
        },
        {
            "name": "Md Obayed Raihan, Ph.D",
            "role": "Assistant Professor, Department of Pharmaceutical Science, Chicago State University",
            "image": "Obayed_Raihan.png",
            "links": {}
        },
        {
            "name": "Other Member",
            "role": "Research Analyst",
            "image": "ast.jpg",
            "links": {}
        },
    ]

    def team_card(member):
        img_src = b64_img(member["image"])
        img_html = (
            f"<img src='{img_src}' style='width:96px; height:96px; border-radius:20px; object-fit:cover; border:1px solid rgba(255,255,255,0.14);'/>"
            if img_src else
            "<div style='width:96px;height:96px;border-radius:20px;background:rgba(255,255,255,0.10);border:1px solid rgba(255,255,255,0.12);'></div>"
        )

        links = member.get("links", {})
        links_html = ""
        if links:
            parts = [f"<a href='{url}' target='_blank'>{label}</a>" for label, url in links.items() if url]
            links_html = "<div class='small' style='margin-top:0.6rem; display:flex; gap:0.9rem; flex-wrap:wrap;'>" + " ".join(parts) + "</div>"

        st.markdown(
            f"""
<div class="card" style="display:flex; gap:1rem; align-items:center; margin-bottom: 22px;">
  {img_html}
  <div style="flex:1;">
    <div style="font-weight:900; font-size:1.05rem;">{member['name']}</div>
    <div class="small">{member['role']}</div>
    {links_html}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    cols = st.columns(3, gap="large")
    for i, m in enumerate(team_members):
        with cols[i % 3]:
            team_card(m)

    st.write("")
    st.markdown(
        """
<div class="card">
  <h3>Contact</h3>
  <div class="small">For questions or collaboration: <b>tamim.ahmedsaif@gmail.com</b></div>
</div>
""",
        unsafe_allow_html=True,
    )
