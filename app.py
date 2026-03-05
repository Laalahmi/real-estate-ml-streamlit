# app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import MODEL_PATH
from src.logger import setup_logger

logger = setup_logger()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Modern CSS + FIX “title cut” issue
# (Increase top padding + remove Streamlit default top offset issues)
# ---------------------------
st.markdown(
    """
    <style>
      /* Fix top cropping: add more top padding */
      .block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 2rem !important;
      }

      /* Slightly reduce any unexpected margins */
      h1, h2, h3 { margin-top: 0.2rem !important; }

      .card {
        background: rgba(0,0,0,0.035);
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 18px;
        padding: 18px;
      }
      .subtle {
        color: rgba(0,0,0,0.65);
        font-size: 0.95rem;
      }
      .title {
        font-size: 2.05rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.15;
      }
      .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.10);
        font-size: 0.85rem;
        margin-right: 0.35rem;
        margin-top: 0.2rem;
      }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load model bundle (cached)
# ---------------------------
@st.cache_resource
def load_bundle():
    logger.info(f"Loading model bundle from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]          # must match training exactly
    metrics = bundle.get("metrics", {})
    logger.info("Model bundle loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model bundle: {e}")
    st.error(
        "❌ Could not load the trained model.\n\n"
        "Make sure you trained the model and that the file exists:\n"
        f"`{MODEL_PATH}`\n\n"
        "Run: `python -m src.train`"
    )
    st.stop()

# ---------------------------
# Header with logo + credits
# ---------------------------
logo_path = Path("assets/algonquin_logo.png")

hcol1, hcol2 = st.columns([0.18, 0.82], vertical_alignment="center")
with hcol1:
    if logo_path.exists():
        try:
            img = Image.open(logo_path)
            st.image(img, use_container_width=True)
        except UnidentifiedImageError:
            st.warning("Logo file exists but is not a valid image. Replace `assets/algonquin_logo.png`.")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")
    else:
        st.caption("📌 Add `assets/algonquin_logo.png` to show the logo")

with hcol2:
    st.markdown('<p class="title">🏠 Real Estate Price Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <span class="badge">CST2216</span>
        <span class="badge">Modularizing & Deploying ML Code</span>
        <span class="badge">Streamlit App</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtle'>Predict house prices using a trained ML model (Linear Regression + MinMax scaling).</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtle'><b>Created by:</b> Mohammed Laalahmi &nbsp; • &nbsp; "
        "<b>Instructor:</b> Dr. Umer Altaf &nbsp; • &nbsp; <b>Algonquin College</b></p>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------
# Sidebar inputs (ALL inputs here)
# - No max values (as requested)
# - hasGuestRoom is numeric (as requested)
# ---------------------------
st.sidebar.title("🏡 House Inputs")
st.sidebar.caption("Enter values below, then click **Predict Price**.")

def yesno(label: str, default: str = "No") -> int:
    choice = st.sidebar.selectbox(label, ["No", "Yes"], index=0 if default == "No" else 1)
    return 1 if choice == "Yes" else 0

with st.sidebar.expander("📏 Size & Rooms", expanded=True):
    squareMeters = st.sidebar.number_input("Square meters (squareMeters)", min_value=0.0, value=100.0)
    numberOfRooms = st.sidebar.number_input("Number of rooms (numberOfRooms)", min_value=0, step=1, value=3)
    floors = st.sidebar.number_input("Floors (floors)", min_value=0, step=1, value=1)

with st.sidebar.expander("🏙️ Location & Ownership", expanded=True):
    cityCode = st.sidebar.number_input("City code (cityCode)", min_value=0, step=1, value=100)
    cityPartRange = st.sidebar.number_input("City part range (cityPartRange)", min_value=0, step=1, value=1)
    numPrevOwners = st.sidebar.number_input("Previous owners (numPrevOwners)", min_value=0, step=1, value=0)

with st.sidebar.expander("🏗️ Spaces", expanded=True):
    basement = st.sidebar.number_input("Basement size (basement)", min_value=0.0, value=0.0)
    attic = st.sidebar.number_input("Attic size (attic)", min_value=0.0, value=0.0)
    garage = st.sidebar.number_input("Garage size (garage)", min_value=0.0, value=0.0)

with st.sidebar.expander("🛏️ Amenities", expanded=True):
    hasYard = yesno("Has yard? (hasYard)")
    hasPool = yesno("Has pool? (hasPool)")
    isNewBuilt = yesno("Is new built? (isNewBuilt)")
    hasStormProtector = yesno("Storm protector? (hasStormProtector)")
    hasStorageRoom = yesno("Storage room? (hasStorageRoom)")

    # FIX: hasGuestRoom is numeric, not boolean
    hasGuestRoom = st.sidebar.number_input(
        "Number of guest rooms (hasGuestRoom)",
        min_value=0,
        step=1,
        value=0
    )

with st.sidebar.expander("🗓️ Build Year", expanded=True):
    year_built = st.sidebar.number_input(
        "Year built (made)",
        min_value=0,
        step=1,
        value=2000
    )
    current_year = datetime.now().year
    houseAge = max(0, current_year - int(year_built))
    st.caption(f"Computed: **houseAge = {houseAge}** (current year = {current_year})")

st.sidebar.divider()
predict_btn = st.sidebar.button("🚀 Predict Price", use_container_width=True)

st.sidebar.markdown(
    "<p class='subtle'>Credits:<br><b>Mohammed Laalahmi</b><br>"
    "<b>Dr. Umer Altaf</b> (Instructor)<br>"
    "<b>Algonquin College</b></p>",
    unsafe_allow_html=True,
)

# ---------------------------
# Build input df (must match training feature columns)
# Training columns provided (X = all except price):
# squareMeters, numberOfRooms, hasYard, hasPool, floors, cityCode, cityPartRange,
# numPrevOwners, isNewBuilt, hasStormProtector, basement, attic, garage, hasStorageRoom,
# hasGuestRoom, houseAge
# ---------------------------
data = {
    "squareMeters": float(squareMeters),
    "numberOfRooms": int(numberOfRooms),
    "hasYard": int(hasYard),
    "hasPool": int(hasPool),
    "floors": int(floors),
    "cityCode": int(cityCode),
    "cityPartRange": int(cityPartRange),
    "numPrevOwners": int(numPrevOwners),
    "isNewBuilt": int(isNewBuilt),
    "hasStormProtector": int(hasStormProtector),
    "basement": float(basement),
    "attic": float(attic),
    "garage": float(garage),
    "hasStorageRoom": int(hasStorageRoom),
    "hasGuestRoom": int(hasGuestRoom),
    "houseAge": int(houseAge),
}

input_df = pd.DataFrame([data])

# Enforce exact training order saved in the model bundle
try:
    input_df = input_df[feature_names]
except Exception as e:
    logger.exception(f"Feature mismatch: {e}")
    st.error(
        "❌ Feature mismatch between the app and the trained model.\n\n"
        "Your model expects these columns:\n"
        f"{feature_names}\n\n"
        "Re-train the model or update the app inputs to match."
    )
    st.stop()

# ---------------------------
# Main tabs
# ---------------------------
tab_pred, tab_info, tab_about = st.tabs(["🔮 Prediction", "📊 Model Info", "ℹ️ About"])

with tab_pred:
    left, right = st.columns([1.2, 1], vertical_alignment="top")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if predict_btn:
            try:
                logger.info("Prediction requested.")
                scaled = scaler.transform(input_df)
                pred = float(model.predict(scaled)[0])
                st.success(f"Estimated House Price: **${pred:,.2f}**")
                logger.info(f"Prediction generated: {pred}")
            except Exception as e:
                logger.exception(f"Prediction failed: {e}")
                st.error("❌ Prediction failed. Check `logs/app.log` for details.")
        else:
            st.info("Enter values in the sidebar, then click **Predict Price**.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
        st.subheader("Quick Tips")
        st.write(
            "- `houseAge` is computed automatically from the build year.\n"
            "- Inputs are scaled using the same scaler saved during training.\n"
            "- If predictions look odd, try more realistic values."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Summary")
        st.caption("This is the exact feature row sent to the model.")
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Training Metrics (Saved)")

    c1, c2, c3 = st.columns(3)
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    r2 = metrics.get("r2")

    c1.metric("RMSE", f"{rmse:.3f}" if rmse is not None else "N/A")
    c2.metric("MAE", f"{mae:.3f}" if mae is not None else "N/A")
    c3.metric("R²", f"{r2:.3f}" if r2 is not None else "N/A")

    st.divider()
    st.write(
        "**Model:** Linear Regression  \n"
        "**Scaler:** MinMaxScaler  \n"
        "**Inputs:** 16 features (including numeric `hasGuestRoom` and engineered `houseAge`)  \n"
        f"**Artifact:** `{MODEL_PATH}`"
    )

    with st.expander("Show feature list (training order)"):
        st.write(feature_names)

    st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About This Project")
    st.write(
        "This Streamlit app is part of **CST2216** at **Algonquin College**. "
        "The original notebook was modularized into reusable Python modules (`src/`) "
        "for data loading, feature engineering, training, evaluation, and deployment."
    )
    st.write("**Credits:** Mohammed Laalahmi • Dr. Umer Altaf • Algonquin College")
    st.caption("If the model file is missing, run training: `python -m src.train`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<hr/>"
    "<p class='subtle'>© 2026 • Mohammed Laalahmi • CST2216 • Dr. Umer Altaf • Algonquin College</p>",
    unsafe_allow_html=True,
)