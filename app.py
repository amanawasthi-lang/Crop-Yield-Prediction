import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/crop_yield_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model()


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("data/crop_yield.csv")

    data["Crop"] = data["Crop"].str.strip()
    data["Season"] = data["Season"].str.strip()
    data["State"] = data["State"].str.strip()

    return data


df = load_data()


# --------------------------------------------------
# Initialize Prediction History
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌾 Crop Yield Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict agricultural crop yield using crop, location, '
    'weather and farming information.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Project Metrics
# --------------------------------------------------

st.subheader("📊 Project Overview")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("📚 Dataset Records", f"{len(df):,}")

with metric2:
    st.metric("🌾 Crops", df["Crop"].nunique())

with metric3:
    st.metric("📍 States", df["State"].nunique())

with metric4:
    st.metric("🎯 R² Score", "0.98")


st.divider()


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("🌱 Enter Crop Information")

col1, col2 = st.columns(2)


with col1:

    crop_options = sorted(df["Crop"].unique())

    crop = st.selectbox(
        "🌾 Crop",
        crop_options
    )

    season_options = sorted(df["Season"].unique())

    season = st.selectbox(
        "🌱 Season",
        season_options
    )

    state_options = sorted(df["State"].unique())

    state = st.selectbox(
        "📍 State",
        state_options
    )

    crop_year = st.number_input(
        "📅 Crop Year",
        min_value=1997,
        max_value=2025,
        value=2020,
        step=1
    )


with col2:

    area = st.number_input(
        "📐 Area",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    annual_rainfall = st.number_input(
        "🌧️ Annual Rainfall",
        min_value=0.0,
        value=1200.0,
        step=10.0
    )

    fertilizer = st.number_input(
        "🧪 Fertilizer",
        min_value=0.0,
        value=1500000.0,
        step=10000.0
    )

    pesticide = st.number_input(
        "🧴 Pesticide",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Crop Yield",
    use_container_width=True,
    type="primary"
):

    if area <= 0:
        st.error("Area must be greater than 0.")

    elif annual_rainfall < 0:
        st.error("Annual rainfall cannot be negative.")

    elif fertilizer < 0:
        st.error("Fertilizer cannot be negative.")

    elif pesticide < 0:
        st.error("Pesticide cannot be negative.")

    else:

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "Crop": crop,
            "Crop_Year": crop_year,
            "Season": season,
            "State": state,
            "Area": area,
            "Annual_Rainfall": annual_rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide
        }])

        # Preprocess input
        input_processed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_processed)[0]

        # Save prediction to history
        st.session_state.prediction_history.append({
            "Crop": crop,
            "Year": crop_year,
            "Season": season,
            "State": state,
            "Predicted Yield": round(prediction, 2)
        })

        # Display result
        st.success("Prediction completed successfully! 🌾")

        st.metric(
            label="🌾 Predicted Crop Yield",
            value=f"{prediction:.2f}"
        )

        st.caption(
            "Prediction generated using the trained Random Forest model."
        )


# --------------------------------------------------
# Prediction History
# --------------------------------------------------

if st.session_state.prediction_history:

    st.divider()

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()


# --------------------------------------------------
# About Project
# --------------------------------------------------

with st.expander("ℹ️ About this project"):

    st.write(
        """
        This machine learning application predicts crop yield based on
        agricultural and environmental features.

        **Input Features**
        - Crop
        - Crop Year
        - Season
        - State
        - Area
        - Annual Rainfall
        - Fertilizer
        - Pesticide

        **Machine Learning Model**
        - Random Forest Regressor
        - One-Hot Encoding for categorical features

        **Model Performance**
        - R² Score: approximately 0.98
        - MAE: approximately 9.54
        - RMSE: approximately 126.38

        Production is intentionally excluded from the prediction features
        because it is directly related to the calculation of crop yield
        and could cause target leakage.
        """
    )