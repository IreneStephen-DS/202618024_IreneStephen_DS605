# ============================================================
# DS605 - Airbnb Price Prediction
# Streamlit Application
# ============================================================

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "airbnb_price_pipeline.pkl"
)


@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    return model


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model. "
        "Please make sure models/airbnb_price_pipeline.pkl exists."
    )

    st.stop()


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🏠 Airbnb Nightly Price Prediction"
)

st.write(
    """
    Enter the Airbnb listing information below.
    The trained machine learning model will estimate
    the expected nightly price.
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "Listing Information"
)


neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island"
    ]
)


neighbourhood = st.text_input(
    "Neighbourhood",
    value="Upper East Side"
)


latitude = st.number_input(
    "Latitude",
    min_value=40.4,
    max_value=41.0,
    value=40.7736,
    format="%.6f"
)


longitude = st.number_input(
    "Longitude",
    min_value=-74.3,
    max_value=-73.6,
    value=-73.9566,
    format="%.6f"
)


room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)


minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=3
)


number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=1000,
    value=50
)


reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=100.0,
    value=2.0
)


calculated_host_listings_count = st.number_input(
    "Host Listing Count",
    min_value=1,
    max_value=500,
    value=1
)


availability_365 = st.number_input(
    "Availability (days/year)",
    min_value=0,
    max_value=365,
    value=200
)


review_year = st.number_input(
    "Review Year",
    min_value=0,
    max_value=2025,
    value=2019
)


review_month = st.number_input(
    "Review Month",
    min_value=0,
    max_value=12,
    value=6
)


review_days_old = st.number_input(
    "Days Since Last Review",
    min_value=0,
    max_value=5000,
    value=30
)


# ============================================================
# FEATURE ENGINEERING FOR USER INPUT
# ============================================================

total_reviews = number_of_reviews

availability_ratio = (
    availability_365 / 365
)

minimum_nights_log = np.log1p(
    minimum_nights
)

number_of_reviews_log = np.log1p(
    number_of_reviews
)

availability_365_log = np.log1p(
    availability_365
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "neighbourhood_group": [
        neighbourhood_group
    ],

    "neighbourhood": [
        neighbourhood
    ],

    "latitude": [
        latitude
    ],

    "longitude": [
        longitude
    ],

    "room_type": [
        room_type
    ],

    "minimum_nights": [
        minimum_nights
    ],

    "number_of_reviews": [
        number_of_reviews
    ],

    "reviews_per_month": [
        reviews_per_month
    ],

    "calculated_host_listings_count": [
        calculated_host_listings_count
    ],

    "availability_365": [
        availability_365
    ],

    "review_year": [
        review_year
    ],

    "review_month": [
        review_month
    ],

    "review_days_old": [
        review_days_old
    ],

    "total_reviews": [
        total_reviews
    ],

    "availability_ratio": [
        availability_ratio
    ],

    "minimum_nights_log": [
        minimum_nights_log
    ],

    "number_of_reviews_log": [
        number_of_reviews_log
    ],

    "availability_365_log": [
        availability_365_log
    ]
})


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "💰 Predict Nightly Price",
    type="primary"
):

    prediction = model.predict(
        input_data
    )[0]


    # Avoid negative prediction

    prediction = max(
        0,
        prediction
    )


    st.success(
        f"### Estimated Nightly Price: ${prediction:,.2f}"
    )


    st.info(
        """
        This estimate is generated by a machine learning model
        trained on the NYC Airbnb Open Data dataset. It should be
        treated as an estimate rather than a guaranteed market price.
        """
    )
