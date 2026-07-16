"""
London Residential Property Price Predictor
Streamlit web app wrapping a trained Random Forest model.

Reproduces the exact preprocessing / feature engineering pipeline described
in the accompanying dissertation (Chapters 4 and 7), so that raw user inputs
are converted into the same feature representation the model was trained on.
"""

import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
from geopy.distance import geodesic

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="London Property Price Predictor",
    page_icon="🏠",
    layout="centered",
)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/random_forest_model_v2.pkl")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "model/model_features_v2.pkl")

# If set, the model/features are downloaded from a Hugging Face Hub model
# repo instead of read from local disk — used when deployed (e.g. on
# Streamlit Community Cloud), since the ~500MB+ model file isn't committed
# to the GitHub repo the app itself is deployed from.
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "random_forest_model.pkl")
HF_FEATURES_FILENAME = os.environ.get("HF_FEATURES_FILENAME", "model_features.pkl")

CHARING_CROSS = (51.5074, -0.1278)

# ----------------------------------------------------------------------
# Reference tables (extracted exactly from the training notebooks)
# ----------------------------------------------------------------------

# Year -> market_index, exact values from DataRefinement.ipynb output.
# Base year 1995 = 1.0 (median price / 1995 median price).
MARKET_INDEX = {
    1995: 1.000000, 1996: 1.082803, 1997: 1.229299, 1998: 1.426752,
    1999: 1.764331, 2000: 2.063694, 2001: 2.286624, 2002: 2.636943,
    2003: 2.777070, 2004: 3.057325, 2005: 3.184076, 2006: 3.490446,
    2007: 4.038217, 2008: 3.974522, 2009: 4.203822, 2010: 4.649682,
    2011: 4.777070, 2012: 5.000000, 2013: 5.478994, 2014: 6.242038,
    2015: 6.656051, 2016: 7.070064, 2017: 7.267516, 2018: 7.210191,
    2019: 7.292994, 2020: 7.808917, 2021: 7.770701, 2022: 8.280255,
    2023: 8.089172, 2024: 8.025478, 2025: 7.898089, 2026: 8.280255,
}

ENERGY_MAPPING = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}

PROPERTY_CATEGORIES = [
    "Bungalow",  # baseline (dropped in one-hot encoding)
    "Detached",
    "Flat",
    "Maisonette",
    "Other",
    "Semi-Detached",
    "Terraced",
]

CONSTRUCTION_PERIODS = [
    "1900-1949",  # baseline (dropped in one-hot encoding)
    "1950-1975",
    "1976-1995",
    "1996-2011",
    "2012+",
    "Pre-1900",
    "Unknown",
]

LONDON_BOROUGHS = [
    "Camden", "City of Westminster", "Islington", "Kensington and Chelsea",
    "City of London", "Hackney", "Lambeth", "Southwark", "Tower Hamlets",
    "Haringey", "Newham", "Wandsworth",
]

# Held-out test set performance of the final Random Forest model.
# Percentiles are of *absolute* percentage error, so they're used here to
# build an error-based price range around each point prediction rather than
# a formal statistical prediction interval.
MODEL_METRICS = {
    "R2": 0.8317,
    "MAE": 145167.47,
    "RMSE": 553548.29,
    "MAPE": 29.51,
    "median_pct_error": 14.32,
    "p25_pct_error": 6.09,
    "p75_pct_error": 29.35,
}

POSTCODES_IO_URL = "https://api.postcodes.io/postcodes/"

_LAST_KNOWN_YEAR = max(MARKET_INDEX.keys())
_LAST_KNOWN_INDEX = MARKET_INDEX[_LAST_KNOWN_YEAR]
_BASE_YEAR = min(MARKET_INDEX.keys())
_BASE_INDEX = MARKET_INDEX[_BASE_YEAR]
# Long-run compound annual growth rate across the full observed history,
# used only to extrapolate years beyond what the model was actually
# trained on. This is a simple projection, not a forecast.
_EXTRAPOLATION_CAGR = (_LAST_KNOWN_INDEX / _BASE_INDEX) ** (1 / (_LAST_KNOWN_YEAR - _BASE_YEAR)) - 1

DROPDOWN_MAX_YEAR = 2099


def get_market_index(year: int) -> float:
    """
    Return the market index for a given year. Years within the observed
    training range use the exact historical value. Years beyond that are
    extrapolated using the long-run CAGR — treat these as rough projections,
    not data-backed values.
    """
    if year in MARKET_INDEX:
        return MARKET_INDEX[year]
    years_beyond = year - _LAST_KNOWN_YEAR
    return _LAST_KNOWN_INDEX * ((1 + _EXTRAPOLATION_CAGR) ** years_beyond)


def get_model_input_market_index(year: int) -> float:
    """
    The value actually fed to the model. Capped at the last year seen in
    training — Random Forests cannot meaningfully split on values beyond
    the range they were trained on, so feeding raw extrapolated indices in
    (e.g. 10+ for year 2028) has no effect beyond "index is high", causing
    identical predictions for every future year. Instead, future-year
    trend is applied afterwards as an explicit multiplier (see
    get_extrapolation_multiplier), decoupled from the model itself.
    """
    return get_market_index(min(year, _LAST_KNOWN_YEAR))


def get_extrapolation_multiplier(year: int) -> float:
    """
    Post-model scaling factor for years beyond the training range. 1.0 for
    any year within training data (no adjustment). This is a simple trend
    projection applied on top of the model's output, not something the
    model itself is aware of.
    """
    if year <= _LAST_KNOWN_YEAR:
        return 1.0
    return get_market_index(year) / _LAST_KNOWN_INDEX


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    # Local files take priority — used for local development/testing.
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        return model, feature_columns

    # Otherwise, fetch from a Hugging Face Hub model repo — used when
    # deployed, since the large model file isn't committed to GitHub.
    if HF_REPO_ID:
        from huggingface_hub import hf_hub_download
        try:
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
            features_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FEATURES_FILENAME)
        except Exception as e:
            st.error(f"Could not download model from Hugging Face Hub ({HF_REPO_ID}): {e}")
            return None, None
        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        return model, feature_columns

    return None, None


def geocode_postcode(postcode: str):
    """
    Look up a UK postcode via the free postcodes.io API.
    Returns (lat, long, admin_district) or raises ValueError with a
    user-friendly message on failure.
    """
    postcode = postcode.strip()
    if not postcode:
        raise ValueError("Please enter a postcode.")

    try:
        resp = requests.get(POSTCODES_IO_URL + postcode, timeout=6)
    except requests.RequestException:
        raise ValueError(
            "Couldn't reach the postcode lookup service. Check your internet "
            "connection, or switch to manual coordinate entry below."
        )

    if resp.status_code == 404:
        raise ValueError(f"Postcode '{postcode}' was not recognised. Please check it and try again.")
    if resp.status_code != 200:
        raise ValueError("Postcode lookup service returned an error. Try again shortly.")

    result = resp.json().get("result")
    if not result:
        raise ValueError(f"No location data found for '{postcode}'.")

    return result["latitude"], result["longitude"], result.get("admin_district", "Unknown")


def price_range_from_error(point_estimate: float, band: str = "typical"):
    """
    Build a price range around a point estimate using the model's held-out
    test-set percentage-error distribution.
    - 'typical' uses the median absolute percentage error (~50% of past
      predictions were within this margin).
    - 'wide' uses the 75th percentile (~75% of past predictions were within
      this margin) — a more conservative, safer-to-quote range.
    """
    pct = MODEL_METRICS["median_pct_error"] if band == "typical" else MODEL_METRICS["p75_pct_error"]
    low = point_estimate * (1 - pct / 100)
    high = point_estimate * (1 + pct / 100)
    return max(low, 0), high


# ----------------------------------------------------------------------
# Feature engineering (mirrors DataRefinement.ipynb / Final_RF_Model.ipynb)
# ----------------------------------------------------------------------
def build_feature_row(
    lat, long,
    total_floor_area,
    number_habitable_rooms,
    extension_count,
    floor_level_known, floor_level_value,
    top_floor_flag,
    mains_gas,
    new_build_flag,
    energy_rating_letter,
    valuation_year,
    property_category,
    construction_period,
):
    # Distance to Central London (Charing Cross) - geodesic, same as training
    distance_to_centre_km = geodesic((lat, long), CHARING_CROSS).km

    # Floor level: same missing-indicator design as training
    if floor_level_known:
        floor_level_numeric = floor_level_value
        floor_level_missing = 0
    else:
        floor_level_numeric = 0
        floor_level_missing = 1

    energy_rating_score = ENERGY_MAPPING[energy_rating_letter]
    market_index = get_model_input_market_index(valuation_year)

    row = {
        "lat": lat,
        "long": long,
        "TOTAL_FLOOR_AREA": total_floor_area,
        "NUMBER_HABITABLE_ROOMS": number_habitable_rooms,
        "EXTENSION_COUNT": extension_count,
        "floor_level_numeric": floor_level_numeric,
        "floor_level_missing": floor_level_missing,
        "top_floor_flag": int(top_floor_flag),
        "mains_gas": int(mains_gas),
        "new_build_flag": int(new_build_flag),
        "energy_rating_score": energy_rating_score,
        "market_index": market_index,
        "distance_to_centre_km": distance_to_centre_km,
    }

    # One-hot encode property_category (baseline = Bungalow, dropped)
    for cat in PROPERTY_CATEGORIES[1:]:
        row[f"property_category_{cat}"] = 1 if property_category == cat else 0

    # One-hot encode construction_period (baseline = 1900-1949, dropped)
    for period in CONSTRUCTION_PERIODS[1:]:
        row[f"construction_period_{period}"] = 1 if construction_period == period else 0

    return row


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏠 London Residential Property Price Predictor")
st.caption(
    "Machine learning-based automated valuation model (AVM) trained on "
    "HM Land Registry, EPC and ONS data across twelve London boroughs."
)

model, feature_columns = load_model_and_features()

if model is None:
    st.warning(
        "⚠️ No trained model found at `model/random_forest_model.pkl`. "
        "The form below is fully functional for review, but predictions "
        "are disabled until a model file is placed in the `model/` folder. "
        "See the README for deployment instructions."
    )

use_manual_coords = st.toggle(
    "Enter coordinates manually instead of postcode",
    value=False,
    help="Use this if postcode lookup isn't working, or you already know the exact lat/long.",
)

with st.form("property_form"):
    st.subheader("Location")

    lat, long = None, None
    postcode = None

    if not use_manual_coords:
        col1, col2 = st.columns(2)
        with col1:
            postcode = st.text_input("Postcode", placeholder="e.g. SW1A 1AA")
        with col2:
            district = st.selectbox("District (borough)", LONDON_BOROUGHS)
        st.caption(
            "Postcode is used to automatically locate the property "
            "(via postcodes.io). District is for your reference and should "
            f"match one of the twelve boroughs the model was trained on: "
            f"{', '.join(LONDON_BOROUGHS)}."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=51.5074, format="%.6f", min_value=51.20, max_value=51.75)
        with col2:
            long = st.number_input("Longitude", value=-0.1278, format="%.6f", min_value=-0.55, max_value=0.35)

    st.subheader("Property Characteristics")
    col1, col2 = st.columns(2)
    with col1:
        property_category = st.selectbox("Property Type", PROPERTY_CATEGORIES, index=6)
        construction_period = st.selectbox("Construction Period", CONSTRUCTION_PERIODS, index=1)
        total_floor_area = st.number_input("Total Floor Area (sqm)", min_value=10.0, max_value=1000.0, value=75.0, step=1.0)
    with col2:
        number_habitable_rooms = st.number_input("Number of Habitable Rooms", min_value=1, max_value=20, value=3, step=1)
        extension_count = st.number_input("Extension Count", min_value=0, max_value=10, value=0, step=1)
        energy_rating_letter = st.selectbox("Energy Rating (EPC)", list(ENERGY_MAPPING.keys()), index=3)

    st.subheader("Floor & Building Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        floor_level_known = st.checkbox("Floor level known?", value=True)
        floor_level_value = st.number_input(
            "Floor level (0 = ground, -1 = basement)",
            min_value=-1, max_value=50, value=0, step=1,
            disabled=not floor_level_known,
        )
    with col2:
        top_floor_flag = st.checkbox("Top floor?", value=False)
        mains_gas = st.checkbox("Connected to mains gas?", value=True)
    with col3:
        new_build_flag = st.checkbox("New build?", value=False)

    st.subheader("Market Year")
    st.caption(
        "This tool estimates a property's value **given known market "
        "conditions at a point in time** — it does not forecast future "
        "markets. Select a historical year for a genuine model-based "
        "estimate; years beyond the training data are an optional, clearly "
        "separate trend projection, not a real prediction."
    )

    year_options = list(range(DROPDOWN_MAX_YEAR, 1994, -1))
    valuation_year = st.selectbox(
        f"Market year ({_LAST_KNOWN_YEAR} = most recent data)",
        year_options,
        index=year_options.index(_LAST_KNOWN_YEAR),
    )
    if valuation_year > _LAST_KNOWN_YEAR:
        st.caption(
            f"🧪 **Experimental projection.** {valuation_year} is beyond "
            f"this model's training data (which covers up to "
            f"{_LAST_KNOWN_YEAR}). The result below applies a simple "
            "compound-growth trend on top of the model's real "
            f"{_LAST_KNOWN_YEAR}-basis estimate — it is not a genuine "
            "machine learning prediction for that year. Treat it as a "
            "'what if trends continued' illustration only."
        )
    else:
        st.caption(
            "✅ Genuine model prediction — this year is within the data "
            "the model was trained on."
        )

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    if not use_manual_coords:
        try:
            lat, long, resolved_district = geocode_postcode(postcode)
            st.caption(f"📍 Located '{postcode.strip().upper()}' in {resolved_district} ({lat:.5f}, {long:.5f})")
        except ValueError as e:
            st.error(str(e))
            st.stop()

    row = build_feature_row(
        lat=lat, long=long,
        total_floor_area=total_floor_area,
        number_habitable_rooms=number_habitable_rooms,
        extension_count=extension_count,
        floor_level_known=floor_level_known, floor_level_value=floor_level_value,
        top_floor_flag=top_floor_flag,
        mains_gas=mains_gas,
        new_build_flag=new_build_flag,
        energy_rating_letter=energy_rating_letter,
        valuation_year=valuation_year,
        property_category=property_category,
        construction_period=construction_period,
    )

    if model is None:
        st.info("Model not loaded — here is the exact feature vector that would be sent to the model:")
        st.json(row)
    else:
        X = pd.DataFrame([row])
        # Reindex to the exact training feature order; anything unseen becomes 0
        X = X.reindex(columns=feature_columns, fill_value=0)

        log_pred = model.predict(X)[0]
        price_pred = float(np.exp(log_pred))

        extrapolation_multiplier = get_extrapolation_multiplier(valuation_year)
        if extrapolation_multiplier != 1.0:
            price_pred *= extrapolation_multiplier
            st.warning(
                f"🧪 **Experimental trend projection applied (×{extrapolation_multiplier:.2f}).** "
                f"The model itself predicted this property's value under "
                f"{_LAST_KNOWN_YEAR} market conditions (its trained range). "
                f"The figure below then scales that by a simple compound-growth "
                f"assumption to illustrate {valuation_year} — this last step "
                "is arithmetic, not machine learning, and should not be read "
                "as a genuine forecast."
            )

        low_typical, high_typical = price_range_from_error(price_pred, band="typical")
        low_wide, high_wide = price_range_from_error(price_pred, band="wide")

        headline_label = (
            "Estimated Property Value"
            if extrapolation_multiplier == 1.0
            else f"Estimated Property Value (experimental {valuation_year} projection)"
        )
        st.success(f"### {headline_label}")
        st.markdown(f"## £{low_wide:,.0f} – £{high_wide:,.0f}")
        st.caption(
            f"Central estimate: £{price_pred:,.0f}. Range reflects the "
            "model's historical error distribution on held-out test data, "
            "not a formal statistical confidence interval."
        )

        st.markdown(
            """
            <style>
            .range-box {
                background-color: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 8px;
                height: 100%;
            }
            .range-label {
                font-size: 0.8rem;
                color: #9aa0a6;
                margin-bottom: 4px;
            }
            .range-value {
                font-size: clamp(1.1rem, 2.2vw, 1.5rem);
                font-weight: 600;
                white-space: normal;
                word-break: break-word;
                line-height: 1.3;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="range-box">
                    <div class="range-label">Narrower range (typical case)</div>
                    <div class="range-value">£{low_typical:,.0f} – £{high_typical:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="range-box">
                    <div class="range-label">Wider range (safer estimate)</div>
                    <div class="range-value">£{low_wide:,.0f} – £{high_wide:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Model accuracy details"):
            st.markdown(
                f"""
On a held-out test set, this Random Forest model achieved:

| Metric | Value |
|---|---|
| R² | {MODEL_METRICS['R2']:.4f} |
| MAE | £{MODEL_METRICS['MAE']:,.0f} |
| RMSE | £{MODEL_METRICS['RMSE']:,.0f} |
| MAPE | {MODEL_METRICS['MAPE']:.2f}% |
| Median % error | {MODEL_METRICS['median_pct_error']:.2f}% |
| 25th percentile % error | {MODEL_METRICS['p25_pct_error']:.2f}% |
| 75th percentile % error | {MODEL_METRICS['p75_pct_error']:.2f}% |

The **narrower range** above is built from the median percentage error
(roughly half of test predictions fell within this margin of the true
price). The **wider range** uses the 75th percentile error (roughly
three-quarters of test predictions fell within this margin) — a more
conservative band, better suited for a single unverified estimate.

RMSE is notably higher than MAE, which indicates the model's errors are
larger and less consistent for higher-value or unusual properties — so
treat estimates for expensive or atypical homes with extra caution.
"""
            )

        with st.expander("View feature vector sent to model"):
            st.dataframe(X.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Built as part of an MSc dissertation project on ML-based residential "
    "property valuation for London. Data sources: HM Land Registry Price "
    "Paid Dataset, EPC Register, ONS Postcode Directory."
)