import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Early Fraud Signal System", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("main/code/models/xgb_model.pkl")
    scaler = joblib.load("main/code/models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("Early Fraud Signal System")
st.caption("Bank Account: Application-Stage Detection")

# --- SIDEBAR ---
st.sidebar.header("Application Details")

st.sidebar.subheader("Applicant Profile")
income_raw = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=200000, value=40000, step=1000)
income = income_raw / 200000
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)
proposed_credit_limit = st.sidebar.number_input("Proposed Credit Limit", min_value=0, value=5000)

employment_label = st.sidebar.selectbox("Employment Status", [
    "Full Time", "Part Time", "Self Employed", "Student", "Unemployed", "Retired", "Other"
])
employment_map = {
    "Full Time": "CA", "Part Time": "CB", "Self Employed": "CC",
    "Student": "CD", "Unemployed": "CE", "Retired": "CF", "Other": "CG"
}
employment_status = employment_map[employment_label]

housing_label = st.sidebar.selectbox("Housing Status", [
    "Owner with Mortgage", "Outright Homeowner", "Private Renter",
    "Living with Family", "Temporary Accommodation", "Social Housing", "Other"
])
housing_map = {
    "Owner with Mortgage": "BC", "Outright Homeowner": "BB", "Private Renter": "BA",
    "Living with Family": "BE", "Temporary Accommodation": "BD",
    "Social Housing": "BF", "Other": "BG"
}
housing_status = housing_map[housing_label]

st.sidebar.subheader("Contact & Identity")
phone_mobile_valid = st.sidebar.selectbox("Mobile Phone Valid?", [1, 0])
has_other_cards = st.sidebar.selectbox("Has Other Cards?", [1, 0])

source_label = st.sidebar.selectbox("Application Source", ["Online / App", "Bank Assisted"])
source = "INTERNET" if source_label == "Online / App" else "TELEAPP"

# --- SCENARIO SELECTOR ---
st.sidebar.subheader("Background Scenario")
st.sidebar.caption("Simulates backend signals the bank system would compute automatically")
scenario = st.sidebar.selectbox("Select Scenario", [
    "Typical Legit", "Suspicious", "Edge Case"
])

scenario_defaults = {
    "Typical Legit": {
        "name_email_similarity": 0.8,
        "prev_address_months_count": 24,
        "current_address_months_count": 36,
        "keep_alive_session": 1,
        "credit_risk_score": 120,
        "foreign_request": 0,
        "phone_home_valid": 1,
        "email_is_free": 0,
        "bank_months_count": 24,
        "device_os": "windows",
    },
    "Suspicious": {
        "name_email_similarity": 0.1,
        "prev_address_months_count": -1,
        "current_address_months_count": 6,
        "keep_alive_session": 0,
        "credit_risk_score": 200,
        "foreign_request": 1,
        "phone_home_valid": 0,
        "email_is_free": 1,
        "bank_months_count": -1,
        "device_os": "linux",
    },
    "Edge Case": {
        "name_email_similarity": 0.5,
        "prev_address_months_count": -1,
        "current_address_months_count": 12,
        "keep_alive_session": 1,
        "credit_risk_score": 160,
        "foreign_request": 0,
        "phone_home_valid": 0,
        "email_is_free": 1,
        "bank_months_count": 6,
        "device_os": "windows",
    },
}

fixed_defaults = {
    "days_since_request": 0.0002,
    "session_length_in_minutes": 0.0717,
    "date_of_birth_distinct_emails_4w": 9.0,
    "device_distinct_emails_8w": 1.0,
    "intended_balcon_amount": 0.0,
}

COLUMN_ORDER = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w",
    "has_prev_address", "credit_to_income_ratio",
    "payment_type_AA", "payment_type_AB", "payment_type_AC", "payment_type_AD", "payment_type_AE",
    "employment_status_CA", "employment_status_CB", "employment_status_CC", "employment_status_CD",
    "employment_status_CE", "employment_status_CF", "employment_status_CG",
    "housing_status_BA", "housing_status_BB", "housing_status_BC", "housing_status_BD",
    "housing_status_BE", "housing_status_BF", "housing_status_BG",
    "device_os_linux", "device_os_macintosh", "device_os_other", "device_os_windows", "device_os_x11",
    "source_INTERNET", "source_TELEAPP"
]

def preprocess_input():
    bg = scenario_defaults[scenario]

    row = {
        "income": income,
        "customer_age": customer_age,
        "proposed_credit_limit": proposed_credit_limit,
        "phone_mobile_valid": phone_mobile_valid,
        "has_other_cards": has_other_cards,
        "name_email_similarity": bg["name_email_similarity"],
        "prev_address_months_count": bg["prev_address_months_count"],
        "current_address_months_count": bg["current_address_months_count"],
        "keep_alive_session": bg["keep_alive_session"],
        "credit_risk_score": bg["credit_risk_score"],
        "foreign_request": bg["foreign_request"],
        "phone_home_valid": bg["phone_home_valid"],
        "email_is_free": bg["email_is_free"],
        "bank_months_count": bg["bank_months_count"],
        "days_since_request": fixed_defaults["days_since_request"],
        "session_length_in_minutes": fixed_defaults["session_length_in_minutes"],
        "date_of_birth_distinct_emails_4w": fixed_defaults["date_of_birth_distinct_emails_4w"],
        "device_distinct_emails_8w": fixed_defaults["device_distinct_emails_8w"],
        "intended_balcon_amount": fixed_defaults["intended_balcon_amount"],
    }

    df = pd.DataFrame([row])

    df["has_prev_address"] = (df["prev_address_months_count"] != -1).astype(int)
    df["credit_to_income_ratio"] = df["proposed_credit_limit"] / (df["income"] + 1)

    for pt in ["AA", "AB", "AC", "AD", "AE"]:
        df[f"payment_type_{pt}"] = int("AB" == pt)

    for es in ["CA", "CB", "CC", "CD", "CE", "CF", "CG"]:
        df[f"employment_status_{es}"] = int(employment_status == es)

    for hs in ["BA", "BB", "BC", "BD", "BE", "BF", "BG"]:
        df[f"housing_status_{hs}"] = int(housing_status == hs)

    for os_ in ["linux", "macintosh", "other", "windows", "x11"]:
        df[f"device_os_{os_}"] = int(bg["device_os"] == os_)

    for src in ["INTERNET", "TELEAPP"]:
        df[f"source_{src}"] = int(source == src)

    scale_cols = ["days_since_request", "session_length_in_minutes", "intended_balcon_amount"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    df = df[COLUMN_ORDER]

    return df

# --- ANALYZE BUTTON ---
if st.sidebar.button("Analyze Application"):

    input_df = preprocess_input()

    prob = model.predict_proba(input_df)[0][1]
    score = round(prob * 100, 2)

    dmatrix = xgb.DMatrix(input_df)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = contribs[0][:-1]
    base_value = contribs[0][-1]

    shap_df = pd.DataFrame({
        "feature": COLUMN_ORDER,
        "shap_value": shap_vals
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    st.subheader("Risk Assessment")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Fraud Risk Score", value=f"{score:.2f}%")

    with col2:
        THRESHOLD = 51.87
        if score < 30:
            st.success("Risk Band: LOW")
        elif score < THRESHOLD:
            st.warning("Risk Band: MEDIUM")
        else:
            st.error("Risk Band: HIGH")

    st.divider()

    st.subheader("Why was this flagged?")
    st.caption("Waterfall chart: How each feature pushes the score up (red) or down (blue) from the baseline")

    top_n = shap_df.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in top_n["shap_value"]]
    bars = ax.barh(top_n["feature"], top_n["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on fraud score)")
    ax.set_title(f"Top 10 Features Driving This Decision\nBase score: {base_value:.3f} → Final log-odds: {base_value + shap_vals.sum():.3f}")

    x_range = top_n["shap_value"].abs().max()
    for bar, val in zip(bars, top_n["shap_value"]):
        offset = x_range * 0.04
        ax.text(val + (offset if val >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                clip_on=False)

    ax.set_xlim(
        top_n["shap_value"].min() * 1.3,
        top_n["shap_value"].max() * 1.3
    )

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Fill in the application details in the sidebar and click **Analyze Application** to get a fraud risk score.")