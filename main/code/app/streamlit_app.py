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
income = st.sidebar.number_input("Income (0-1 normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)
proposed_credit_limit = st.sidebar.number_input("Proposed Credit Limit", min_value=0, value=5000)
employment_status = st.sidebar.selectbox("Employment Status", ["CA", "CB", "CC", "CD", "CE", "CF", "CG"])
housing_status = st.sidebar.selectbox("Housing Status", ["BA", "BB", "BC", "BD", "BE", "BF", "BG"])
payment_type = st.sidebar.selectbox("Payment Type", ["AA", "AB", "AC", "AD", "AE"])

st.sidebar.subheader("Financial Details")
intended_balcon_amount = st.sidebar.number_input("Intended Balance/Transfer Amount", min_value=-10.0, value=0.0, step=0.1, format="%.2f")

st.sidebar.subheader("Contact & Identity")
phone_mobile_valid = st.sidebar.selectbox("Mobile Phone Valid?", [1, 0])
phone_home_valid = st.sidebar.selectbox("Home Phone Valid?", [1, 0])
has_other_cards = st.sidebar.selectbox("Has Other Cards?", [1, 0])
foreign_request = st.sidebar.selectbox("Foreign Request?", [0, 1])
email_is_free = st.sidebar.selectbox("Free Email Provider?", [0, 1])

# We're taking REAL SYSTEM DEFAULTS (medians from X_train)
system_defaults = {
    "velocity_6h": 5961.19,
    "velocity_24h": 5020.41,
    "velocity_4w": 5117.56,
    "zip_count_4w": 1361.0,
    "bank_branch_count_8w": 10.0,
    "date_of_birth_distinct_emails_4w": 9.0,
    "credit_risk_score": 120.0,
    "name_email_similarity": 0.4997,
    "device_fraud_count": 0.0,
    "device_distinct_emails_8w": 1.0,
    "session_length_in_minutes": 0.0717,
    "keep_alive_session": 1.0,
    "days_since_request": 0.0002,
    "prev_address_months_count": -1.0,
    "current_address_months_count": 53.0,
    "bank_months_count": 5.0,
    "device_os": "linux",
    "source": "INTERNET",
}

# Using 55-column order from X_test.csv
COLUMN_ORDER = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w",
    "device_fraud_count", "has_prev_address", "is_dirty_device", "velocity_ratio",
    "credit_to_income_ratio", "payment_type_AA", "payment_type_AB", "payment_type_AC",
    "payment_type_AD", "payment_type_AE", "employment_status_CA", "employment_status_CB",
    "employment_status_CC", "employment_status_CD", "employment_status_CE",
    "employment_status_CF", "employment_status_CG", "housing_status_BA",
    "housing_status_BB", "housing_status_BC", "housing_status_BD", "housing_status_BE",
    "housing_status_BF", "housing_status_BG", "device_os_linux", "device_os_macintosh",
    "device_os_other", "device_os_windows", "device_os_x11", "source_INTERNET",
    "source_TELEAPP"
]

def preprocess_input():
    row = {
        "income": income,
        "customer_age": customer_age,
        "proposed_credit_limit": proposed_credit_limit,
        "employment_status": employment_status,
        "housing_status": housing_status,
        "payment_type": payment_type,
        "intended_balcon_amount": intended_balcon_amount,
        "phone_mobile_valid": phone_mobile_valid,
        "phone_home_valid": phone_home_valid,
        "has_other_cards": has_other_cards,
        "foreign_request": foreign_request,
        "email_is_free": email_is_free,
        **system_defaults,
    }

    df = pd.DataFrame([row])

    # Feature engineering — exact same as preprocess.py
    df["has_prev_address"] = (df["prev_address_months_count"] != -1).astype(int)
    df["is_dirty_device"] = (df["device_fraud_count"] > 0).astype(int)
    df["velocity_ratio"] = df["velocity_6h"] / (df["velocity_4w"] + 1)
    df["credit_to_income_ratio"] = df["proposed_credit_limit"] / (df["income"] + 1)

    # One-hot encode
    for pt in ["AA", "AB", "AC", "AD", "AE"]:
        df[f"payment_type_{pt}"] = int(payment_type == pt)
    for es in ["CA", "CB", "CC", "CD", "CE", "CF", "CG"]:
        df[f"employment_status_{es}"] = int(employment_status == es)
    for hs in ["BA", "BB", "BC", "BD", "BE", "BF", "BG"]:
        df[f"housing_status_{hs}"] = int(housing_status == hs)
    for os_ in ["linux", "macintosh", "other", "windows", "x11"]:
        df[f"device_os_{os_}"] = int(system_defaults["device_os"] == os_)
    for src in ["INTERNET", "TELEAPP"]:
        df[f"source_{src}"] = int(system_defaults["source"] == src)

    # Drop raw categorical columns
    df = df.drop(columns=["payment_type", "employment_status", "housing_status",
                           "device_os", "source"])

    # Scale
    scale_cols = ["days_since_request", "session_length_in_minutes", "intended_balcon_amount"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Enforce exact 55-column order
    df = df[COLUMN_ORDER]

    return df

# ANALYZE BUTTON
if st.sidebar.button("Analyze Application"):

    input_df = preprocess_input()

    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    score = round(prob * 100, 2)

    # SHAP via pred_contribs
    dmatrix = xgb.DMatrix(input_df)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = contribs[0][:-1]
    base_value = contribs[0][-1]
    feature_names = COLUMN_ORDER

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    # MAIN PANEL
    st.subheader("Risk Assessment")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Fraud Risk Score", value=f"{score:.2f}%")

    with col2:
        if score < 30:
            st.success("Risk Band: LOW")
        elif score < 60:
            st.warning("Risk Band: MEDIUM")
        else:
            st.error("Risk Band: HIGH")

    st.divider()

    # SHAP Waterfall chart
    st.subheader("Why was this flagged?")
    st.caption("Waterfall chart: How each feature pushes the score up (red) or down (blue) from the baseline")

    top_n = shap_df.head(10).iloc[::-1]  # reverse for bottom-up display

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