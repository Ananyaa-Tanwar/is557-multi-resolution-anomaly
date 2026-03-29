import streamlit as st

st.set_page_config(page_title="Early Fraud Signal System", layout="wide")

st.title("Early Fraud Signal System")
st.caption("Bank Account Opening — Application-Stage Detection")

# --- SIDEBAR: Input Form ---
st.sidebar.header("Application Details")

st.sidebar.subheader("Applicant Profile")
income = st.sidebar.number_input("Income", min_value=0, value=50000)
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)
credit_risk_score = st.sidebar.number_input("Credit Risk Score", min_value=0, max_value=1000, value=500)
employment_status = st.sidebar.selectbox("Employment Status", ["CA", "CB", "CC", "CD", "CE", "CF", "CG"])
proposed_credit_limit = st.sidebar.number_input("Proposed Credit Limit", min_value=0, value=5000)

st.sidebar.subheader("Velocity Signals")
velocity_6h = st.sidebar.number_input("Velocity (6h)", min_value=0, value=1)
velocity_24h = st.sidebar.number_input("Velocity (24h)", min_value=0, value=2)
velocity_4w = st.sidebar.number_input("Velocity (4w)", min_value=0, value=5)
zip_count_4w = st.sidebar.number_input("Zip Count (4w)", min_value=0, value=1)

st.sidebar.subheader("Device & Digital")
device_fraud_count = st.sidebar.number_input("Device Fraud Count", min_value=0, value=0)
email_is_free = st.sidebar.selectbox("Free Email Provider?", [0, 1])
foreign_request = st.sidebar.selectbox("Foreign Request?", [0, 1])

st.sidebar.subheader("Banking History")
bank_months_count = st.sidebar.number_input("Bank Months Count (-1 if none)", min_value=-1, value=12)
has_other_cards = st.sidebar.selectbox("Has Other Cards?", [0, 1])
prev_address_months_count = st.sidebar.number_input("Prev Address Months (-1 if none)", min_value=-1, value=24)

# --- MAIN PANEL ---
st.subheader("Risk Assessment")

# Hardcoded dummy score — swap this out when model is ready
score = 73

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Fraud Risk Score", value=f"{score}%")

with col2:
    if score < 30:
        st.success("Risk Band: LOW")
    elif score < 60:
        st.warning("Risk Band: MEDIUM")
    else:
        st.error("Risk Band: HIGH")

st.divider()

# --- SHAP PLACEHOLDER ---
st.subheader("Why was this flagged?")
st.caption("Feature importance — placeholder until SHAP is wired in")

fake_shap = {
    "velocity_6h": 0.42,
    "device_fraud_count": 0.31,
    "credit_risk_score": 0.18,
    "foreign_request": 0.15,
    "bank_months_count": 0.10,
    "email_is_free": 0.07,
}

st.bar_chart(fake_shap)