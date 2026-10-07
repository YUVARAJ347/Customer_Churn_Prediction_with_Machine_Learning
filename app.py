import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Load trained model
# ==========================================

model = joblib.load(
    "models/churn_model.pkl"
)


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information to predict whether "
    "the customer is likely to churn."
)


# ==========================================
# Customer Information
# ==========================================

st.header("Customer Information")


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


senior_citizen_text = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)


partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple_lines = st.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


online_security = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


online_backup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device_protection = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


tech_support = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_tv = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


streaming_movies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0,
    step=1.0
)


total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0,
    step=10.0
)


# ==========================================
# Convert Senior Citizen
# ==========================================

senior_citizen = (
    1 if senior_citizen_text == "Yes" else 0
)


# ==========================================
# Create customer DataFrame
# ==========================================

customer = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [senior_citizen],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone_service],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet_service],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless_billing],

    "PaymentMethod": [payment_method],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]
})


# ==========================================
# Prediction Button
# ==========================================

if st.button(
    "🔮 Predict Churn",
    type="primary"
):

    # Make prediction
    prediction = model.predict(
        customer
    )[0]


    # Get churn probability
    probability = model.predict_proba(
        customer
    )[0][1]


    # ======================================
    # Display result
    # ======================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )


    # ======================================
    # Display probability
    # ======================================

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )