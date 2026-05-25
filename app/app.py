import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Load feature columns
model_columns = joblib.load("models/model_columns.pkl")

st.title("Customer Churn Prediction System")

st.write("Predict whether a telecom customer is likely to churn.")

# User Inputs
gender = st.selectbox("Gender", ["Female", "Male"])

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

Partner = st.selectbox("Partner", ["Yes", "No"])

Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

TotalCharges = st.slider(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# Create Input DataFrame
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_df = pd.DataFrame([input_dict])

# feature order

input_df['gender'] = 1 if gender == "Male" else 0
input_df['Partner'] = 1 if Partner == "Yes" else 0
input_df['Dependents'] = 1 if Dependents == "Yes" else 0
input_df['PhoneService'] = 1 if PhoneService == "Yes" else 0
input_df['PaperlessBilling'] = 1 if PaperlessBilling == "Yes" else 0

multi_lines_map = {"No": 0, "Yes": 1, "No phone service": 2}
input_df['MultipleLines'] = multi_lines_map[MultipleLines]

service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
svc_map = {"No": 0, "Yes": 1, "No internet service": 2}
for col in service_cols:
    input_df[col] = svc_map[input_df[col].iloc[0]]

internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
input_df['InternetService'] = internet_map[InternetService]

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
input_df['Contract'] = contract_map[Contract]

payment_map = {
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}
input_df['PaymentMethod'] = payment_map[PaymentMethod]
# Apply mappings

input_df['InternetService'] = input_df['InternetService'].replace(internet_map)

input_df['Contract'] = input_df['Contract'].replace(contract_map)

input_df['PaymentMethod'] = input_df['PaymentMethod'].replace(payment_map)

# Ensure correct feature order
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Convert all columns to numeric
input_df = input_df.astype(float)

# Predict
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Customer likely to churn. Probability: {probability:.2f}")
    else:
        st.success(f"Customer likely to stay. Probability: {probability:.2f}")

    # Business Recommendation
    st.subheader("Business Recommendation")

    if probability > 0.7:
        st.warning(
            "High churn risk. Recommend retention offers and proactive customer support."
        )
    elif probability > 0.4:
        st.info(
            "Moderate churn risk. Monitor customer engagement closely."
        )
    else:
        st.success(
            "Low churn risk. Customer retention probability is high."
        )