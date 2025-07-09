import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Loading the model and scaler
model = joblib.load("Customer_Churn_Prediction_Model/model.pkl")
scaler = joblib.load("Customer_Churn_Prediction_Model/scaler.pkl")

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="centered")
st.title("Telco Customer Churn Prediction")
st.markdown("Predict whether a customer will churn based on their details or upload a CSV for batch prediction.")

option = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Batch Prediction (CSV)"])

# Labeling encoding mapping for categorical columns
label_order = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'No phone service', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'No internet service', 'Yes'],
    'OnlineBackup': ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport': ['No', 'No internet service', 'Yes'],
    'StreamingTV': ['No', 'No internet service', 'Yes'],
    'StreamingMovies': ['No', 'No internet service', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
}

if option == "Single Prediction":
    def user_input_features():
        gender = st.selectbox('Gender', ['Female', 'Male'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Has Partner?', ['No', 'Yes'])
        dependents = st.selectbox('Has Dependents?', ['No', 'Yes'])
        tenure = st.slider('Tenure (months)', 0, 72, 1)
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check',
                                                         'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, step=1.0)
        total_charges = st.number_input('Total Charges', min_value=0.0, max_value=9000.0, step=10.0)

        data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    for col, order in label_order.items():
        input_df[col] = pd.Categorical(input_df[col], categories=order, ordered=True).codes

    scaled_input = scaler.transform(input_df)

    if st.button('Predict Churn'):
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        if prediction == 1:
            st.error(f"The customer is likely to churn. Probability: {prob:.2f}")
        else:
            st.success(f"The customer is likely to stay. Probability: {1 - prob:.2f}")

elif option == "Batch Prediction (CSV)":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Dropping irrelevant or labelled columns
        for col in ['customerID', 'Churn']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

        for col, order in label_order.items():
            df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]

        df_results = df.copy()
        df_results['PredictedChurn'] = np.where(preds == 1, 'Yes', 'No')
        df_results['ChurnProbability'] = probs.round(2)

        st.write("Prediction Results")
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )