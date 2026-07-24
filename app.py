# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(page_title="Fraud Detection App", page_icon="🛡️", layout="centered")

# --- Load Model with Caching ---
@st.cache_resource
def load_model():
    model_path = Path("models/fraud_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    features_path = Path("models/feature_columns.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(features_path)
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'models/fraud_model.pkl', 'models/scaler.pkl', and 'models/feature_columns.pkl' exist.")
    st.stop()

# --- App Title ---
st.title("🛡️ Ethereum Fraud Detection System")
st.markdown("Enter transaction details below to check for potential fraud.")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("Transaction Features")
    
    # Input fields for the top 5 most important features from your model
    col1, col2 = st.columns(2)
    with col1:
        sent_tnx = st.number_input("📤 Sent Transactions", min_value=0, value=10)
        received_tnx = st.number_input("📥 Received Transactions", min_value=0, value=5)
        total_ether_sent = st.number_input("💰 Total Ether Sent", min_value=0.0, value=0.5)
    with col2:
        avg_min_between_sent = st.number_input("⏱️ Avg Min Between Sent Tx", min_value=0.0, value=100.0)
        total_ether_balance = st.number_input("⚖️ Total Ether Balance", min_value=0.0, value=1.2)
        unique_sent_to = st.number_input("👤 Unique Sent To Addresses", min_value=0, value=2)
    
    # Use an expander for optional advanced features
    with st.expander("Advanced Features (Optional)"):
        st.markdown("**Fill these only if you have the data for higher accuracy**")
        # Dynamically create input fields for other features
        other_features = {}
        for col in feature_columns:
            if col not in ['Sent_tnx', 'Received_Tnx', 'Avg min between sent tnx', 
                           'total Ether sent', 'total ether balance', 'Unique Sent To Addresses']:
                other_features[col] = st.number_input(f"{col}", value=0.0, key=col)
    
    submitted = st.form_submit_button("🔍 Predict Fraud", type="primary", use_container_width=True)

# --- Prediction Logic ---
if submitted:
    # Create input dictionary
    input_data = {
        'Sent_tnx': sent_tnx,
        'Received_Tnx': received_tnx,
        'Avg min between sent tnx': avg_min_between_sent,
        'total Ether sent': total_ether_sent,
        'total ether balance': total_ether_balance,
        'Unique Sent To Addresses': unique_sent_to,
        **other_features
    }
    
    # Convert to DataFrame and align columns with training data
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Scale and predict
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    # Display Result
    st.divider()
    st.subheader("📊 Prediction Result")
    
    # Create a nice output
    fraud_prob = probability * 100
    if prediction == 1:
        st.error(f"### 🚨 FRAUD DETECTED")
        st.metric("Fraud Probability", f"{fraud_prob:.2f}%", delta="High Risk", delta_color="inverse")
        st.warning("This transaction has been flagged for review.")
    else:
        st.success(f"### ✅ TRANSACTION NORMAL")
        st.metric("Fraud Probability", f"{fraud_prob:.2f}%", delta="Low Risk", delta_color="normal")
        st.info("This transaction appears to be legitimate.")

    # Log prediction (optional)
    with open("streamlit_predictions.log", "a") as f:
        f.write(f"Prob: {probability:.4f} | Pred: {prediction} | Input: {input_data}\n")