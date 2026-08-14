import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# Replace: from train import add_custom_features
from features import add_custom_features  # Imports the function for pipeline compatibility

st.set_page_config(
    page_title="Car Insurance Risk & Claim Predictor",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return joblib.load("car_insurance_pipeline.pkl")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}. Please run 'python train.py' first.")
    st.stop()

st.title("🚗 Car Insurance Claim & Premium Risk Predictor")
st.markdown("Enter policyholder and vehicle information to evaluate claim risk and calculate dynamic annual premiums.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Policyholder Profile")
    age = st.slider("Age", min_value=18, max_value=85, value=30, step=1)
    driving_exp = st.slider("Driving Experience (Years)", min_value=0, max_value=max(0, age - 18), value=min(6, age - 18), step=1)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=10)
    annual_mileage = st.number_input("Annual Mileage (Miles)", min_value=1000, max_value=60000, value=12000, step=500)

with col2:
    st.subheader("🚘 Vehicle & Driving History")
    vehicle_type = st.selectbox("Vehicle Type", options=["sedan", "suv", "sports"])
    vehicle_age = st.selectbox("Vehicle Age", options=["new", "moderate", "old"])
    past_accidents = st.number_input("Past Accidents Count", min_value=0, max_value=15, value=0, step=1)
    speeding_violations = st.number_input("Past Speeding Violations", min_value=0, max_value=20, value=0, step=1)

st.divider()

if st.button("📊 Evaluate Policyholder Risk", type="primary", use_container_width=True):
    # Construct raw single-row DataFrame
    input_df = pd.DataFrame([{
        'age': age,
        'driving_experience': driving_exp,
        'annual_mileage': float(annual_mileage),
        'past_accidents': int(past_accidents),
        'speeding_violations': int(speeding_violations),
        'credit_score': float(credit_score),
        'vehicle_age': vehicle_age,
        'vehicle_type': vehicle_type
    }])

    # Pipeline handles feature generation and scaling directly
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    # Dynamic Pricing Engine
    base_premium = 500.0
    risk_surcharge = prob * 1200.0
    type_surcharge = 350.0 if vehicle_type == 'sports' else (150.0 if vehicle_type == 'suv' else 0.0)
    total_premium = round(base_premium + risk_surcharge + type_surcharge, 2)

    # 1. Summary Cards
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric(label="Claim Prediction", value="⚠️ High Claim Risk" if pred == 1 else "✅ Low Claim Risk")
    with res_col2:
        st.metric(label="Claim Probability", value=f"{prob * 100:.1f}%")
    with res_col3:
        st.metric(label="Recommended Premium", value=f"${total_premium:,.2f}")

    # 2. Risk Assessment Tier
    st.write("---")
    st.subheader("Underwriting Assessment")
    if prob >= 0.60:
        st.error(f"**High Risk Tier ({prob*100:.1f}%):** Substantial claim probability. Recommend increased deductible or policy review.")
    elif prob >= 0.30:
        st.warning(f"**Moderate Risk Tier ({prob*100:.1f}%):** Standard terms apply.")
    else:
        st.success(f"**Low Risk Tier ({prob*100:.1f}%):** Eligible for safe-driver discount pricing.")

    # 3. Individual Risk Factor Impact
    st.write("---")
    st.subheader("🔍 Individual Risk Factor Contributions")
    
    # Transform raw data through pipeline steps to get feature impacts
    feat_engineered = pipeline.named_steps['feature_engineering'].transform(input_df)
    preprocessor = pipeline.named_steps['preprocessing']
    transformed_vals = preprocessor.transform(feat_engineered)[0]

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(['vehicle_age', 'vehicle_type']))
    num_feature_names = [
        'age', 'driving_experience', 'annual_mileage', 
        'past_accidents', 'speeding_violations', 'credit_score', 
        'violation_rate', 'mileage_per_adult_year'
    ]
    all_feature_names = num_feature_names + cat_feature_names

    coefficients = pipeline.named_steps['classifier'].coef_[0]
    contributions = transformed_vals * coefficients

    impact_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Impact': contributions
    }).sort_values(by='Impact', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#d9534f' if val > 0 else '#5cb85c' for val in impact_df['Impact']]
    ax.barh(impact_df['Feature'], impact_df['Impact'], color=colors)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Contribution to Claim Likelihood (Log-Odds)")
    ax.set_title("Factor Breakdown (Red = Increases Risk, Green = Decreases Risk)")
    st.pyplot(fig)