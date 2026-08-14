import pandas as pd
import joblib
from features import add_custom_features  # Required for loading the pipeline

# 1. Load the trained pipeline
try:
    pipeline = joblib.load('car_insurance_pipeline.pkl')
except Exception as e:
    print(f"Error loading pipeline: {e}. Please run 'python train.py' first.")
    exit()

# 2. Define sample test drivers
sample_drivers = pd.DataFrame([
    {
        'name': 'Driver A (Low Risk Profile)',
        'age': 45,
        'driving_experience': 25,
        'annual_mileage': 10000,
        'past_accidents': 0,
        'speeding_violations': 0,
        'credit_score': 780,
        'vehicle_age': 'new',
        'vehicle_type': 'sedan'
    },
    {
        'name': 'Driver B (High Risk Profile)',
        'age': 20,
        'driving_experience': 1,
        'annual_mileage': 25000,
        'past_accidents': 2,
        'speeding_violations': 3,
        'credit_score': 540,
        'vehicle_age': 'old',
        'vehicle_type': 'sports'
    }
])

driver_names = sample_drivers.pop('name')

# 3. Predict directly on raw inputs
predictions = pipeline.predict(sample_drivers)
probabilities = pipeline.predict_proba(sample_drivers)[:, 1]

print("=" * 60)
print(" CAR INSURANCE CLAIM INFERENCE RESULTS")
print("=" * 60)

for name, pred, prob in zip(driver_names, predictions, probabilities):
    status = "⚠️ LIKELY CLAIM" if pred == 1 else "✅ LOW RISK"
    
    # Premium calculation
    base_premium = 500.0
    risk_surcharge = prob * 1200.0
    total_premium = base_premium + risk_surcharge
    
    print(f"\nProfile: {name}")
    print(f"  Prediction   : {status}")
    print(f"  Probability  : {prob * 100:.2f}%")
    print(f"  Est. Premium : ${total_premium:,.2f}")

print("\n" + "=" * 60)