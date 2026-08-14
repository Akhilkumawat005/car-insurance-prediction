import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from features import add_custom_features  # Required for loading the pipeline

# 1. Load pipeline
try:
    pipeline = joblib.load('car_insurance_pipeline.pkl')
except Exception as e:
    print(f"Error loading pipeline: {e}. Please run 'python train.py' first.")
    exit()

# 2. Extract feature names from preprocessing stage
preprocessor = pipeline.named_steps['preprocessing']
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = list(cat_encoder.get_feature_names_out(['vehicle_age', 'vehicle_type']))

num_features = [
    'age', 'driving_experience', 'annual_mileage', 
    'past_accidents', 'speeding_violations', 'credit_score', 
    'violation_rate', 'mileage_per_adult_year'
]

all_feature_names = num_features + cat_features

# 3. Extract Logistic Regression coefficients & calculate Odds Ratios
model = pipeline.named_steps['classifier']
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Log_Odds_Coeff': coefficients,
    'Odds_Ratio': odds_ratios
}).sort_values(by='Log_Odds_Coeff', ascending=False)

print("=" * 65)
print(" GLOBAL MODEL FEATURE IMPORTANCE & ODDS RATIOS")
print("=" * 65)
print(importance_df.to_string(index=False))

# 4. Generate and save visualization plot
plt.figure(figsize=(10, 6))
colors = ['#d9534f' if c > 0 else '#5cb85c' for c in importance_df['Log_Odds_Coeff']]
bars = plt.barh(importance_df['Feature'], importance_df['Log_Odds_Coeff'], color=colors)

plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Impact on Claim Likelihood (Positive = Increases Risk, Negative = Decreases Risk)", fontsize=11)
plt.xlabel("Log-Odds Coefficient (Effect on Claim Risk)")
plt.gca().invert_yaxis()  # Put highest positive risk at the top
plt.tight_layout()

output_path = 'feature_importance.png'
plt.savefig(output_path, dpi=300)
print(f"\nSUCCESS: Feature importance chart saved as '{output_path}'.")