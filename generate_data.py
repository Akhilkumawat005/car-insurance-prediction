import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 5000

# 1. Generate demographic & vehicle attributes
age = np.random.randint(18, 75, n_samples)
driving_experience = np.clip(age - 18 - np.random.randint(0, 5, n_samples), 0, None)
annual_mileage = np.random.normal(15000, 4000, n_samples).clip(3000, 40000)
past_accidents = np.random.poisson(0.4, n_samples)
speeding_violations = np.random.poisson(0.6, n_samples)
credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
vehicle_age = np.random.choice(['new', 'moderate', 'old'], size=n_samples, p=[0.25, 0.5, 0.25])
vehicle_type = np.random.choice(['sedan', 'suv', 'sports'], size=n_samples, p=[0.5, 0.4, 0.1])

# 2. Risk calculation formula
risk = (
    -0.03 * (age - 40)
    - 0.05 * driving_experience
    + 0.00005 * (annual_mileage - 15000)
    + 0.8 * past_accidents
    + 0.5 * speeding_violations
    - 0.004 * (credit_score - 650)
    + np.where(vehicle_type == 'sports', 1.2, 0.0)
    + np.where(vehicle_age == 'old', 0.4, 0.0)
    - 1.8
)

prob = 1 / (1 + np.exp(-risk))
claim_status = (np.random.rand(n_samples) < prob).astype(int)

df = pd.DataFrame({
    'age': age,
    'driving_experience': driving_experience,
    'annual_mileage': annual_mileage,
    'past_accidents': past_accidents,
    'speeding_violations': speeding_violations,
    'credit_score': credit_score,
    'vehicle_age': vehicle_age,
    'vehicle_type': vehicle_type,
    'claim_status': claim_status
})

# Add missing values for real-world preprocessing
df.loc[np.random.choice(df.index, size=150, replace=False), 'credit_score'] = np.nan
df.loc[np.random.choice(df.index, size=100, replace=False), 'annual_mileage'] = np.nan

df.to_csv('insurance_data.csv', index=False)
print("SUCCESS: Dataset generated and saved to 'insurance_data.csv'.")