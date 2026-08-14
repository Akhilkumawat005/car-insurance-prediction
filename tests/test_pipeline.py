import pytest
import pandas as pd
import numpy as np
import joblib
# Replace: from train import add_custom_features
from features import add_custom_features

@pytest.fixture(scope="module")
def model_pipeline():
    return joblib.load("car_insurance_pipeline.pkl")

def test_feature_engineering():
    df = pd.DataFrame([{
        'age': 20,
        'driving_experience': 1,
        'annual_mileage': 15000,
        'past_accidents': 1,
        'speeding_violations': 1
    }])
    transformed = add_custom_features(df)
    assert transformed.loc[0, 'violation_rate'] == 1.0
    assert transformed.loc[0, 'mileage_per_adult_year'] == 5000.0

def test_pipeline_prediction(model_pipeline):
    df = pd.DataFrame([{
        'age': 35,
        'driving_experience': 10,
        'annual_mileage': 12000.0,
        'past_accidents': 0,
        'speeding_violations': 0,
        'credit_score': 720.0,
        'vehicle_age': 'moderate',
        'vehicle_type': 'sedan'
    }])
    prob = model_pipeline.predict_proba(df)[0][1]
    assert 0.0 <= prob <= 1.0