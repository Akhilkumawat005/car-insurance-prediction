import numpy as np
import pandas as pd

def add_custom_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d['violation_rate'] = (d['speeding_violations'] + d['past_accidents']) / (d['driving_experience'] + 1)
    d['mileage_per_adult_year'] = d['annual_mileage'] / np.maximum(d['age'] - 17, 1)
    return d