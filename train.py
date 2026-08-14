import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from features import add_custom_features

def build_pipeline():
    feature_engineer = FunctionTransformer(add_custom_features, validate=False)

    num_cols = [
        'age', 'driving_experience', 'annual_mileage', 
        'past_accidents', 'speeding_violations', 'credit_score', 
        'violation_rate', 'mileage_per_adult_year'
    ]
    cat_cols = ['vehicle_age', 'vehicle_type']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    full_pipeline = Pipeline([
        ('feature_engineering', feature_engineer),
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    return full_pipeline

if __name__ == '__main__':
    df = pd.read_csv('insurance_data.csv')
    X = df.drop(columns=['claim_status'])
    y = df['claim_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=['roc_auc', 'f1_weighted'])
    print(f"Mean CV ROC-AUC: {scores['test_roc_auc'].mean():.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(pipeline, 'car_insurance_pipeline.pkl')
    print("\nSUCCESS: Model pipeline retrained and saved to car_insurance_pipeline.pkl")