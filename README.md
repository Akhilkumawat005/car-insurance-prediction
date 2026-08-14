# 🚗 Car Insurance Claim Prediction & Risk Pricing System

An end-to-end Machine Learning pipeline and interactive web application that predicts the likelihood of policyholder insurance claims and estimates risk-adjusted policy premiums in real time.

---

## 📌 Features

- **End-to-End ML Pipeline:** Complete preprocessing with median imputation, scaling, categorical encoding, and mathematical feature engineering.
- **Handling Class Imbalance:** Employs balanced class weighting to handle minority claim events effectively, evaluated via **Stratified 5-Fold Cross-Validation** and **ROC-AUC**.
- **Interactive Streamlit Web Dashboard:** Allows underwriters and users to enter driver parameters and instantly view:
  - Binary claim prediction and claim probability.
  - Recommended annual policy premium.
  - Individual risk factor contribution chart (local log-odds breakdown).
- **Global Explainability:** Calculates and plots global feature importance coefficients and odds ratios (`feature_importance.png`).
- **Automated Testing:** Unit test coverage using **Pytest** for mathematical feature transformations and pipeline outputs.

---

## 🏗️ Project Structure

```text
car-insurance-prediction/
│
├── insurance_data.csv          # Simulated / sourced tabular dataset
├── generate_data.py            # Synthetic dataset generator
├── features.py                 # Isolated feature engineering functions
├── train.py                    # Pipeline definition, CV, and model training
├── predict.py                  # CLI inference test script
├── explain.py                  # Global feature importance generator
├── app.py                      # Streamlit interactive UI
├── requirements.txt            # Python dependencies
├── feature_importance.png      # Global risk factors chart
├── car_insurance_pipeline.pkl  # Trained ML pipeline artifact
│
└── tests/
    ├── __init__.py             # Test package marker
    └── test_pipeline.py        # Automated pytest test cases
```

---

## 🛠️ Tech Stack

- **Core & Data Processing:** Python, NumPy, Pandas
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Web App / UI:** Streamlit
- **Testing:** Pytest

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/Akhilkumawat005/car-insurance-prediction.git](https://github.com/Akhilkumawat005/car-insurance-prediction.git)
cd car-insurance-prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell):
.\venv\Scripts\activate

# Activate (macOS/Linux):
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Data & Train Model
```bash
# Generate synthetic dataset
python generate_data.py

# Train pipeline and export model artifact
python train.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*The app will automatically launch in your browser at `http://localhost:8501`.*

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest -v
```

---

## 📊 Methodology & Business Logic

### 1. Feature Engineering

* **Violation Rate:** Ratio of total past violations and accidents relative to driving experience:

  ```text
  Violation Rate = (Speeding Violations + Past Accidents) / (Driving Experience + 1)
  ```

* **Mileage per Adult Year:** Annual mileage normalized against driving-eligible age:

  ```text
  Mileage Ratio = Annual Mileage / max(Age - 17, 1)
  ```

---

### 2. Dynamic Premium Pricing Formula

The estimated annual premium combines baseline risk with predicted claim probabilities:

```text
Annual Premium = $500 (Base) + (Claim Probability × $1,200) + Vehicle Surcharge
```