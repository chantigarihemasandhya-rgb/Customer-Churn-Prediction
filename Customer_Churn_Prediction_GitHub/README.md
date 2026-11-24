# Customer Churn Prediction Using Machine Learning

A runnable end-to-end portfolio project that trains multiple classification models, compares their metrics, saves the best model, and serves interactive churn-risk predictions through Streamlit.

## Features
- Data cleaning, duplicate removal, missing-value handling and numeric conversion
- Exploratory dataset suitable for charts and analysis
- Feature preprocessing with median/mode imputation, scaling and one-hot encoding
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and optional XGBoost
- Evaluation using accuracy, precision, recall, F1-score and ROC-AUC
- Best-model persistence with Joblib
- Streamlit interface for real-time prediction
- SQLite storage of recent predictions

## Important dataset note
`data/telco_customer_churn_sample.csv` is **synthetic demo data** generated for this project so it runs immediately. It is not a real telecom-company dataset and results must not be represented as production or real-customer performance. For a genuine portfolio experiment, replace it with an appropriately licensed Telco Customer Churn dataset while keeping the expected column names and `Churn` target.

## Project structure
```text
customer_churn_prediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── telco_customer_churn_sample.csv
├── models/                 # generated model artifact
├── reports/                # generated model comparison
└── src/
    └── database.py
```

## Setup (Windows PowerShell)
```powershell
cd customer_churn_prediction
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train and compare models
```powershell
python train.py
```
This creates:
- `models/best_model.joblib`
- `reports/model_comparison.csv`

The script selects the model with the highest held-out ROC-AUC. Because this split is also used for selection, these scores are not an independent final estimate of generalization. XGBoost is attempted when installed; the other models still train if XGBoost is unavailable.

## Launch the application
```powershell
streamlit run app.py
```
Open the local URL printed in the terminal (usually `http://localhost:8501`).

## SQL / prediction history
Predictions submitted in the app are stored in `data/predictions.db` using SQLite. The database is created automatically.

Example query:
```sql
SELECT created_at, churn_probability, prediction
FROM predictions
ORDER BY id DESC;
```

## Dataset columns
The demo file follows the common Telco Churn schema: tenure, contract, internet/service options, payment method, monthly charges, total charges and churn label. `customerID` is excluded from model features.

## Responsible use
A churn score is a statistical estimate, not a certainty. Review data quality, class imbalance, fairness, calibration and business costs before using a model to make customer decisions. Choose a decision threshold based on retention-team capacity and the relative cost of false positives and false negatives.
