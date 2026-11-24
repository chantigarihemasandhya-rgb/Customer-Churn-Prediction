# Project Portfolio — Customer Churn Prediction

## Project
Customer Churn Prediction Using Machine Learning

## Objective
Estimate which customers are likely to discontinue a subscription/service using customer profile, contract, tenure, service and billing attributes. The output can help retention teams prioritize customer outreach.

## Technology stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQL (SQLite), Random Forest, Gradient Boosting, optional XGBoost, Streamlit.

## Implemented workflow
1. Load and inspect customer churn data.
2. Remove duplicate records and convert numeric fields safely.
3. Split data into stratified training and test sets.
4. Impute missing values, scale numeric features and encode categorical features.
5. Train Logistic Regression, Decision Tree, Random Forest and Gradient Boosting; optionally train XGBoost.
6. Compare accuracy, precision, recall, F1-score and ROC-AUC.
7. Save the best model and expose probability predictions in Streamlit.
8. Store prediction history in SQLite.

## Deliverables
- `train.py`: model training and evaluation
- `app.py`: Streamlit prediction application
- `src/database.py`: SQLite prediction history
- `data/telco_customer_churn_sample.csv`: synthetic demo dataset
- `reports/model_comparison.csv`: generated after training
- `models/best_model.joblib`: generated after training

## Resume-ready summary
Built an end-to-end customer churn prediction application using Python and Scikit-learn. Performed data preprocessing and feature engineering, trained and compared classification models, evaluated results using precision, recall, F1-score and ROC-AUC, and deployed the selected model through a Streamlit interface for interactive churn-risk prediction.

## Before claiming business results
Run the training workflow and use the actual generated metrics. Clearly identify the included dataset as synthetic unless replaced with a genuine, licensed dataset.
