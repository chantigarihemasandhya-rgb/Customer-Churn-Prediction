from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "telco_customer_churn_sample.csv"
MODEL_DIR, REPORT_DIR = ROOT/"models", ROOT/"reports"
MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df = df.drop_duplicates()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
target = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
X = df.drop(columns=drop_cols)
features = X.columns.tolist()
numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42, stratify=target)
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
results, fitted = [], {}
for name, classifier in candidates.items():
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("classifier", classifier)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    row = {"model":name, "accuracy":accuracy_score(y_test,pred),
           "precision":precision_score(y_test,pred,zero_division=0),
           "recall":recall_score(y_test,pred,zero_division=0),
           "f1":f1_score(y_test,pred,zero_division=0),
           "roc_auc":roc_auc_score(y_test,proba)}
    results.append(row); fitted[name] = (pipe, row)
    print(name, row)

# Optional XGBoost model, included when the package is installed.
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.05,
                        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
                        random_state=42)
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("classifier", xgb)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test); proba = pipe.predict_proba(X_test)[:,1]
    row = {"model":"XGBoost", "accuracy":accuracy_score(y_test,pred),
           "precision":precision_score(y_test,pred,zero_division=0),
           "recall":recall_score(y_test,pred,zero_division=0),
           "f1":f1_score(y_test,pred,zero_division=0), "roc_auc":roc_auc_score(y_test,proba)}
    results.append(row); fitted["XGBoost"] = (pipe,row)
except ImportError:
    print("XGBoost not installed; skipping optional XGBoost model.")

comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
comparison.to_csv(REPORT_DIR/"model_comparison.csv", index=False)
best_name = comparison.iloc[0]["model"]
best_model, best_metrics = fitted[best_name]
artifact = {"model":best_model, "model_name":best_name, "features":features,
            "metrics":{k:float(best_metrics[k]) for k in ["accuracy","precision","recall","f1","roc_auc"]},
            "training_rows":len(X_train)}
joblib.dump(artifact, MODEL_DIR/"best_model.joblib")
print(f"\nBest model: {best_name}")
print("Confusion matrix:\n", confusion_matrix(y_test, best_model.predict(X_test)))
print(f"Saved model to {MODEL_DIR/'best_model.joblib'}")
