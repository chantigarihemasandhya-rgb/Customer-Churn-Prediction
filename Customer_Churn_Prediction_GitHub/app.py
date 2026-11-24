import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from database import save_prediction, get_recent_predictions

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("Estimate churn risk, understand retention drivers, and explore customer trends.")

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
if artifact is None:
    st.error("Trained model not found. Run `python train.py` from the project folder first.")
    st.stop()

model = artifact["model"]
features = artifact["features"]
metrics = artifact.get("metrics", {})
with st.sidebar:
    st.header("Model information")
    st.write(f"**Selected model:** {artifact.get('model_name', 'Classifier')}")
    st.write(f"**Training rows:** {artifact.get('training_rows', '—'):,}")
    st.caption("Demo dataset is synthetic. Replace it with a licensed/approved real dataset for business use.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict churn", "📊 Model performance", "🗃️ Recent predictions"])
with tab1:
    st.subheader("Customer profile")
    defaults = {
        "gender":"Female", "SeniorCitizen":0, "Partner":"Yes", "Dependents":"No",
        "tenure":12, "PhoneService":"Yes", "MultipleLines":"No",
        "InternetService":"Fiber optic", "OnlineSecurity":"No", "OnlineBackup":"No",
        "DeviceProtection":"No", "TechSupport":"No", "StreamingTV":"Yes",
        "Contract":"Month-to-month", "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check", "MonthlyCharges":85.0, "TotalCharges":1020.0
    }
    options = {
        "gender":["Female","Male"], "SeniorCitizen":[0,1], "Partner":["Yes","No"],
        "Dependents":["Yes","No"], "PhoneService":["Yes","No"],
        "MultipleLines":["Yes","No","No phone service"],
        "InternetService":["DSL","Fiber optic","No"],
        "OnlineSecurity":["Yes","No","No internet service"],
        "OnlineBackup":["Yes","No","No internet service"],
        "DeviceProtection":["Yes","No","No internet service"],
        "TechSupport":["Yes","No","No internet service"],
        "StreamingTV":["Yes","No","No internet service"],
        "Contract":["Month-to-month","One year","Two year"],
        "PaperlessBilling":["Yes","No"],
        "PaymentMethod":["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    }
    with st.form("prediction_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feature in enumerate(features):
            with cols[i % 3]:
                if feature in options:
                    inputs[feature] = st.selectbox(feature, options[feature], index=options[feature].index(defaults[feature]))
                elif feature == "tenure":
                    inputs[feature] = st.slider("tenure (months)", 0, 72, defaults[feature])
                elif feature == "MonthlyCharges":
                    inputs[feature] = st.number_input(feature, 18.0, 150.0, defaults[feature], step=1.0)
                elif feature == "TotalCharges":
                    inputs[feature] = st.number_input(feature, 0.0, 10000.0, defaults[feature], step=50.0)
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    if submitted:
        row = pd.DataFrame([{f: inputs[f] for f in features}])
        probability = float(model.predict_proba(row)[0][1])
        prediction = "Yes" if probability >= 0.5 else "No"
        c1,c2,c3 = st.columns(3)
        c1.metric("Churn probability", f"{probability:.1%}")
        c2.metric("Prediction", "Likely to churn" if prediction=="Yes" else "Likely to stay")
        c3.metric("Risk band", "High" if probability>=0.70 else "Medium" if probability>=0.40 else "Low")
        st.progress(probability)
        if probability >= 0.70:
            st.warning("High-risk customer. Consider proactive outreach, service review, or a suitable retention offer.")
        elif probability >= 0.40:
            st.info("Moderate risk. Monitor engagement and address service concerns.")
        else:
            st.success("Low predicted churn risk.")
        save_prediction(inputs, probability, prediction)

with tab2:
    st.subheader("Evaluation metrics")
    if metrics:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_"," ").title(), f"{value:.3f}")
    st.info("Metrics are calculated on a held-out test split. See reports/model_comparison.csv for model comparison.")
    report = ROOT / "reports" / "model_comparison.csv"
    if report.exists():
        st.dataframe(pd.read_csv(report), use_container_width=True)
    st.caption("Use recall, precision, F1 and ROC-AUC alongside accuracy; churn datasets can be imbalanced.")

with tab3:
    st.subheader("Latest saved predictions")
    records = get_recent_predictions()
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.write("No predictions saved yet.")
