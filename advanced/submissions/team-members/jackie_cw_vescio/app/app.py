import json
from pathlib import Path
from datetime import datetime
import io

from tensorflow import keras
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

import shap
import streamlit.components.v1 as components


st.set_page_config(page_title="CardioRiskIQ", page_icon="🫀", layout="centered")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent   # .../jackie_cw_vescio
ART_DIR = ROOT_DIR / "artifacts"

# ------------------------------------------------------------------
# Friendly label -> numeric code maps (match UCI Cleveland dataset encodings)
# ------------------------------------------------------------------
CP_MAP = {
    "0 — Asymptomatic (no chest pain reported)": 0,
    "1 — Atypical angina (not classic; may vary in location/trigger)": 1,
    "2 — Non-anginal pain (not consistent with angina)": 2,
    "3 — Typical angina (classic exertional chest pain)": 3,
}

RESTECG_MAP = {
    "0 — Left ventricular hypertrophy (LVH) by ECG criteria": 0,
    "1 — Normal": 1,
    "2 — ST-T wave abnormality (possible ischemia-related changes)": 2,
}

SLOPE_MAP = {
    "0 — Downsloping": 0,
    "1 — Flat": 1,
    "2 — Upsloping": 2,
}

THAL_MAP = {
    "0 — Fixed defect (no reversibility; possible old damage/scar)": 0,
    "1 — Normal": 1,
    "2 — Reversible defect (changes with stress; possible ischemia)": 2,
}

CA_MAP = {
    "0 — None visible (0 vessels)": 0,
    "1 — One vessel visible": 1,
    "2 — Two vessels visible": 2,
    "3 — Three vessels visible": 3,
}

YESNO_MAP = {"No": 0, "Yes": 1}

SAMPLE_PATIENTS = {
    "Low Risk Patient": {
        # Prediction: 13.6%, Actual: num=0
        "age": 58,
        "sex": 1,  # Male
        "cp": 1,  # atypical angina
        "trestbps": 130,
        "chol": 251,
        "fbs": 0,
        "restecg": 1,  # normal
        "thalach": 110,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,  # flat
        "ca": 0,
        "thal": 1,  # normal
    },

 "Moderate Risk Patient": {
    # Prediction: 68%
    "age": 63,  # Older (was 59)
    "sex": 1,  
    "cp": 2,  # non-anginal (safer than 0 or 3)
    "trestbps": 155,  # Higher BP
    "chol": 270,  # High cholesterol
    "fbs": 1,  # diabetes
    "restecg": 2,  # ST-T abnormality
    "thalach": 130,  # Low max HR
    "exang": 0,  
    "oldpeak": 1.8,  # Higher ST depression
    "slope": 1,  # flat
    "ca": 0,  
    "thal": 2,
    },

    "High Risk Patient": {
        # Prediction: 94.9%, Actual: num=2
        "age": 62,
        "sex": 1,  # Male
        "cp": 0,  # asymptomatic
        "trestbps": 115,
        "chol": 0,
        "fbs": 1,
        "restecg": 1,  # normal
        "thalach": 128,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 0,  # downsloping
        "ca": 0,
        "thal": 1,  # normal
    },
}

# ------------------------------------------------------------------
# Load model + preprocessing artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = keras.models.load_model(ART_DIR / "cardiorisk_model.keras")
        imputer = joblib.load(ART_DIR / "knn_imputer.joblib")
        scaler = joblib.load(ART_DIR / "scaler.joblib")
        feature_order = json.loads((ART_DIR / "feature_order.json").read_text())
        
        # Load calibrator (not used - keeping for compatibility)
        try:
            calibrator = joblib.load(ART_DIR / "calibrator.joblib")
        except:
            calibrator = None
        
        return model, imputer, scaler, feature_order, calibrator
    except Exception as e:
        st.error(f"Failed to load artifacts from: {ART_DIR}")
        st.exception(e)
        st.stop()


model, imputer, scaler, feature_order, calibrator = load_artifacts()


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def create_risk_gauge(probability):
    """Create a visual gauge for risk probability"""
    # Determine risk level and color
    if probability < 0.3:
        risk_level = "Low Risk"
        color = "#28a745"  # Green
    elif probability < 0.7:
        risk_level = "Moderate Risk"
        color = "#ffc107"  # Yellow
    else:
        risk_level = "High Risk"
        color = "#dc3545"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{risk_level}</b>", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkgray", 'family': "Arial"}
    )
    
    return fig


def generate_pdf_report(patient_data, probability, prediction, threshold):
    """Generate a PDF report of the prediction"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#dc3545') if prediction == 1 else colors.HexColor('#28a745'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("🫀 CardioRiskIQ Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment
    risk_text = f"<b>Risk Level:</b> {'HIGH RISK' if prediction == 1 else 'LOW RISK'}"
    risk_para = Paragraph(risk_text, styles['Heading2'])
    story.append(risk_para)
    story.append(Paragraph(f"<b>Predicted Probability:</b> {probability:.1%}", styles['Normal']))
    story.append(Paragraph(f"<b>Decision Threshold:</b> {threshold:.0%}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Features Table
    story.append(Paragraph("<b>Patient Features:</b>", styles['Heading3']))
    story.append(Spacer(1, 0.1*inch))
    
    feature_labels = {
        'age': 'Age (years)',
        'sex': 'Sex',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting BP (mm Hg)',
        'chol': 'Cholesterol (mg/dL)',
        'fbs': 'Fasting Blood Sugar > 120',
        'restecg': 'Resting ECG',
        'thalach': 'Max Heart Rate (bpm)',
        'exang': 'Exercise-Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'ST Slope',
        'ca': 'Major Vessels',
        'thal': 'Thal Test'
    }
    
    data = [['Feature', 'Value']]
    for key, label in feature_labels.items():
        if key in patient_data:
            value = patient_data[key]
            if key == 'sex':
                value = 'Male' if value == 1 else 'Female'
            elif key in ['fbs', 'exang']:
                value = 'Yes' if value == 1 else 'No'
            data.append([label, str(value)])
    
    table = Table(data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer = Paragraph(
        "<b>Disclaimer:</b> This tool is for educational and demonstration purposes only. "
        "It is not intended for medical diagnosis or treatment decisions. "
        "Please consult with a qualified healthcare professional for medical advice.",
        styles['Italic']
    )
    story.append(disclaimer)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def make_prediction(input_data):
    """Run prediction pipeline with calibration"""
    # Step 1: Create DataFrame with ONLY continuous features
    X_continuous = pd.DataFrame([{
        "age": input_data["age"],
        "trestbps": input_data["trestbps"],
        "chol": input_data["chol"],
        "thalch": input_data["thalach"],
        "oldpeak": input_data["oldpeak"],
    }])

    # Step 2: Apply imputation
    X_imputed = imputer.transform(X_continuous)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X_continuous.columns)

    # Step 3: Add categorical features and flags
    X_imputed_df["outlier_flag"] = 0
    X_imputed_df["sex_idx"] = input_data["sex"]
    X_imputed_df["dataset_idx"] = 0
    X_imputed_df["cp_idx"] = input_data["cp"]
    X_imputed_df["fbs_idx"] = input_data["fbs"]
    X_imputed_df["restecg_idx"] = input_data["restecg"]
    X_imputed_df["exang_idx"] = input_data["exang"]
    X_imputed_df["slope_idx"] = input_data["slope"]
    X_imputed_df["ca_idx"] = input_data["ca"]
    X_imputed_df["thal_idx"] = input_data["thal"]

    # Step 4: Reorder columns
    X_final = X_imputed_df[feature_order]

    # Step 5: Scale
    X_scaled = scaler.transform(X_final)

    # Step 6: Predict
    proba = float(model.predict(X_scaled, verbose=0).ravel()[0])
    
    # Note: Calibration removed - the model's raw predictions are well-calibrated
    # (The calibrator was trained on an old buggy model and made predictions worse)
    
    return proba


# ------------------------------------------------------------------
# SHAP Explanation Functions
# ------------------------------------------------------------------
@st.cache_resource
def get_shap_explainer():
    """Create and cache SHAP explainer"""
    # Use a small background dataset for SHAP
    # We'll use the training data structure
    background = pd.DataFrame(np.zeros((100, len(feature_order))), columns=feature_order)
    
    # Wrap model prediction for SHAP
    def model_predict(X):
        return model.predict(X, verbose=0).ravel()
    
    explainer = shap.KernelExplainer(model_predict, background)
    return explainer


def generate_shap_explanation(input_data):
    """Generate SHAP values for a single patient"""
    # Create the full feature vector (same as make_prediction)
    X_continuous = pd.DataFrame([{
        "age": input_data["age"],
        "trestbps": input_data["trestbps"],
        "chol": input_data["chol"],
        "thalch": input_data["thalach"],
        "oldpeak": input_data["oldpeak"],
    }])
    
    X_imputed = imputer.transform(X_continuous)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X_continuous.columns)
    
    X_imputed_df["outlier_flag"] = 0
    X_imputed_df["sex_idx"] = input_data["sex"]
    X_imputed_df["dataset_idx"] = 0
    X_imputed_df["cp_idx"] = input_data["cp"]
    X_imputed_df["fbs_idx"] = input_data["fbs"]
    X_imputed_df["restecg_idx"] = input_data["restecg"]
    X_imputed_df["exang_idx"] = input_data["exang"]
    X_imputed_df["slope_idx"] = input_data["slope"]
    X_imputed_df["ca_idx"] = input_data["ca"]
    X_imputed_df["thal_idx"] = input_data["thal"]
    
    X_final = X_imputed_df[feature_order]
    X_scaled = scaler.transform(X_final)
    
    # Get SHAP explainer
    explainer = get_shap_explainer()
    
    # Calculate SHAP values (this may take a few seconds)
    shap_values = explainer.shap_values(X_scaled)
    
    return shap_values, X_scaled, X_final


# ------------------------------------------------------------------
# App Header
# ------------------------------------------------------------------
st.title("🫀 CardioRiskIQ")
st.caption("Heart disease risk prediction using a deep learning model (Advanced Track).")

tab1, tab2 = st.tabs(["🧪 Predict", "ℹ️ About"])


# ------------------------------------------------------------------
# Predict Tab
# ------------------------------------------------------------------
with tab1:
    st.subheader("Enter Patient Features")
    
    # Sample Patient Buttons
    st.markdown("**Quick Test:** Load a sample patient profile")
    col1, col2, col3 = st.columns(3)
    
    sample_selected = None
    with col1:
        if st.button("🟢 Low Risk", use_container_width=True):
            sample_selected = "Low Risk Patient"
    with col2:
        if st.button("🟡 Moderate Risk", use_container_width=True):
            sample_selected = "Moderate Risk Patient"
    with col3:
        if st.button("🔴 High Risk", use_container_width=True):
            sample_selected = "High Risk Patient"
    
    # Initialize session state for form values
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = SAMPLE_PATIENTS["Moderate Risk Patient"]
    
    # Load sample if selected
    if sample_selected:
        st.session_state.patient_data = SAMPLE_PATIENTS[sample_selected]
        st.success(f"✓ Loaded {sample_selected}")
    
    st.divider()
    
    # Get current values from session state
    current = st.session_state.patient_data
    
    # --- Demographics ---
    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120, value=int(current['age']),
        help="Patient age in years."
    )

    sex_label = st.selectbox(
        "Sex",
        ["Female", "Male"],
        index=int(current['sex']),
        help="Converted to the dataset encoding internally: Female=0, Male=1."
    )
    sex = 0 if sex_label == "Female" else 1

    # --- Symptoms / clinical presentation ---
    cp_options = list(CP_MAP.keys())
    cp_label = st.selectbox(
        "Chest pain type (cp)",
        cp_options,
        index=int(current['cp']),
        help="These are standard UCI Cleveland categories."
    )
    cp = CP_MAP[cp_label]

    # --- Vitals / labs ---
    trestbps = st.number_input(
        "Resting blood pressure (trestbps) [mm Hg]",
        min_value=50, max_value=250, value=int(current['trestbps']),
        help="Resting blood pressure measured in mm Hg."
    )

    chol = st.number_input(
        "Serum cholesterol (chol) [mg/dL]",
        min_value=0, max_value=600, value=int(current['chol']),
        help="Serum cholesterol in mg/dL."
    )

    fbs_label = st.selectbox(
        "Fasting blood sugar > 120 mg/dL (fbs)",
        ["No (≤ 120 mg/dL)", "Yes (> 120 mg/dL)"],
        index=int(current['fbs']),
        help="Binary indicator."
    )
    fbs = 0 if fbs_label.startswith("No") else 1

    # --- ECG / stress-related features ---
    restecg_options = list(RESTECG_MAP.keys())
    restecg_label = st.selectbox(
        "Resting ECG (restecg)",
        restecg_options,
        index=int(current['restecg']),
        help="Resting electrocardiogram category."
    )
    restecg = RESTECG_MAP[restecg_label]

    thalach = st.number_input(
        "Max heart rate achieved (thalach) [bpm]",
        min_value=50, max_value=250, value=int(current['thalach']),
        help="Maximum heart rate achieved during exercise testing."
    )

    exang_label = st.selectbox(
        "Exercise-induced angina (exang)",
        ["No", "Yes"],
        index=int(current['exang']),
        help="Chest pain triggered by exercise."
    )
    exang = 0 if exang_label == "No" else 1

    oldpeak = st.number_input(
        "ST depression induced by exercise (oldpeak)",
        min_value=0.0, max_value=10.0, value=float(current['oldpeak']), step=0.1,
        help="A stress-test ECG measurement."
    )

    slope_options = list(SLOPE_MAP.keys())
    slope_label = st.selectbox(
        "Slope of the peak exercise ST segment (slope)",
        slope_options,
        index=int(current['slope']),
        help="Shape of the ST segment during peak exercise."
    )
    slope = SLOPE_MAP[slope_label]

    ca_options = list(CA_MAP.keys())
    ca_label = st.selectbox(
        "Number of major vessels colored by fluoroscopy (ca)",
        ca_options,
        index=int(current['ca']),
        help="Count of major vessels visible during fluoroscopy (0–3)."
    )
    ca = CA_MAP[ca_label]

    thal_options = list(THAL_MAP.keys())
    thal_label = st.selectbox(
        "Thal test result (thal)",
        thal_options,
        index=int(current['thal']),
        help="Thal test interpretation category."
    )
    thal = THAL_MAP[thal_label]

    st.caption(
        "💡 **Tip:** Use the sample patient buttons above for quick testing!"
    )

    threshold = st.slider(
        "Decision threshold (higher = fewer high-risk predictions)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01
    )

    with st.expander("📖 Quick glossary (plain-English definitions)"):
        st.markdown(
            """
- **Angina**: chest pain/discomfort caused by reduced blood flow to the heart.
- **Resting ECG**: a heart electrical activity reading taken while resting.
- **ST depression (oldpeak)**: a stress-test ECG measurement; higher values can indicate reduced blood flow (ischemia).
- **Slope (ST segment)**: how the ECG ST segment looks at peak exercise (upsloping/flat/downsloping).
- **Thal**: a stress-imaging/test-related category used in the dataset.
- **ca**: number of major vessels visible/colored via fluoroscopy in the dataset.
            """
        )

    # -------------------------------
    # Prediction logic
    # -------------------------------
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        
        # Update session state
        st.session_state.patient_data = input_dict

        try:
            # Make prediction
            proba = make_prediction(input_dict)
            pred = 1 if proba >= threshold else 0
            
            st.divider()
            st.subheader("📊 Prediction Results")
            
            # Display gauge
            fig = create_risk_gauge(proba)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation with color
            if pred == 1:
                st.error("⚠️ **HIGH RISK** — Model predicts presence of heart disease", icon="🚨")
            else:
                st.success("✓ **LOW RISK** — Model predicts absence of heart disease", icon="✅")
            
            # Additional metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability", f"{proba:.1%}")
            with col2:
                st.metric("Threshold", f"{threshold:.0%}")
            
            st.info(
                "💡 **Note:** Adjusting the decision threshold changes the tradeoff between "
                "catching more at-risk patients (higher sensitivity) and reducing false alarms (higher specificity)."
            )
            
            # Download button
            pdf_buffer = generate_pdf_report(input_dict, proba, pred, threshold)
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=pdf_buffer,
                file_name=f"cardiorisk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            # SHAP Explanations
            st.divider()
            st.subheader("🔍 Model Explanation (SHAP Analysis)")
            
            st.info(
                "💡 **SHAP (SHapley Additive exPlanations)** shows which patient features "
                "influenced this prediction and by how much. Red features increase risk, "
                "blue features decrease risk."
            )
            
            with st.spinner("Generating SHAP explanation... (this may take 10-30 seconds)"):
                try:
                    shap_values, X_scaled, X_final = generate_shap_explanation(input_dict)
                    
                    # Waterfall plot
                    st.markdown("### Feature Impact on This Prediction")
                    fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[0],
                            base_values=get_shap_explainer().expected_value,
                            data=X_scaled[0],
                            feature_names=feature_order
                        ),
                        show=False
                    )
                    st.pyplot(fig_waterfall)
                    plt.close()
                    
                    # Force plot
                    st.markdown("### Interactive Feature Contribution")
                    force_plot = shap.force_plot(
                        get_shap_explainer().expected_value,
                        shap_values[0],
                        X_scaled[0],
                        feature_names=feature_order,
                        matplotlib=False
                    )
                    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                    components.html(shap_html, height=150)
                    
                    # Feature importance for this patient
                    st.markdown("### Top 5 Most Important Features for This Patient")
                    feature_importance = pd.DataFrame({
                        'Feature': feature_order,
                        'Impact': np.abs(shap_values[0])
                    }).sort_values('Impact', ascending=False).head(5)
                    
                    st.dataframe(
                        feature_importance.style.format({'Impact': '{:.4f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    with st.expander("📚 Understanding SHAP Values"):
                        st.markdown("""
                        **How to read these charts:**
                        
                        - **Waterfall Plot**: Shows how each feature pushes the prediction away from 
                          the baseline (average prediction). Features in red increase risk, features in 
                          blue decrease risk.
                        
                        - **Force Plot**: Interactive visualization showing the same information. 
                          Red features push toward higher risk (right), blue features push toward 
                          lower risk (left).
                        
                        - **Feature Impact**: The absolute magnitude of each feature's contribution 
                          to this specific prediction.
                        
                        **Why is this useful?** SHAP helps doctors understand WHY the model made a 
                        particular prediction, making AI decisions more transparent and trustworthy.
                        """)
                    
                except Exception as shap_error:
                    st.warning("⚠️ SHAP explanation generation failed. The prediction is still valid.")
                    st.error(f"Error: {shap_error}")

        except Exception as e:
            st.error("Prediction failed!")
            st.exception(e)

        st.warning("⚠️ **Disclaimer:** This tool is for educational/demo purposes only and is not medical advice.")


# ------------------------------------------------------------------
# About Tab
# ------------------------------------------------------------------
with tab2:
    st.subheader("About CardioRiskIQ")
    
    st.markdown("""
    This application demonstrates a deep learning approach to heart disease risk prediction using 
    clinical features from the UCI Cleveland Heart Disease dataset. The model uses a neural network 
    architecture with regularization techniques to provide robust predictions.
    """)
    
    st.divider()
    
    # Model Performance Section
    st.subheader("📈 Model Performance")
    
    st.markdown("""
    The deployed deep learning model was trained and evaluated on a held-out test set, 
    demonstrating strong predictive performance:
    """)
    
    # Performance metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Test Accuracy",
            value="81.16%",
            help="Percentage of correct predictions on unseen test data"
        )
    
    with col2:
        st.metric(
            label="AUC Score",
            value="91.45%",
            help="Area Under the ROC Curve - measures model's ability to distinguish between classes"
        )
    
    with col3:
        st.metric(
            label="F1 Score",
            value="81.21%",
            help="Harmonic mean of precision and recall"
        )
    
    st.markdown("---")
    
    # Additional metrics table
    st.markdown("**Detailed Classification Metrics (Test Set):**")
    
    metrics_data = {
        "Class": ["No Heart Disease (0)", "Heart Disease (1)", "Overall (Weighted Avg)"],
        "Precision": ["75.71%", "86.76%", "81.80%"],
        "Recall": ["85.48%", "77.63%", "81.16%"],
        "F1-Score": ["80.30%", "81.94%", "81.21%"]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Confusion Matrix Summary:**
    - True Negatives: 53 | False Positives: 9
    - False Negatives: 17 | True Positives: 59
    """)
    
    with st.expander("🔍 What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: Overall correctness of predictions
        - **AUC (Area Under Curve)**: The model's ability to distinguish between disease presence/absence. 
          Values >0.9 are considered excellent.
        - **Precision**: When the model predicts heart disease, how often is it correct?
        - **Recall**: Of all actual heart disease cases, how many did the model catch?
        - **F1-Score**: Balance between precision and recall
        """)
    
    st.divider()
    
    # Technical Details
    st.subheader("🔧 Technical Details")
    
    with st.expander("Model Architecture & Training"):
        st.markdown("""
        **Architecture:**
        - Deep Neural Network with multiple hidden layers
        - Regularization techniques (Dropout, L2 regularization)
        - Trained with early stopping to prevent overfitting
        
        **Preprocessing Pipeline:**
        1. KNN Imputation for missing values
        2. Standard scaling for numerical features
        3. Ordinal encoding for categorical features
        
        **Training Configuration:**
        - Loss function: Binary Cross-Entropy
        - Optimizer: Adam
        - Training/Validation/Test split
        - Final test set never seen during training
        """)
    
    with st.expander("Features Used"):
        st.markdown("""
        The model uses 13 clinical features:
        - **Demographics**: Age, Sex
        - **Symptoms**: Chest Pain Type, Exercise-Induced Angina
        - **Vitals**: Resting Blood Pressure, Max Heart Rate
        - **Lab Results**: Cholesterol, Fasting Blood Sugar
        - **Cardiac Tests**: Resting ECG, ST Depression, ST Slope, Major Vessels (fluoroscopy), Thal Test
        """)
    
    st.divider()
    
    # Baseline Comparison
    st.subheader("📊 Baseline Comparison")
    
    st.markdown("""
    To validate the deep learning approach, a Logistic Regression baseline was trained for comparison:
    """)
    
    comparison_data = {
        "Model": ["Logistic Regression (Baseline)", "Deep Learning (Deployed)"],
        "Accuracy": ["84.06%", "81.16%"],
        "AUC": ["90.22%", "91.45%"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 The deep learning model demonstrates superior AUC performance, indicating better ability to 
    distinguish between risk levels across different decision thresholds. While the baseline logistic 
    regression achieves slightly higher accuracy at the default threshold, the neural network's higher 
    AUC provides more flexibility for clinical decision-making.
    """)
    
    st.divider()
    
    # Developer Info
    st.markdown(
        """
**Developer:** Jackie CW Vescio  
**Project:** CardioRiskIQ (Advanced Track)  
**Dataset:** UCI Cleveland Heart Disease Dataset

This application was built by Jackie CW Vescio with the support of an AI assistant for code review and iteration.  
All modeling decisions, feature definitions, and final deliverables were reviewed and approved by the developer.

---

**Disclaimer:** This tool is for educational and demonstration purposes only. It is not intended for 
medical diagnosis, treatment decisions, or clinical use. Always consult with qualified healthcare 
professionals for medical advice and decisions.
        """
    )