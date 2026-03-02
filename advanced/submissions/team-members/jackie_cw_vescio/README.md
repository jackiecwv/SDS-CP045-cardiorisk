# 🫀 CardioRiskIQ - Advanced Heart Disease Risk Prediction

**Developer:** Jackie CW Vescio  
**Project:** SDS-CP045 CardioRisk (Advanced Track)  
**Institution:** SuperDataScience  
**Date:** March 2026

---

## 📋 Project Overview

CardioRiskIQ is an advanced deep learning application for predicting heart disease risk using clinical features from the UCI Cleveland Heart Disease dataset. The project features a professional Streamlit web interface with **SHAP (SHapley Additive exPlanations)** for model interpretability, meeting Advanced Track requirements for explainable AI.

**Key Achievement:** 81.16% accuracy with 91.45% AUC using a deep neural network with advanced preprocessing and feature engineering.

---

## ✨ Key Features

### 🧠 Advanced Machine Learning
- **Deep Neural Network (ANN)** with multiple hidden layers
- **Advanced Preprocessing Pipeline:**
  - KNN imputation for missing values
  - Standard scaling for numerical features
  - Ordinal encoding for categorical features
- **Regularization Techniques:** Dropout and L2 regularization to prevent overfitting
- **Early Stopping:** Automated training optimization

### 🔍 **SHAP Explainability (Advanced Track)**
- **Waterfall Plots:** Visual explanation of how each feature impacts predictions
- **Force Plots:** Interactive visualization of feature contributions
- **Feature Importance Rankings:** Top 5 most important features per patient
- **Clinical Transparency:** Helps doctors understand WHY the model makes predictions

### 🎯 Interactive Web Application
- **Professional Streamlit Interface** with intuitive design
- **Sample Patient Profiles:** Three pre-loaded patients (Low, Moderate, High Risk)
- **Risk Visualization Gauge:** Color-coded risk assessment (Green/Yellow/Red)
- **PDF Report Generation:** Downloadable prediction reports with patient details
- **Adjustable Decision Threshold:** Customize sensitivity vs. specificity tradeoff

### 📊 Model Performance Dashboard
- **Comprehensive Metrics Display:** Accuracy, AUC, F1-Score, Precision, Recall
- **Confusion Matrix Summary:** Detailed classification breakdown
- **Baseline Comparison:** Logistic Regression vs. Deep Learning performance

---

## 📈 Model Performance

### Test Set Results (138 patients, never seen during training)

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.16% |
| **AUC Score** | 91.45% |
| **F1 Score** | 81.21% |
| **Precision (Disease)** | 86.76% |
| **Recall (Disease)** | 77.63% |

### Confusion Matrix
- **True Negatives:** 53 | **False Positives:** 9
- **False Negatives:** 17 | **True Positives:** 59

### Baseline Comparison
| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression (Baseline) | 84.06% | 90.22% |
| **Deep Learning (Deployed)** | **81.16%** | **91.45%** |

*Note: The deep learning model achieves superior AUC, indicating better ability to distinguish between risk levels across different decision thresholds.*

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.9+**
- **TensorFlow/Keras:** Deep learning model
- **Streamlit:** Web application framework
- **SHAP:** Model explainability
- **Scikit-learn:** Preprocessing and evaluation
- **Pandas/NumPy:** Data manipulation

---

## 📦 Installation & Usage

### Running the Streamlit App
```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

#### **Quick Start with Sample Patients:**
1. Click one of the three sample patient buttons:
   - 🟢 **Low Risk** (9.6% predicted risk)
   - 🟡 **Moderate Risk** (60.8% predicted risk)
   - 🔴 **High Risk** (96.2% predicted risk)

2. Click **"Predict Risk"**

3. View results:
   - Risk gauge visualization
   - Probability score
   - SHAP explanation (wait 10-30 seconds for first prediction)

#### **Understanding SHAP Explanations:**
- **Waterfall Plot:** Shows how each feature pushes risk up (red) or down (blue)
- **Force Plot:** Interactive visualization - hover over features to see values
- **Top 5 Features:** Most important features for this specific patient

---

## 📊 SHAP Explainability Examples

### Low Risk Patient (9.6% risk)
**Key Protective Factors:**
- No vessels visible on fluoroscopy (ca_idx = 0): -0.27 impact
- No exercise-induced angina (exang_idx = 0): -0.18 impact
- Low ST depression (oldpeak = 0.0): -0.14 impact

### High Risk Patient (96.2% risk)
**Key Risk Factors:**
- High ST depression (oldpeak = 2.5): +0.13 impact
- Downsloping ST segment (slope = 0): +0.12 impact
- Exercise-induced angina (exang = 1): +0.28 impact

*SHAP values show clinical reasoning behind each prediction, making AI decisions transparent and trustworthy.*

---

## 🎯 Advanced Track Compliance

This project meets **Advanced Track** requirements:

✅ **Deep Learning Implementation:** Custom ANN with multiple hidden layers  
✅ **Advanced Preprocessing:** KNN imputation + feature engineering  
✅ **Model Explainability:** SHAP integration with visual explanations  
✅ **Professional Deployment:** Production-ready Streamlit application  
✅ **Comprehensive Documentation:** Detailed README, code comments, notebook  
✅ **Model Evaluation:** Multiple metrics, baseline comparison, test set validation  
✅ **User Experience:** Sample patients, PDF reports, interactive visualizations  

---

## 👨‍💻 Developer Information

**Jackie CW Vescio**  
Data Science Professional | Advanced Track Participant

**Acknowledgments:**  
This application was built with the support of an AI assistant for code review, debugging, and iteration. All modeling decisions, feature engineering, architecture choices, and final deliverables were reviewed and approved by the developer.

---

## ⚖️ Disclaimer

**This tool is for educational and demonstration purposes only.**  

CardioRiskIQ is NOT intended for:
- Medical diagnosis
- Treatment decisions
- Clinical use
- Replacement of professional medical advice

**Always consult qualified healthcare professionals for medical decisions.**

---

## 🎉 Project Status

**Status:** ✅ **COMPLETE - Advanced Track**  
**Submission Date:** March 2026  
**Key Achievement:** Production-ready AI application with explainability

---

**Thank you for reviewing CardioRiskIQ!** 🫀💙