## 🛡️ FraudShield — Fraudulent Transaction Detection System

FraudShield is an end-to-end machine learning project designed to detect fraudulent customer transactions using the **IEEE-CIS Fraud Detection dataset**, containing more than **590,000 transactions and 400+ features**.

The project covers the complete machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, explainability, and deployment.

### 🔍 Key Features

- Merged transaction and identity datasets containing **590K+ transactions**
- Handled missing values, numerical and categorical features, and high-dimensional data
- Engineered time-based features such as **RelativeHour** and **RelativeWeekday**
- Addressed severe class imbalance using techniques such as **SMOTE**
- Built and compared multiple supervised learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - CatBoost
  - Multi-Layer Perceptron (MLP)
- Implemented unsupervised anomaly detection using:
  - Isolation Forest
  - Autoencoder
- Developed ensemble approaches using:
  - Soft Voting
  - Stacking
- Used **PR-AUC as the primary evaluation metric** because of the highly imbalanced fraud dataset
- Selected classification thresholds using validation-set F1 scores
- Calculated **95% bootstrap confidence intervals** for PR-AUC and ROC-AUC
- Analysed model stability across different transaction time periods
- Used **SHAP** to explain model predictions and identify important fraud indicators

### 📊 Results

The **Stacking Ensemble** achieved the strongest overall performance:

- **PR-AUC:** 0.4792
- **ROC-AUC:** 0.8882

Other strong models included:

- XGBoost — PR-AUC: **0.4599**
- Random Forest — PR-AUC: **0.4573**

### 🌐 Fraud Detection Web Application

A web application was developed using **Flask, Jinja2, JavaScript and Google Cloud Run**.

The application includes:

- Real-time transaction replay using **Server-Sent Events (SSE)**
- Live fraud probability predictions
- Manual transaction scoring
- Fraud analytics dashboard
- Model explainability using SHAP
- API demonstration functionality

### 🛠️ Tech Stack

**Python | Pandas | NumPy | Scikit-learn | XGBoost | CatBoost | PyTorch | SHAP | Flask | JavaScript | HTML/CSS | Google Cloud Run**

### 🎯 Project Goal

The goal of FraudShield is to demonstrate how machine learning, anomaly detection, ensemble learning and explainable AI can be combined to build a practical fraud detection system for highly imbalanced financial transaction data.
