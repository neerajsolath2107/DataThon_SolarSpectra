# ☀️ SolarSpectra: Predict Solar Panel Efficiency

## 🌍 Overview
The **SolarSpectra Challenge** is a data science competition focused on **predicting solar panel energy conversion efficiency (%)** under diverse environmental and operational conditions.  
This project applies advanced regression modeling techniques to forecast how efficiently a solar panel converts sunlight into electricity — a key problem in **renewable energy optimization**.

---

## 🧠 Problem Statement
Solar panel efficiency depends on complex, nonlinear interactions among:
- **Sunlight intensity & wavelength**
- **Temperature & humidity**
- **Dust, wind, & cloud cover**
- **Panel material, aging, & degradation**

The task is to **develop a regression model** that accurately predicts the **energy conversion efficiency (%)** from anonymized environmental and physical sensor data.

---

## 📂 Dataset

| File | Description |
|------|--------------|
| `train.csv` | Training dataset containing all features and target variable `efficiency` |
| `test.csv` | Test dataset (without target values) for prediction |
| `sample_submission.csv` | Example submission format |

**Target Variable:**  
`efficiency` → Solar panel energy conversion efficiency (%)

---

## 📊 Evaluation Metric
The competition uses **Root Mean Squared Error (RMSE)** as the evaluation metric:

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y_i} - y_i)^2}
\]

Lower RMSE → Better model performance.

---

## 🧩 Approach

### 1. **Exploratory Data Analysis (EDA)**
- Analyzed feature distributions, missing values, and potential correlations.  
- Visualized relationships between features and target variable.  
- Identified redundant, noisy, and drift-prone features.

### 2. **Data Preprocessing**
- Handled **missing values** using imputation (mean/median for numeric, mode/frequency for categorical).  
- **Standardized numeric features** for model stability.  
- **Encoded textual features** using TF-IDF / embeddings where applicable.  
- **Feature selection** via correlation analysis and feature importance.

### 3. **Feature Engineering**
- Created derived variables (e.g., interaction terms, polynomial features).  
- Extracted latent patterns using PCA / dimensionality reduction when helpful.  
- Detected and handled mild **data drift** between training and test sets.

### 4. **Model Development**
Tested multiple regression algorithms:
- **Linear Regression (Baseline)**
- **Random Forest Regressor**
- **XGBoost / LightGBM**
- **CatBoost**
- **Stacking Ensemble (final model)**

Final submission used **stacked ensemble** combining LightGBM, CatBoost, and XGBoost — tuned using **Optuna** for hyperparameter optimization.

### 5. **Model Validation**
- Employed **K-Fold Cross Validation** (K=5).  
- Ensured consistent RMSE across folds to avoid overfitting.  
- Compared models based on validation RMSE and generalization to test data.

---

## 🚀 Results

| Model | Cross-Validation RMSE | Public LB RMSE | Private LB RMSE |
|--------|----------------------|----------------|----------------|
| Linear Regression | 2.513 | — | — |
| Random Forest | 1.892 | — | — |
| XGBoost | 1.732 | — | — |
| LightGBM | 1.701 | — | — |
| **Stacking Ensemble (Final)** | **1.648** | **1.65** | **1.64** |

✅ **Final Model:** Stacked Ensemble (LightGBM + CatBoost + XGBoost)


