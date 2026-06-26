# Machine Learning-Based Diabetes Diagnosis

**Bachelor's Thesis Project – Mathematics Engineering**

This repository contains the implementation developed as part of my undergraduate thesis. The project presents an machine learning pipeline for diabetes diagnosis using the **Pima Indians Diabetes Dataset**, with a focus on class imbalance handling, model reliability, explainability, and clinical decision support.

---

## Project Objective

The main objective of this project is to develop a reliable and explainable machine learning model for early diabetes diagnosis.

Unlike conventional classification studies, this thesis focuses not only on predictive performance but also on:

- Handling class imbalance
- Preventing data leakage
- Reliable model evaluation
- Probability calibration
- Clinical threshold optimization
- Explainable Artificial Intelligence (XAI)

---

## Dataset

**Dataset:** Pima Indians Diabetes Database

- 768 patient records
- 8 clinical features
- Binary classification
- Outcome:
  - 0 → Healthy
  - 1 → Diabetes

---

## Project Workflow

```
Data Cleaning
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
SMOTE
      ↓
Nested Cross Validation
      ↓
Hyperparameter Optimization
      ↓
Model Comparison
      ↓
Threshold Optimization
      ↓
Calibration Analysis
      ↓
SHAP Explainability
```

---

## Data Preprocessing

The preprocessing stage includes:

- Missing value handling using median imputation
- Winsorization for extreme insulin values
- StandardScaler feature normalization
- Stratified Train/Test Split
- SMOTE oversampling
- Data leakage prevention using Pipeline

---

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Stacking Ensemble

---

## Advanced Techniques

This thesis incorporates several advanced machine learning techniques:

- Nested Cross Validation
- Grid Search
- Threshold Optimization
- Probability Calibration
- Brier Score
- SHAP Explainability

---

## Final Model

After comparing all candidate models, the final model selected was:

**Balanced Logistic Regression (Threshold = 0.45)**

Reasons:

- High Recall
- Balanced Precision
- Highest AUC
- Reliable Calibration
- Clinically meaningful decision threshold

---

## Results

Main findings include:

- Reliable performance through Nested Cross Validation
- Improved minority class detection using SMOTE
- Clinically optimized threshold selection
- Explainable predictions using SHAP
- Probability reliability verified using Calibration Curve and Brier Score

---

## Repository Structure

```
diabetes-diagnosis-thesis
│
├── data/
├── figures/
├── presentation/
├── src/
├── thesis/
├── requirements.txt
└── README.md
```

---

## Thesis Report

The complete bachelor's thesis is available in:

```
thesis/Bachelor_Thesis.pdf
```

**Language:** Turkish

---

## Presentation

Presentation slides used during the thesis defense are available in:

```
presentation/Thesis_Presentation.pdf
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- SHAP
- Matplotlib
- SciPy

---

## Future Work

Possible future improvements include:

- Deep learning approaches
- Multi-center clinical datasets
- Real-time clinical decision support systems
- Web-based deployment
- External validation on larger datasets

---

## Author

**Merve Hilal**

Bachelor's Degree in Mathematics Engineering
