🏦 Loan Approval Prediction 
---
🚀 Machine Learning Project | Model Evaluation + Decision Tree Model + SHAP Interpretation
---
🌟 Level-2 → Task 4  + Bonus Completed ✅

https://elevvoml-loanapproval-prediction.streamlit.app/
___

📌 Task Description
---

Build a machine learning model to predict whether a loan application will be Approved or Rejected based on applicant financial and demographic information.

Objectives:
- Predict loan approval status using classification algorithms
- Handle data preprocessing and categorical encoding
- Address class imbalance using SMOTE
- Compare model performance (Logistic Regression vs Decision Tree)
- Optimize model using hyperparameter tuning
- Focus on evaluation metrics beyond accuracy (Precision, Recall, F1-score, ROC AUC)

 ---
 
📂 Dataset
---

Source: Loan Approval Dataset
Total Records: 4,269 loan applications

🎯 Target Variable:

 - loan_status
   - 0 → Approved
   - 1 → Rejected

📂 Features Used:

    - no_of_dependent
    - education
    - self_employed
    - income_annum
    - loan_amount
    - loan_term
    - cibil_score
    - residential_assets_value
    - commercial_assets_value
    - luxury_assets_value
    - bank_asset_value
 ---
 🧰 Tools & Libraries Used
---
- 🐍 Python
- 📊 Pandas & NumPy
- 📈 Matplotlib & Seaborn
- 📊 Scikit-learn
- 📊 SHAP (Model Explainability)
___

⚙️ Project Workflow
---

<img width="1490" height="1490" alt="image" src="https://github.com/user-attachments/assets/06e55edf-ee2e-453d-96a6-60edc365dd08" />

1️⃣ Data Cleaning & Preprocessing
  - Checked and handled negative values in asset columns
  - Encoded categorical variables (education, self_employed)
  - Feature scaling (where necessary)
  - Removed inconsistencies

2️⃣ Handling Class Imbalance
   - Applied SMOTE (Synthetic Minority Oversampling Technique)
   - Improved recall for minority class
   - Reduced bias toward majority class

3️⃣ Model Training
   - Logistic Regression
   - Decision Tree Classifier

4️⃣ Model Evaluation

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/002f8f63-c673-4258-8fc0-11c395825ee9" />


|Model	| Accuracy	| ROC AUC	| Precision	| Recall	| F1 Score	| F2 Score|
|-------|-----------|---------|-----------|---------|-----------|---------|
|Decision Tree | 96.82%	| 99.23%	| 92.98%	| 99.07%	| 95.93%	| 97.79%|
|Logistic Regression | 92.23% | 97.22% | 87.17% | 93.15% |90.06% | 91.89%|

 ---

📈 Business Insight
---

This project demonstrates how machine learning can improve loan risk assessment.

**Key Insights:**
 - High CIBIL score strongly correlates with loan approval.
 - Applicants with strong asset backing have lower rejection probability.
 - Income-to-loan ratio significantly impacts approval decisions.
 - SMOTE improved detection of minority class (Rejected loans).

**Business Impact:**
 - Faster loan decision-making
 - Reduced default risk
 - Data-driven credit evaluation
 - Improved financial risk management
 ---
 
🧠 Concepts Covered
---

 - Classification (Binary)
 - Decision Tree Algorithm
 - Logistic Regression
 - SMOTE for Imbalanced Data
 - Feature Encoding (Categorical Variables)
 - Model Evaluation Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - F2 Score
    - ROC AUC
 - Hyperparameter Tuning (GridSearchCV)
 - Model Deployment with Streamlit
 ---
 
🚀 Deployment
---

The model is deployed using Streamlit, allowing users to:
 - Input loan application details
 - Receive instant prediction (Approved / Rejected)
 - View probability confidence
 - Analyze approval trends
 - Track prediction history
