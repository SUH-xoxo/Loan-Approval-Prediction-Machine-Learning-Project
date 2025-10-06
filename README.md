
# 🧾 Loan Approval Prediction — Machine Learning Project

## 📘 Project Overview
This project predicts whether a **loan application will be approved or rejected** based on applicant details such as income, credit history, and property area.  
It demonstrates **data preprocessing, class imbalance handling (SMOTE)**, and **model comparison** between **Logistic Regression** and **Decision Tree** classifiers using Python and Scikit-learn.

---

## 🎯 Objective
To build a predictive machine learning model that determines **loan approval status** and evaluate performance on **imbalanced data** using metrics beyond accuracy — including **Precision, Recall, and F1-score**.

---

## 🧠 Dataset
**Name:** Loan Approval Prediction Dataset  
**Source:** Kaggle — [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

**Features:**
| Column | Description |
|---------|--------------|
| Gender | Applicant Gender |
| Married | Marital Status |
| Dependents | Number of Dependents |
| Education | Graduate/Not Graduate |
| Self_Employed | Employment Status |
| ApplicantIncome | Applicant’s Monthly Income |
| CoapplicantIncome | Co-applicant’s Income |
| LoanAmount | Loan Amount |
| Loan_Amount_Term | Duration of Loan (Months) |
| Credit_History | 1 = Good Credit, 0 = Bad Credit |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target Variable (Y = Approved, N = Rejected) |

---

## ⚙️ Technologies Used
| Category | Libraries / Tools |
|-----------|------------------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Modeling | Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |

---

## 🧩 Machine Learning Workflow
1. **Data Loading & Exploration**
   - Import dataset and inspect missing values and data types.  
2. **Data Preprocessing**
   - Fill missing values.  
   - Encode categorical variables.  
   - Standardize numeric features.  
3. **Train-Test Split**
   - 80% training, 20% testing.  
4. **Class Imbalance Correction**
   - Apply **SMOTE** to oversample the minority class.  
5. **Model Building**
   - Train two models:
     - Logistic Regression  
     - Decision Tree Classifier  
6. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score.  
   - Visuals: Confusion Matrix and ROC Curves.  
7. **Model Comparison**
   - Evaluate both models side by side for better interpretability.  

---

## 📊 Evaluation Metrics
| Metric | Description |
|---------|--------------|
| Accuracy | Overall correctness of predictions |
| Precision | Correct positive predictions out of total predicted positives |
| Recall | Correct positive predictions out of total actual positives |
| F1-Score | Balance between Precision and Recall |
| ROC-AUC | Probability that the model ranks a random positive higher than a random negative |

---

🔍 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	0.932	0.908	0.913	0.910
Decision Tree	0.977	0.975	0.963	0.969
*(Values may vary depending on random seed and dataset distribution.)*

---

## 🧮 Bonus Enhancements
✅ Apply **GridSearchCV** for hyperparameter tuning.  
✅ Add **Feature Importance Plot** for interpretability.  
✅ Export trained model using `joblib` for deployment.

---

## 🚀 How to Run the Project
1. Clone this repository or download the notebook and dataset.  
2. Place your dataset file as `loan_approval_dataset.csv` in the same directory.  
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
4. Open Jupyter Notebook and run:
   ```bash
   Loan_Approval_Prediction.ipynb
   ```
5. Execute each cell in order.

---

## 🧑‍💻 Author
**Saad Ul Hassan**  
Bachelor of Science in Artificial Intelligence  
University of Kotli, AJK  

---

## 🏁 Conclusion
- **Logistic Regression** performed slightly better on generalization.  
- **Decision Tree** captured non-linear patterns but showed minor overfitting.  
- Using **SMOTE** effectively handled imbalance and improved recall for the minority class.  
