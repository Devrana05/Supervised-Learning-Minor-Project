# CreditWise — Loan Approval Prediction

A machine learning project that predicts loan approval outcomes using classification algorithms. Built as part of a supervised learning study, this project covers the full ML pipeline — from raw data to model comparison and feature engineering.

---

## Project Overview

CreditWise trains and evaluates multiple binary classification models to predict whether a loan application will be approved (`Loan_Approved`: Yes/No). The workflow includes data cleaning, exploratory data analysis, categorical encoding, feature engineering, model training, and performance evaluation across three algorithms.

---

## Dataset

> **Note:** The dataset is not included in this repository. To run the notebook, you will need to provide your own `loan_approval_data.csv` and place it in the project root. Can find on Kaggle.

- **Rows:** 1,000 applicants (50 missing values per column, imputed)
- **Target:** `Loan_Approved` (Yes / No) — imbalanced: ~70% No, ~30% Yes
- **Features:**

| Feature | Type |
|---|---|
| `Applicant_Income` | Numerical |
| `Coapplicant_Income` | Numerical |
| `Age` | Numerical |
| `Dependents` | Numerical |
| `Credit_Score` | Numerical |
| `Existing_Loans` | Numerical |
| `DTI_Ratio` | Numerical |
| `Savings` | Numerical |
| `Collateral_Value` | Numerical |
| `Loan_Amount` | Numerical |
| `Loan_Term` | Numerical |
| `Gender` | Categorical |
| `Education_Level` | Categorical |
| `Employment_Status` | Categorical |
| `Marital_Status` | Categorical |
| `Loan_Purpose` | Categorical |
| `Property_Area` | Categorical |
| `Employer_Category` | Categorical |

---

## Notebook Workflow

1. **Load & Inspect** — Read the CSV, check for nulls, and review descriptive statistics.
2. **Data Cleaning** — Impute missing values: mean for numerical columns, mode for categorical columns using `SimpleImputer`.
3. **EDA** — Visualize class balance, gender split, education levels, and distributions of key numerical features.
4. **Encoding** — Apply Label Encoding to `Education_Level` and `Loan_Approved`; One-Hot Encoding (drop-first) to remaining categorical columns.
5. **Correlation Analysis** — Generate a heatmap to identify features most correlated with loan approval.
6. **Train/Test Split** — 80/20 split with `random_state=42`.
7. **Feature Scaling** — Standardize numerical features using `StandardScaler` (fit on train, transform on test).
8. **Baseline Model Training** — Train and evaluate three classifiers: Logistic Regression, k-Nearest Neighbors (k=13), and Gaussian Naive Bayes.
9. **Feature Engineering** — Add polynomial features (`DTI_Ratio²`, `Credit_Score²`) and retrain all three models to measure improvement.
10. **Model Comparison** — Select the best model based on **precision** (minimising false approvals).

---

## Model Results

### Baseline (no feature engineering)

| Model | Precision | Recall | Accuracy | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.783 | 0.770 | 86.5% | 0.777 |
| k-Nearest Neighbors (k=13) | 0.732 | 0.492 | 79.0% | 0.588 |
| **Naive Bayes** | **0.804** | 0.738 | **86.5%** | **0.769** |

### After Feature Engineering

| Model | Precision | Recall | Accuracy | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.790** | **0.803** | **87.5%** | **0.797** |
| k-Nearest Neighbors (k=13) | 0.737 | 0.459 | 78.5% | 0.566 |
| Naive Bayes | 0.783 | 0.770 | 86.5% | 0.777 |

> **Best model overall: Logistic Regression after feature engineering** — highest precision and accuracy post-engineering. Naive Bayes leads on baseline precision. kNN underperforms due to the high dimensionality of the encoded feature space.

---

## Why Precision?

In a loan approval context, a **false positive** means approving a loan for someone likely to default — a costly mistake for the lender. Precision (minimising false positives) is therefore the primary evaluation metric over accuracy.

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## Usage

1. Clone the repository.
2. Obtain `loan_approval_data.csv` and place it in the project root (not included — see Dataset section).
3. Open `CreditWise_Loan.ipynb` in Jupyter Notebook or VS Code.
4. Run all cells in order from top to bottom.

---

## Key Takeaways

- Mean/mode imputation is a practical strategy for datasets with uniformly distributed missing values.
- One-hot encoding significantly expands the feature space (8 categorical → 17 binary columns), which hurts kNN but is handled well by Logistic Regression and Naive Bayes.
- Polynomial features for `DTI_Ratio` and `Credit_Score` improved Logistic Regression precision and recall, confirming non-linear relationships with the target.
- Precision-recall tradeoffs should drive model selection in financial classification tasks — not accuracy alone.

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn`
