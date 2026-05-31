# CreditWise Loan Approval Notebook

This README describes the `CreditWise_Loan.ipynb` notebook, which trains a loan approval prediction model using the `loan_approval_data.csv` dataset.

## Overview

The notebook performs:
- data loading and missing value handling
- exploratory data analysis (EDA)
- categorical encoding
- feature scaling
- model training and evaluation using Logistic Regression

## Dataset

- `loan_approval_data.csv`
- Target column: `Loan_Approved`
- Features include: `Gender`, `Education_Level`, `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Credit_Score`, `DTI_Ratio`, `Savings`, `Loan_Amount`, `Applicant_Income`, `Coapplicant_Income`, `Age`, `Employer_Category`, and more.

## Notebook Steps

1. **Load data**
   - Read the CSV from the notebook directory.

2. **Missing value handling**
   - Numeric columns: impute missing values using `SimpleImputer(strategy='mean')`
   - Categorical columns: impute missing values using `SimpleImputer(strategy='most_frequent')`

3. **Exploratory Data Analysis**
   - Check class balance for loan approval.
   - Visualize distributions of income, credit score, age, DTI ratio, savings, loan amount, and approval outcomes.
   - Use boxplots and histograms to inspect relationships between features and the target.

4. **Feature preparation**
   - Remove `Applicant_ID` because it does not affect loan approval probability.
   - Label-encode `Education_Level` and `Loan_Approved`.
   - One-hot encode categorical columns:
     - `Employment_Status`
     - `Marital_Status`
     - `Loan_Purpose`
     - `Property_Area`
     - `Gender`
     - `Employer_Category`

5. **Correlation analysis**
   - Compute numerical feature correlations.
   - Draw a heatmap to understand relationships with `Loan_Approved`.

6. **Train/test split and scaling**
   - Split data into training and test sets with `test_size=0.2` and `random_state=42`.
   - Standardize features using `StandardScaler`.

7. **Model training and evaluation**
   - Train a `LogisticRegression` model.
   - Evaluate using:
     - precision
     - recall
     - accuracy
     - F1 score
     - confusion matrix

## Requirements

Install the Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Usage

Open `CreditWise_Loan.ipynb` in Jupyter Notebook or VS Code, then run the cells sequentially.

> Make sure `loan_approval_data.csv` is present in the same folder as the notebook.

## Notes

- The notebook focuses on binary classification for loan approval.
- Precision is highlighted as an important metric to reduce false positives in the loan approval decision.
