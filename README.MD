# Telecom Customer Churn Prediction with XGBoost

A machine learning project that predicts customer churn for a telecommunications company using the **XGBoost classifier**. The model is trained on the popular Telco Customer Churn dataset and identifies which customers are likely to cancel their service.

## 📊 Dataset

The project uses the **WA_Fn-UseC_-Telco-Customer-Churn** dataset, which contains:
- **7,043 customers** with **20 features** (after dropping `PaymentMethod`)
- A mix of demographic info (gender, senior citizen status, partner, dependents), account info (tenure, contract type, payment details), and service usage (phone, internet, streaming, security add-ons)
- Target variable: **Churn** (Yes/No) — class distribution is imbalanced, with ~73% "No" and ~27% "Yes"

## 🔧 Project Workflow

1. **Data Loading** — Read the CSV with `customerID` as the index.
2. **Exploratory Checks** — Inspect data types, null counts, and churn class balance.
3. **Cleaning & Preprocessing**
   - Dropped the `PaymentMethod` column
   - Fixed inconsistent values in `MultipleLines` (typo correction: `"No phone service"` → `0`)
   - Converted `TotalCharges` to numeric, coercing invalid entries to `NaN`
   - Imputed missing `TotalCharges` values with the column mean
   - Label-encoded binary categorical columns (e.g., gender, Partner, Dependents, PhoneService, etc.)
   - One-hot encoded multi-class categorical columns (`InternetService`, `Contract`) using `pd.get_dummies`
4. **Train/Test Split** — 80/20 split, stratified on the target to preserve class balance.
5. **Model Training** — Fit an `XGBClassifier` (XGBoost) on the training data.
6. **Evaluation** — Assessed performance with classification report, confusion matrix, and accuracy score on both train and test sets.

## 🧠 Model

- **Algorithm:** XGBoost (`XGBClassifier`)
- **Features:** 20 engineered features after encoding
- **Split:** 80% train / 20% test, stratified by `Churn`

## 📈 Results

| Metric | Train Set | Test Set |
|---|---|---|
| Accuracy | 93.5% | 77.9% |
| Precision (Churn=Yes) | 0.90 | 0.60 |
| Recall (Churn=Yes) | 0.85 | 0.50 |
| F1-score (Churn=Yes) | 0.87 | 0.55 |

**Test Set Confusion Matrix:**

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 910 | 125 |
| **Actual Yes** | 186 | 188 |

> ⚠️ **Note:** There's a noticeable gap between train accuracy (93.5%) and test accuracy (77.9%), suggesting the model is **overfitting**. The recall on the minority "Yes" (churn) class is also relatively low (0.50) on test data, meaning the model misses about half of the customers who actually churn. See **Future Improvements** below for ways to address this.

## 🛠️ Tech Stack

- Python
- pandas, NumPy
- scikit-learn (preprocessing, model selection, metrics)
- XGBoost
- matplotlib, seaborn, missingno (visualization & missing data inspection)

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
missingno
scikit-learn
xgboost
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn xgboost
```

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Place the dataset CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) in your working directory and update the file path in the notebook if needed.
3. Open and run the notebook:
   ```bash
   jupyter notebook XGBoost_Telecommunication.ipynb
   ```

## 🔮 Future Improvements

- Address overfitting with regularization (`max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`) and/or early stopping
- Use `GridSearchCV` / `RandomizedSearchCV` for hyperparameter tuning (already imported, not yet applied)
- Handle class imbalance with `scale_pos_weight`, SMOTE, or class-weighted loss
- Apply feature selection (`SelectKBest` is imported but unused) to reduce noise
- Add cross-validation for more robust performance estimates
- Try alternative models (Random Forest, LightGBM, Logistic Regression) as benchmarks

## 📁 Project Structure

```
.
├── XGBoost_Telecommunication.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
└── README.md                          # Project documentation
```

## 📄 License

Add your preferred license here (e.g., MIT).
