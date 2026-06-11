# Advanced Task 2: End-to-End Machine Learning Pipeline

## 📌 Objective
Build a complete, production-ready ML pipeline for **customer churn prediction** with automated preprocessing, model training, hyperparameter tuning, and model export.

---

## 🗂️ Dataset
- **Name:** Telco Customer Churn Dataset
- **Source:** IBM Public GitHub Repository
- **URL:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Size:** 7,043 rows × 21 columns
- **Target:** `Churn` — Yes = customer left, No = customer stayed

---

## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | EDA visualizations |
| `sklearn.pipeline` | End-to-end ML pipeline construction |
| `sklearn.compose` | ColumnTransformer for mixed feature types |
| `sklearn.preprocessing` | StandardScaler, OneHotEncoder |
| `sklearn.impute` | SimpleImputer for missing values |
| `sklearn.ensemble` | RandomForestClassifier |
| `sklearn.linear_model` | LogisticRegression |
| `sklearn.model_selection` | GridSearchCV, train_test_split |
| `joblib` | Model serialization and export |

---

## 📋 Pipeline Architecture

```
Raw Data
   │
   ▼
Data Cleaning (TotalCharges fix, drop customerID, encode target)
   │
   ▼
ColumnTransformer
   ├── Numeric: SimpleImputer(median) → StandardScaler
   └── Categorical: SimpleImputer(most_frequent) → OneHotEncoder
   │
   ▼
Classifier (Logistic Regression / Random Forest)
   │
   ▼
GridSearchCV (Hyperparameter Tuning)
   │
   ▼
Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
   │
   ▼
Export → churn_pipeline.pkl
```

---

## 📋 Steps Performed

### 1. Load & Clean Data
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
```

### 2. EDA
- Bar chart: Churn distribution
- Box plot: Monthly charges vs Churn

### 3. Build Preprocessing Pipeline
```python
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
```

### 4. Train Both Models
```python
lr_pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier', LogisticRegression())])
rf_pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier())])
```

### 5. Hyperparameter Tuning with GridSearchCV
```python
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='f1')
```

### 6. Export Pipeline
```python
joblib.dump(grid_search.best_estimator_, 'churn_pipeline.pkl')
loaded_pipeline = joblib.load('churn_pipeline.pkl')
```

---

## 📊 Evaluation Metrics
| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under ROC curve |
| **Confusion Matrix** | True/False Positive and Negative breakdown |

---

## 📈 Visualizations
- Churn distribution bar chart
- Monthly charges box plot by churn status
- Confusion matrix for tuned Random Forest
- ROC curve with AUC score

---

## 🔍 Key Observations
- Bundling preprocessing inside the Pipeline prevents **data leakage** — a critical mistake beginners commonly make
- Customers with **higher monthly charges** and **month-to-month contracts** show significantly higher churn rates
- GridSearchCV with `cv=3` and `scoring='f1'` ensures the model optimizes for imbalanced classes
- The exported `.pkl` file contains the **full pipeline** — preprocessing + model — so predictions on raw data work out of the box

---

## 📁 Repository Structure
```
Advanced-Task-2-ML-Pipeline/
├── churn_pipeline.ipynb     # Full notebook
├── churn_pipeline.pkl       # Exported trained pipeline
└── README.md
```

---

## ✅ Skills Demonstrated
- End-to-end ML pipeline construction with `sklearn.pipeline`
- Automated preprocessing for mixed numeric/categorical data
- Hyperparameter tuning with GridSearchCV
- Model serialization and production-ready export with joblib

---

## 👨‍💻 Author
**Muhammad Hassaan**
AI/ML Engineering Intern — DevelopersHub Corporation
DHC-488
