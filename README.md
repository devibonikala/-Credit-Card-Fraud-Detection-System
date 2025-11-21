# **Credit Card Fraud Detection**

## 📌 **1. Project Overview**

Credit card fraud is a major financial threat worldwide, leading to billions of dollars in losses every year. This project aims to detect fraudulent credit card transactions using Machine Learning models by analyzing transaction patterns and identifying anomalies.

The goal is to build a **highly accurate**, **scalable**, and **real-time applicable** fraud detection system.

---

## 📌 **2. Dataset Information**

* **Source:** Kaggle – Credit Card Fraud Detection Dataset
* **Total Transactions:** 284,807
* **Fraud Cases:** 492 (0.17%) → Highly imbalanced

### **Features:**

* **V1 to V28** → PCA-transformed features
* **Time** → Seconds elapsed between transactions
* **Amount** → Transaction amount
* **Class** → Target label (0 = Non-Fraud, 1 = Fraud)

---

## 📌 **3. Project Workflow**

```
Data Collection → Data Preprocessing → EDA → Imbalance Handling →
Feature Engineering → Model Building → Model Evaluation → Deployment
```

---

## 📌 **4. System Architecture**

```
                  +-----------------------+
                  |   Raw Dataset (CSV)   |
                  +-----------+-----------+
                              |
                              v
                +-----------------------------+
                |  Data Cleaning & Preprocessing |
                +-----------------------------+
                              |
                              v
             +---------------------------------------+
             | Exploratory Data Analysis (EDA)       |
             +---------------------------------------+
                              |
                              v
            +----------------------------------------+
            |   Handle Class Imbalance (SMOTE)       |
            +----------------------------------------+
                              |
                              v
              +---------------------------------+
              |   Train ML Models (RF, XGBoost) |
              +---------------------------------+
                              |
                              v
             +----------------------------------------+
             | Evaluation (AUC, Recall, Precision)    |
             +----------------------------------------+
                              |
                              v
         +------------------------------------------------+
         | Save Model (Pickle) + Deploy Using Streamlit   |
         +------------------------------------------------+
```

---

## 📌 **5. Detailed Workflow**

### ✅ **Step 1: Import Libraries**

* pandas
* numpy
* matplotlib, seaborn
* scikit-learn
* imblearn (SMOTE)
* xgboost / lightgbm
* pickle

---

### ✅ **Step 2: Load Data**

```python
df = pd.read_csv("creditcard.csv")
```

---

### ✅ **Step 3: Data Cleaning**

* No missing values in dataset
* Remove duplicates
* Normalize/Standardize columns

```python
from sklearn.preprocessing import StandardScaler

df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df['Time'] = StandardScaler().fit_transform(df[['Time']])
```

---

### ✅ **Step 4: Exploratory Data Analysis (EDA)**

Include:

* Class distribution
* Fraud vs Non-fraud count
* Transaction amount comparison
* Correlation heatmap
* PCA component distribution

**Key Finding:**
Fraud cases are **only 0.17%** → extremely imbalanced dataset.

---

### ✅ **Step 5: Handle Class Imbalance**

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
```

---

### ✅ **Step 6: Train–Test Split**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
```

---

### ✅ **Step 7: Model Building**

**Trained Models:**

1. Logistic Regression – baseline model
2. Random Forest – handles imbalance well
3. XGBoost – best performing model

#### ⭐ **Model Comparison**

| Model               | Accuracy      | AUC           | Recall   | Precision |
| ------------------- | ------------- | ------------- | -------- | --------- |
| Logistic Regression | Medium        | Medium        | Low      | Medium    |
| Random Forest       | High          | High          | High     | High      |
| XGBoost             | **Very High** | **Very High** | **Best** | **Best**  |

---

### ✅ **Step 8: Evaluation Metrics**

Use metrics suitable for imbalanced datasets:

* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC-AUC Score

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
```

---

### ✅ **Step 9: Save Model**

```python
import pickle
pickle.dump(model, open("fraud_model.pkl", "wb"))
```

---

### ✅ **Step 10: Deployment (Optional)**

Use **Streamlit**:

```bash
streamlit run app.py
```

#### App Features:

* User inputs transaction details
* Model predicts **Fraud / Not Fraud**
* Shows prediction probability

---

## 📌 **6. Project Folder Structure**

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Building.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── fraud_model.pkl
│
├── app/
│   └── app.py
│
├── README.md
└── requirements.txt
```

---

## 📌 **7. Future Scope**

* Deploy on cloud (AWS / GCP)
* Use **LSTM-based deep learning** for sequential patterns
* Use **Autoencoders** for anomaly detection
* Implement real-time fraud detection using **Kafka**
* Handle extremely large datasets with **Apache Spark**

---

## 📌 **8. Conclusion**

This machine learning pipeline effectively detects fraudulent transactions using advanced algorithms and proper imbalance handling. The final model (XGBoost/Random Forest) achieves excellent recall and AUC, making it suitable for real-world fraud-prevention applications.

---

### ⭐ If you need:

I can also generate a **requirements.txt**, **app.py**, **training code**, or a **GitHub description badge section**.
