# 💳 Credit Card Fraud Detection

## 📌 Overview

This project aims to detect fraudulent credit card transactions using Machine Learning.
It uses a **Random Forest Classifier** to classify transactions as **fraudulent or genuine** based on transaction features.

---

## 🚀 Features

* 📊 Data preprocessing and normalization
* ⚖️ Handles class imbalance in dataset
* 🤖 Machine Learning model (Random Forest)
* 💾 Model saving using `joblib`
* 🌐 Interactive web app using Streamlit
* 📈 Visualization of fraud vs normal transactions

---

## 🧠 How It Works

1. Load transaction dataset
2. Preprocess data (scaling, cleaning)
3. Train model on labeled data
4. Save trained model
5. Use Streamlit app to:

   * Upload new transaction data
   * Predict fraud in real-time
   * Display results and charts

---

## 🏗️ Project Structure

```
Credit Card Fraud Detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd Credit Card Fraud Detection
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train Model

```bash
python src/train.py
```

### 🔹 Run Prediction

```bash
python src/predict.py
```

### 🔹 Run Web App

```bash
streamlit run app.py
```

---

## 📊 Dataset

* Contains anonymized transaction features (`V1–V28`) created using PCA
* Includes `Time`, `Amount`, and `Class` (target variable)

---

## 📈 Evaluation Metrics

* Precision
* Recall
* F1-score

👉 Recall is especially important to detect maximum fraud cases.

---

## 🧪 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit

---

## 🎯 Results

The model successfully classifies transactions and detects fraudulent patterns with high accuracy using ensemble learning.

---

## 📌 Future Improvements

* Add advanced models (XGBoost, Neural Networks)
* Deploy app online
* Improve UI/UX
* Real-time fraud detection system

---

## 👨‍💻 Author

**Achintya Srivastava**

---

## ⭐ Acknowledgement

Dataset inspired by real-world credit card transactions and anonymized using PCA for privacy.
