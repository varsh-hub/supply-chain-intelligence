# Supply Chain Risk & Disruption Intelligence System

## 📌 Project Overview

The **Supply Chain Risk & Disruption Intelligence System** is a web-based decision-support application developed using **Python, Machine Learning, Data Analytics, and Streamlit**.

The system analyzes historical supply-chain data to identify orders that may have a high probability of late delivery, detect unusual order patterns, classify operational risk, and provide explainable recommendations for supply-chain decision-making.

The application is presented through an interactive **Streamlit web dashboard**.

---

## 🎯 Problem Statement

Modern supply chains generate large amounts of operational data from orders, products, shipping, sales, customers, and logistics activities.

However, simply storing and visualizing this data does not answer important operational questions such as:

- Which orders are likely to experience late delivery?
- Which orders require immediate attention?
- Which orders have unusual patterns?
- What level of risk does an order represent?
- What action should a supply-chain manager consider?

Without intelligent analysis, managers may have to manually inspect large amounts of data or react only after a problem has already occurred.

This project addresses this problem by combining **machine learning, anomaly detection, risk classification, and explainable recommendations** into a single web-based system.

---

## 💡 Proposed Solution

The system transforms historical supply-chain data into actionable risk intelligence.

The main workflow is:


Supply Chain Data
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Machine Learning Risk Prediction
       ↓
Anomaly Detection
       ↓
Risk Engine
       ↓
Recommendation Engine
       ↓
Interactive Streamlit Dashboard


The system therefore moves from:

**Data → Prediction → Risk → Recommendation → Decision Support**

---

## 🧠 Machine Learning Components

### 1. Late-Delivery Risk Prediction

A **Random Forest Classifier** is used to predict the probability of late-delivery risk.

The model learns patterns from historical supply-chain records using features such as:

- Shipping mode
- Scheduled shipping duration
- Order quantity
- Sales
- Profit
- Product category
- Customer segment
- Market
- Region
- Order timing
- Product information

The model produces a probability representing the estimated late-delivery risk.

### 2. Risk Classification

The predicted probability is converted into three business-oriented risk levels:

| Probability | Risk Level |
|---|---|
| `< 0.40` | LOW |
| `0.40 – < 0.70` | MEDIUM |
| `≥ 0.70` | HIGH |

This makes the machine-learning output easier for business users to understand.

### 3. Anomaly Detection

An **Isolation Forest** model is used to identify unusual order patterns.

The anomaly detector analyzes numerical order and business characteristics such as:

- Scheduled shipping duration
- Sales
- Order quantity
- Profit
- Profit ratio
- Product price
- Order timing

The anomaly component identifies observations that differ significantly from normal patterns.

An anomaly does **not** automatically mean that a confirmed real-world disruption has occurred. It indicates that the order or operational pattern deserves further investigation.

### 4. Recommendation Engine

A rule-based recommendation engine converts risk and operational signals into explainable suggestions.

Examples include:

- Prioritize high-risk orders for logistics review.
- Monitor medium-risk shipments.
- Investigate unusual order patterns.
- Review high-risk shipping modes.
- Check fulfillment capacity for high-quantity orders.
- Review negative-profit orders.

The recommendation engine is intentionally rule-based so that the reasoning behind each recommendation is understandable.

---

## 🏗️ System Architecture


                 SUPPLY CHAIN DATA
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                Feature Engineering
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Random Forest          Isolation Forest
       Risk Prediction        Anomaly Detection
              │                     │
              ▼                     ▼
       Risk Probability       Anomaly Score
              │                     │
              └──────────┬──────────┘
                         ▼
                    Risk Engine
                         │
                 LOW / MEDIUM / HIGH
                         │
                         ▼
               Recommendation Engine
                         │
                         ▼
               Streamlit Web Application
                         │
                         ▼
                  Business User


## 🖥️ Application Pages

The Streamlit application contains multiple analytical pages.

### Dashboard

Provides an overall view of:

- Supply-chain KPIs
- Historical delivery risk
- ML risk distribution
- Anomaly statistics
- High-risk orders
- Recent orders
- Recommendations

### Suppliers

Provides supplier-related analysis based on the information available in the dataset.

The current dataset does not contain direct supplier-performance fields such as Supplier ID, supplier reliability, supplier lead time, or supplier capacity. Therefore, the system does not fabricate supplier information.

### Inventory

Provides product and order-activity analysis, including:

- Total orders
- Product activity
- Order quantities
- Sales
- Product status
- Category activity

The current dataset does not contain direct stock-level or warehouse inventory fields such as current stock, reorder point, safety stock, or stockout status.

### Disruptions

Provides anomaly intelligence including:

- Total anomalous orders
- Anomaly percentage
- Normal vs anomalous distribution
- Anomaly score distribution
- Highest-scoring anomalous orders
- Comparison of normal and anomalous observations

### Predictions

Provides machine-learning risk analysis including:

- Risk probability
- LOW/MEDIUM/HIGH classification
- Risk distribution
- Probability distribution
- Threshold analysis
- Highest-risk orders
- Individual order prediction

### Recommendations

Combines:

- ML risk probability
- Risk level
- Anomaly status
- Shipping mode
- Scheduled shipping duration
- Order quantity
- Order profitability

to generate explainable business recommendations.

---

## 📊 Dataset

The project uses the **DataCo Supply Chain Dataset**.

Dataset source:

**Zenodo — DataCo Supply Chain Dataset**

The raw dataset contains approximately:

- **180,519 records**
- **53 original columns**

After preprocessing, the project uses:

- **180,519 records**
- **34 processed columns**

The raw dataset is preserved separately from the processed dataset.

---

## 🔧 Data Preprocessing

The preprocessing pipeline performs:

1. Selection of relevant supply-chain variables.
2. CSV encoding handling.
3. Date conversion.
4. Delivery-delay feature creation.
5. Order-to-shipping duration calculation.
6. Invalid-date handling.
7. Removal of unnecessary fields.
8. Preparation of data for machine-learning analysis.

Two important derived features include:

delivery_delay_days =
actual shipping days - scheduled shipping days


and:


order_to_shipping_days =
shipping date - order date


---

## 🛡️ Data Leakage Prevention

An important part of the project is preventing **target leakage** during machine-learning feature selection.

Variables that describe the delivery outcome itself were excluded from the predictive feature set, including:

- Actual shipping duration
- Delivery status
- Delivery delay
- Shipping date
- Order-to-shipping duration
- Late-delivery-risk target

This ensures that the predictive model does not simply use information that would only become available after the delivery outcome has occurred.

---

## 📈 Model Evaluation

The Random Forest model was selected after comparison with Logistic Regression.

### Random Forest

| Metric | Result |
|---|---:|
| Accuracy | ~76.79% |
| Precision | ~85.34% |
| Recall | ~69.64% |
| F1 Score | ~76.70% |
| ROC AUC | ~86.12% |

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | ~71.50% |
| Precision | ~79.37% |
| Recall | ~64.89% |
| F1 Score | ~71.41% |
| ROC AUC | ~78.08% |

Random Forest was selected because it provided stronger overall predictive performance.

---

## ⚠️ Important Project Limitation

The current system uses **historical supply-chain data**.

Therefore, it should be considered a **decision-support prototype**, not a fully deployed real-time production system.

The dataset does not contain direct information about events such as:

- Supplier shutdowns
- Port closures
- Transportation accidents
- Weather events
- Strikes
- Warehouse congestion
- Real-time inventory
- Live shipment location
- Confirmed disruption events

The anomaly detector therefore identifies **unusual patterns**, rather than claiming that a confirmed real-world disruption has occurred.

---

## 🚀 Future Enhancements

The system can be extended into a real-time enterprise supply-chain intelligence platform by integrating live data sources.

Potential future enhancements include:

- Real-time ERP integration
- Warehouse management system integration
- Transportation management system integration
- Live inventory monitoring
- Supplier reliability data
- Supplier lead-time prediction
- Weather and traffic data
- Transportation disruption feeds
- Real-time shipment tracking
- Automated alerts and notifications
- Time-series forecasting
- Advanced deep-learning models
- Cloud deployment
- Role-based access control

A future architecture could be:


ERP / WMS / TMS / Supplier Data
             ↓
       Real-Time Data Pipeline
             ↓
      ML + Anomaly Detection
             ↓
          Risk Engine
             ↓
      Recommendation Engine
             ↓
       Web Dashboard / Alerts
             ↓
       Supply Chain Manager


---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Random Forest
- Logistic Regression
- Isolation Forest
- StandardScaler
- One-Hot Encoding

### Visualization

- Plotly
- Matplotlib

### Web Application

- Streamlit

### Model Persistence

- Joblib

### Development Environment

- Visual Studio Code
- Python Virtual Environment
- Git
- GitHub

---

## 📁 Project Structure

supply-chain-intelligence/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   └── supply_chain_data.csv
│   └── processed/
│
├── models/
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── risk_engine.py
│   ├── anomaly_detection.py
│   └── recommendations.py
│
├── pages/
│   ├── 1_Suppliers.py
│   ├── 2_Inventory.py
│   ├── 3_Disruptions.py
│   ├── 4_Predictions.py
│   └── 5_Recommendations.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── documentation/
│
├── requirements.txt
└── .gitignore


Generated files such as trained models and processed datasets are excluded from normal Git version control through `.gitignore`.

---

## ▶️ Installation and Setup

### 1. Clone the Repository


git clone <YOUR_GITHUB_REPOSITORY_URL>
cd supply-chain-intelligence


### 2. Create a Virtual Environment


python -m venv venv


### 3. Activate the Virtual Environment

#### Windows PowerShell

.\venv\Scripts\Activate.ps1


### 4. Install Dependencies


pip install -r requirements.txt


### 5. Run the Application


streamlit run app.py


The Streamlit application will open in a web browser.

---

## 🔄 Application Workflow

When the application is running, the overall workflow is:

Raw Supply Chain Data
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Random Forest Prediction
        ↓
Risk Probability
        ↓
Risk Classification
        ↓
Anomaly Detection
        ↓
Recommendation Engine
        ↓
Interactive Dashboard


---

## 🎯 Intended Users

The system is designed as a decision-support tool for:

- Supply Chain Managers
- Logistics Managers
- Operations Managers
- Business Analysts
- Supply Chain Analysts
- Management Teams

The system helps users identify potentially risky orders and unusual patterns without requiring them to manually inspect large volumes of raw supply-chain data.

---

## 📌 Key Contribution

The main contribution of this project is the integration of multiple analytical components into one decision-support application:


Machine Learning
       +
Anomaly Detection
       +
Risk Classification
       +
Explainable Recommendations
       +
Interactive Dashboard


Instead of only displaying historical supply-chain statistics, the system attempts to answer three practical questions:

### 1. What might go wrong?

**Machine-learning risk prediction**

### 2. What looks unusual?

**Anomaly detection**

### 3. What should we consider doing?

**Recommendation engine**

---

## 👨‍💻 Project Status

**Status: Completed**

The current version includes:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Machine-learning model development
- Model comparison
- Model evaluation
- Risk engine
- Anomaly detection
- Recommendation engine
- Streamlit dashboard
- Multi-page application
- Git/GitHub version control

---

## 📄 Academic Project

This project was developed as a **final-year academic project** to demonstrate the practical application of:

- Data analytics
- Machine learning
- Anomaly detection
- Decision-support systems
- Business intelligence
- Web application development

The system demonstrates how historical supply-chain data can be transformed into predictive and actionable intelligence.

---

## 📚 References

- DataCo Supply Chain Dataset — Zenodo
- Scikit-learn documentation
- Pandas documentation
- NumPy documentation
- Streamlit documentation
- Plotly documentation

---

## 👨‍🎓 Author

VARSHINI S K

Final Year Project

**Supply Chain Risk & Disruption Intelligence System Using Machine Learning and Data Analytics**
