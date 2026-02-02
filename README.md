#  E-Commerce Demand Forecasting (Multi-Modal AI)

##  Project Overview
This project solves a critical problem in retail operations: **Inventory Management**.
By predicting daily sales volume more accurately, businesses can reduce stockouts (lost revenue) and overstock (storage costs).

Unlike standard time-series models that only look at *history*, this project adds a "Why" factor by analyzing **Customer Sentiment** (NLP) from thousands of reviews to detect demand shifts before they happen.

##  Key Results
We compared a standard industry baseline against our custom AI solution.

| Model | MAE (Mean Absolute Error) | RMSE | Improvement |
|-------|--------------------------|------|-------------|
| **Facebook Prophet (Baseline)** | $8,202 | $10,594 | - |
| **LSTM Neural Network (Proposed)** | **$6,850** | **$8,351** | **+16.48%** |

**Impact:** The LSTM model reduces daily forecasting error by roughly **$1,350 per day**.

<img width="700" height="312" alt="image" src="https://github.com/user-attachments/assets/3d379038-f37a-412a-8e4a-537afab963a2" />


## 🛠️ The Architecture
The system uses a **Multi-Modal** approach, fusing two distinct data pipelines:

1.  **Numerical Pipeline:** Processing 2 years of daily sales data (Lagged features, Volume, Price).
2.  **Linguistic Pipeline (NLP):** Using **BERT (bert-base-multilingual-uncased)** to extract sentiment scores from Portuguese customer reviews.

These streams merge into an **LSTM (Long Short-Term Memory)** network built with **PyTorch** to predict future demand.

## 📂 Project Structure
```text
├── data/               # Raw Olist E-Commerce dataset
├── notebooks/
│   ├── 01_Data_Prep.ipynb    # SQL-style joins & cleaning
│   ├── 02_NLP_Feature_Eng.ipynb # BERT Sentiment Analysis
│   ├── 03_Aggregation.ipynb  # Creating the daily Time-Series
│   ├── 04_Baseline.ipynb     # Facebook Prophet Model
│   └── 05_LSTM_Model.ipynb   # PyTorch LSTM Model
├── src/                # Modular python scripts
└── README.md
