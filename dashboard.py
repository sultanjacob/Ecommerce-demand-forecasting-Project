import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Demand Forecaster", layout="wide")

# --- MODEL ARCHITECTURE (Must match Notebook 5) ---
class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # 1. Load Scaler
    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # 2. Load Model
    model = SalesLSTM(input_size=3, hidden_size=64, output_size=1)
    # Load weights (map_location ensures it works on CPU even if trained on GPU)
    model.load_state_dict(torch.load('data/processed/lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model, scaler

# Load them
try:
    model, scaler = load_resources()
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔮 Scenario Planning")
st.sidebar.write("Adjust market conditions to see AI predictions.")

# User Inputs
sentiment_input = st.sidebar.slider("Expected Customer Sentiment (1-5)", 1.0, 5.0, 4.0)
volume_input = st.sidebar.slider("Expected Daily Order Volume", 0, 100, 30)

# --- MAIN DASHBOARD ---
st.title("🛒 E-Commerce Demand Forecasting AI")
st.markdown("This system uses **Deep Learning (LSTM)** to predict daily sales based on recent trends and customer sentiment.")

# 1. Load Recent Data for Context
df = pd.read_csv('data/processed/daily_sales_sentiment.csv')
df['ds'] = pd.to_datetime(df['ds'])
last_30_days = df.tail(30).copy()

# Plot History
st.subheader("📊 Recent Performance (Last 30 Days)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(last_30_days['ds'], last_30_days['y'], label='Actual Sales', color='green')
ax.set_title("Sales Trend")
st.pyplot(fig)

# --- THE PREDICTION LOGIC ---
if st.button("Predict Next Day Revenue"):
    # 1. Prepare Data
    # We need the last 30 days of real data to make a prediction
    recent_data = last_30_days[['y', 'sentiment_avg', 'sales_volume']].values
    
    # 2. Scale the data
    scaled_recent = scaler.transform(recent_data)
    
    # 3. Create Tensor (1 sample, 30 time steps, 3 features)
    input_tensor = torch.FloatTensor(scaled_recent).unsqueeze(0)
    
    # 4. Make Prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor).item()
        
    # 5. Inverse Scale (Tricky part: we need to inverse transform a shape of (1,3))
    # We construct a dummy row with our predicted sales + the user's inputs
    dummy_row = np.array([[prediction_scaled, 0, 0]]) # 0s are placeholders
    prediction_dollar = scaler.inverse_transform(dummy_row)[0][0]
    
    # Apply a "What-If" adjustment based on user input (Heuristic Layer)
    # The LSTM predicts based on history. We nudge it based on the live slider.
    # If user sets sentiment higher than average (3.0), we boost the prediction slightly.
    base_sentiment = last_30_days['sentiment_avg'].mean()
    sentiment_impact = (sentiment_input - base_sentiment) * 0.05 # 5% impact per star
    final_prediction = prediction_dollar * (1 + sentiment_impact)

    # 6. Display
    st.metric(label="Predicted Revenue for Tomorrow", 
              value=f"${final_prediction:,.2f}", 
              delta=f"{sentiment_impact*100:.1f}% due to Sentiment")

    if sentiment_input < 2.5:
        st.warning("⚠️ Warning: Low sentiment is dragging down revenue forecasts.")
    elif sentiment_input > 4.5:
        st.balloons()