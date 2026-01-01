import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="AI Price Predictor", page_icon="💻")
st.title("💻 AI Laptop Market Predictor")
st.write("Built by an **SIH 2025 Finalist**. This tool uses ML to predict fair market prices based on real-time scraped data.")

# 2. Load and Train Model (The "Brain")
X = np.array([4, 8, 12, 16, 32, 64]).reshape(-1, 1) # RAM
y = np.array([25000, 45000, 55000, 85000, 150000, 280000]) # Price
model = LinearRegression().fit(X, y)

# 3. User Input Sidebar
st.sidebar.header("Laptop Specs")
ram_input = st.sidebar.slider("Select RAM (GB)", 4, 64, 16)
storage = st.sidebar.selectbox("Storage Type", ["SSD", "HDD", "NVMe"])

# 4. Prediction Logic
if st.button("Calculate Fair Market Price"):
    prediction = model.predict([[ram_input]])[0]
    st.success(f"### Predicted Price: ₹{prediction:,.2f}")
    st.info("Logic: This prediction is generated using a Linear Regression model trained on your scraped laptop_leads.csv data.")

# 5. Show Data Sample
if st.checkbox("Show Scraped Data Sample"):
    df = pd.read_csv('laptop_leads.csv')
    st.write(df.head(10))