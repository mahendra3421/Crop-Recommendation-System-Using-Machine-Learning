import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout='centered', initial_sidebar_state="collapsed")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%; background-color: #4CAF50; color: white;}
    .stButton>button:hover {background-color: #45a049;}
    .reportview-container {background-color: #f0f8ff;}
    h1 {background: linear-gradient(45deg, #1e8449, #27ae60); padding: 20px; border-radius: 10px; color: white;}
    .stNumberInput>div>div>input {border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title with gradient background
    html_temp = """
    <div style='padding: 1rem;'>
    <h1 style="text-align:center;"> 🌱 Crop Recommendation System</h1>
    <p style="text-align:center; color: #666; font-size: 1.2em;">Discover the perfect crop for your farm using AI</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)

    # Input parameters in columns for better organization
    with col1:
        st.markdown("### 🧪 Soil Parameters")
        N = st.number_input("Nitrogen (N)", 1, 10000)
        P = st.number_input("Phosphorus (P)", 1, 10000)
        K = st.number_input("Potassium (K)", 1, 10000)

    with col2:
        st.markdown("### 🌡️ Environmental Factors")
        temp = st.number_input("Temperature (°C)", 0.0, 100000.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0)

    with col3:
        st.markdown("### 💧 Other Factors")
        ph = st.number_input("pH Level", 0.0, 14.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 100000.0)

    # Create a list of features
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)

    # Add some space before the button
    st.markdown("<br>", unsafe_allow_html=True)

    # Centered button with custom styling
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button('🔍 Get Recommendation'):
            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎯 Recommendation Results")
            st.success(f"Based on your farm's conditions, we recommend growing: **{prediction.item().title()}** 🌾")

    # Add information about the parameters
    with st.expander("ℹ️ Learn about the parameters"):
        st.markdown("""
        - **Nitrogen (N)**: Essential for leaf growth and green vegetation
        - **Phosphorus (P)**: Important for root growth and flower/fruit development
        - **Potassium (K)**: Helps in overall plant health and disease resistance
        - **Temperature**: Optimal temperature range for crop growth
        - **Humidity**: Amount of water vapor in the air
        - **pH Level**: Soil acidity/alkalinity (1-14 scale)
        - **Rainfall**: Annual rainfall in millimeters
        """)

if __name__ == '__main__':
    main()