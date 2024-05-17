import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'treebag_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('Patient Mortality Prediction')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Data Preview:')
    st.write(data.head())

    data_scaled = scaler.transform(data)
    predictions = model.predict_proba(data_scaled)[:, 1]  # Assuming you want the probability of the positive class

    result = pd.DataFrame({'Prediction': predictions})
    st.write('Predictions:')
    st.write(result)
