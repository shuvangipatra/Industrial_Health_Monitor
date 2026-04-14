
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import os

model_path = '/content/model.pkl'
scaler_path = '/content/scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
else:
    st.error("Model files not found at /content/. Ensure training code ran successfully.")
    st.stop()

st.set_page_config(page_title="Industrial Health Monitor", layout="wide")
st.title("🏭 Advanced Machine Health & Maintenance AI")

st.sidebar.header("📡 Real-time Sensor Data")
f_val = st.sidebar.slider('Footfall (Load)', 0, 5000, 100)
tm_val = st.sidebar.selectbox('Temperature Mode', [0, 1, 2, 3, 4, 5, 6, 7])
aq_val = st.sidebar.slider('Air Quality (AQ)', 0, 10, 5)
uss_val = st.sidebar.slider('Ultrasonic Sensor (USS)', 0, 10, 5)
cs_val = st.sidebar.slider('Current Usage (CS)', 0, 10, 5)
voc_val = st.sidebar.slider('VOC Gases', 0, 10, 5)
rp_val = st.sidebar.slider('RPM (RP)', 0, 100, 50)
ip_val = st.sidebar.slider('Input Pressure (IP)', 0, 10, 5)
temp_val = st.sidebar.slider('Core Temperature', 0, 50, 25)

input_df = pd.DataFrame([[f_val, tm_val, aq_val, uss_val, cs_val, voc_val, rp_val, ip_val, temp_val]],
                        columns=['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature'])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Machine Status Visualization")
    if temp_val > 40:
        st.warning("⚠️ High Heat detected in Thermal Scan")
    else:
        st.info("ℹ️ Thermal Scan: Optimal")

with col2:
    st.subheader("Diagnostic Metrics")
    health_score = 100 - (temp_val * 1.5 + (10 - aq_val) * 2)
    st.progress(max(0, min(int(health_score), 100)))
    st.write(f"Estimated Health Score: {int(health_score)}%")

if st.button('🚀 Start Deep Diagnostic Analysis'):
    with st.spinner('Analyzing sensor patterns...'):
        time.sleep(1)
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 CRITICAL FAILURE RISK: {prob:.2%}")
            st.subheader("🛠️ Suggested Maintenance Timeline")
            st.write("- **Immediate:** Inspect Current Sensor (CS) connections.")
            st.write("- **Next 24h:** Flush Input Pressure (IP) valves.")
        else:
            st.success(f"✅ SYSTEM STABLE: Failure Risk at {prob:.2%}")
            st.subheader("📅 Maintenance Schedule")
            st.write("- Next Routine Check: 14 Days")
            st.write("- Status: Operating within safe parameters.")
