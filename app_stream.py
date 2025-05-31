import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Streamlit app configuration
st.set_page_config(page_title="Battery Failure Prediction Dashboard", layout="wide")

# Title and description
st.title("🔋 Battery Failure Prediction Dashboard")
st.markdown("""
This dashboard provides real-time battery failure predictions using an ensemble model (XGBoost, One-Class SVM, LSTM). 
Select a battery ID and cycle range to predict failure probabilities and visualize trends.
""")

# Load preprocessed data for reference
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('D:/Battery_Failure_Prediction/nasa_battery_data_preprocessed.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar for user inputs
    st.sidebar.header("Prediction Settings")
    battery_id = st.sidebar.selectbox("Select Battery ID", sorted(df['battery_id'].unique()), index=0)
    cycle_range = st.sidebar.slider("Select Cycle Range", min_value=int(df['cycle'].min()), max_value=int(df['cycle'].max()), value=(2, 248))

    # Filter data based on user selection
    filtered_df = df[(df['battery_id'] == battery_id) & (df['cycle'].between(cycle_range[0], cycle_range[1]))]
    if filtered_df.empty:
        st.warning("No data available for the selected battery ID and cycle range.")
    else:
        # Prepare data for prediction
        features = ['cycle', 'voltage', 'current', 'temperature', 'capacity', 'time', 'internal_resistance']
        input_data = filtered_df[features].values.tolist()

        # Send data to Flask API for prediction
        api_url = "http://localhost:5000/predict"
        payload = {"data": input_data}

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            predictions = response.json().get('predictions', [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the prediction API: {e}")
            predictions = []

        if predictions:
            # Convert predictions to DataFrame
            pred_df = pd.DataFrame(predictions)
            pred_df['battery_id'] = battery_id

            # Display prediction results
            st.subheader("Prediction Results")
            st.dataframe(pred_df[['cycle', 'battery_id', 'predicted_failure', 'ensemble_prob']].style.highlight_max(subset=['ensemble_prob'], color='red'))

            # Visualize failure probability over cycles
            st.subheader("Failure Probability Trend")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=pred_df, x='cycle', y='ensemble_prob', marker='o', ax=ax)
            ax.set_title(f"Failure Probability Over Cycles (Battery ID: {battery_id})")
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Failure Probability")
            ax.grid(True)
            st.pyplot(fig)

            # Display key statistics
            st.subheader("Key Statistics")
            avg_prob = pred_df['ensemble_prob'].mean()
            max_prob = pred_df['ensemble_prob'].max()
            failure_count = pred_df['predicted_failure'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Failure Probability", f"{avg_prob:.2f}")
            col2.metric("Max Failure Probability", f"{max_prob:.2f}")
            col3.metric("Predicted Failures", f"{failure_count}/{len(pred_df)}")

        else:
            st.warning("No predictions received. Please check the API connection and try again.")

# Footer
st.markdown("---")
st.markdown(f"© {datetime.now().year} Battery Failure Prediction Project | Built with Streamlit")