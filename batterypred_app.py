import streamlit as st
try:
    import pandas as pd
    import numpy as np
    import joblib
    import tensorflow.keras as keras
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import warnings
    import logging
    from xgboost import XGBClassifier
except ModuleNotFoundError as e:
    st.error(f"Missing required module: {str(e)}. Please ensure all dependencies are installed (see requirements.txt).")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings"D:\Battery_Failure_Prediction\models
warnings.filterwarnings("ignore")

# Define paths (relative for Streamlit Cloud)
MODEL_DIR = "D:\Battery_Failure_Prediction\models"
PREDICTIONS_DIR = "D:\Battery_Failure_Prediction\predictions"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model_tuned.joblib")  # Try .joblib first
XGB_JSON_PATH = os.path.join(MODEL_DIR, "xgboost_model_tuned.json")    # Fallback to .json
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "one_class_svm_model_tuned.joblib")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model_tuned.h5")

# Check if model and scaler files exist
for path in [SCALER_PATH, SVM_MODEL_PATH, LSTM_MODEL_PATH]:
    if not os.path.exists(path):
        st.error(f"File not found: {path}. Please ensure all model and scaler files are in the {MODEL_DIR} directory.")
        logger.error(f"File not found: {path}")
        st.stop()

# Check for XGBoost model
xgb_path = XGB_MODEL_PATH if os.path.exists(XGB_MODEL_PATH) else XGB_JSON_PATH
if not os.path.exists(xgb_path):
    st.error(f"XGBoost model not found at {XGB_MODEL_PATH} or {XGB_JSON_PATH}. Please provide the model file.")
    logger.error(f"XGBoost model not found")
    st.stop()

# Load models and scaler
try:
    scaler = joblib.load(SCALER_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    # Load XGBoost model
    if xgb_path.endswith(".json"):
        xgb_model = XGBClassifier()
        xgb_model.load_model(xgb_path)
    else:
        xgb_model = joblib.load(xgb_path)
    logger.info("Models and scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading models or scaler: {str(e)}")
    logger.error(f"Error loading models or scaler: {str(e)}")
    st.stop()

# Create predictions directory if it doesn't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Streamlit app
st.title("Battery Life Predictor")
st.markdown("""
This app checks your battery’s health and predicts potential failures. Enter details like the battery’s capacity (mAh), voltage, and how often you charge it, or upload a CSV for multiple batteries.
""")

# Sidebar for input selection
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select how to provide battery data:", ("Enter Details", "Upload CSV"))

# Features for prediction
features = ['cycle', 'voltage', 'current', 'temperature', 'capacity', 'time', 'internal_resistance']
sequence_length = 20

# Function to preprocess input data
def preprocess_input(data, scaler, is_manual=False):
    try:
        if is_manual:
            # Convert mAh to Ah
            data['capacity'] = data['capacity_mah'] / 1000.0
            # Calculate SOC and SOH
            data['soc'] = (data['voltage'] - 3.0) / (4.2 - 3.0)
            data['soc'] = np.clip(data['soc'], 0, 1)
            data['soh'] = (data['capacity'] / 2.0) * 100  # Assume 2.0 Ah initial capacity
            # Estimate cycle from usage
            data['cycle'] = (data['battery_age_months'] / 12) * 52 * data['charge_frequency']
            data_df = pd.DataFrame([data])[features]
        else:
            data_df = data[features].copy()
            # Convert mAh to Ah if provided
            if 'capacity_mah' in data_df.columns:
                data_df['capacity'] = data_df['capacity_mah'] / 1000.0
            # Calculate SOC and SOH if not provided
            if 'soc' not in data_df.columns:
                data_df['soc'] = (data_df['voltage'] - 3.0) / (4.2 - 3.0)
                data_df['soc'] = np.clip(data_df['soc'], 0, 1)
            if 'soh' not in data_df.columns:
                data_df['soh'] = (data_df['capacity'] / 2.0) * 100
        
        # Scale features
        scaled_data = scaler.transform(data_df[features])
        return scaled_data, data_df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        logger.error(f"Error preprocessing data: {str(e)}")
        return None, None

# Function to create LSTM sequences
def create_lstm_sequences(data, sequence_length):
    try:
        if len(data) < sequence_length:
            # Pad with repeated input for single data points
            data = np.repeat(data, sequence_length, axis=0)
        return np.array([data[-sequence_length:]])
    except Exception as e:
        st.error(f"Error creating LSTM sequences: {str(e)}")
        logger.error(f"Error creating LSTM sequences: {str(e)}")
        return None

# Manual input form
if input_method == "Enter Details":
    st.header("Enter Battery Details")
    with st.form("manual_input_form"):
        st.markdown("**Provide the following details about your battery:**")
        capacity_mah = st.number_input(
            "Battery Capacity (mAh)",
            min_value=100.0,
            max_value=5000.0,
            value=2000.0,
            help="Check the battery label for rated capacity (e.g., 2000 mAh for a typical phone battery)."
        )
        voltage = st.number_input(
            "Voltage (V)",
            min_value=2.0,
            max_value=5.0,
            value=3.7,
            help="Enter the current or nominal voltage (e.g., 3.7V for lithium-ion batteries)."
        )
        battery_age_months = st.number_input(
            "Battery Age (Months)",
            min_value=0.0,
            max_value=120.0,
            value=12.0,
            help="How many months since you started using the battery?"
        )
        charge_frequency = st.number_input(
            "Charges per Week",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            help="How many times do you charge the battery per week? (e.g., 3 for thrice weekly)"
        )
        ambient_temperature = st.selectbox(
            "Ambient Temperature (°C)",
            [4, 24, 30],
            index=1,
            help="Select the typical operating temperature (e.g., 24°C for room temperature)."
        )
        
        # Optional advanced inputs
        st.markdown("**Optional (Leave as default if unknown):**")
        current = st.number_input(
            "Current (A)",
            min_value=-5.0,
            max_value=5.0,
            value=-1.0,
            help="Average discharge current; default is -1.0A."
        )
        time = st.number_input(
            "Discharge Time (Hours)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            help="Time to discharge the battery; default is 1 hour."
        )
        internal_resistance = st.number_input(
            "Internal Resistance (Ohms)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="Battery resistance; default is 0.1 Ohms."
        )
        
        submit = st.form_submit_button("Check Battery Health")

        if submit:
            input_data = {
                'capacity_mah': capacity_mah,
                'voltage': voltage,
                'battery_age_months': battery_age_months,
                'charge_frequency': charge_frequency,
                'temperature': ambient_temperature,
                'current': current,
                'time': time * 3600,  # Convert hours to seconds
                'internal_resistance': internal_resistance
            }
            scaled_data, input_df = preprocess_input(input_data, scaler, is_manual=True)
            if scaled_data is None or input_df is None:
                st.stop()
            
            # XGBoost prediction
            try:
                xgb_prob = xgb_model.predict_proba(scaled_data)[:, 1][0]
                xgb_pred = 1 if xgb_prob >= 0.5 else 0
            except Exception as e:
                st.error(f"Error with XGBoost prediction: {str(e)}")
                logger.error(f"Error with XGBoost prediction: {str(e)}")
                st.stop()
            
            # One-Class SVM prediction
            try:
                svm_pred = svm_model.predict(scaled_data)[0]
                svm_pred = 1 if svm_pred == -1 else 0
                svm_prob = 0.9 if svm_pred == 1 else 0.1
            except Exception as e:
                st.error(f"Error with SVM prediction: {str(e)}")
                logger.error(f"Error with SVM prediction: {str(e)}")
                st.stop()
            
            # LSTM prediction
            try:
                lstm_seq = create_lstm_sequences(scaled_data, sequence_length)
                if lstm_seq is None:
                    st.stop()
                lstm_prob = lstm_model.predict(lstm_seq, verbose=0)[0][0]
                lstm_pred = 1 if lstm_prob >= 0.2 else 0
            except Exception as e:
                st.error(f"Error with LSTM prediction: {str(e)}")
                logger.error(f"Error with LSTM prediction: {str(e)}")
                st.stop()
            
            # Ensemble prediction
            ensemble_prob = 0.5 * lstm_prob + 0.3 * xgb_prob + 0.2 * svm_prob
            ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
            
            # Display results
            st.header("Battery Health Report")
            st.write(f"**Battery Health (SOH):** {input_df['soh'].iloc[0]:.2f}% (Healthy if >70%)")
            st.write(f"**Failure Risk:** {ensemble_prob:.2%} (Low if <50%)")
            st.write(f"**Status:** {'⚠️ Needs Replacement' if ensemble_pred == 1 else '✅ Healthy'}")
            
            # Estimate remaining cycles
            capacity_ah = input_data['capacity_mah'] / 1000.0
            estimated_cycles = input_df['cycle'].iloc[0]
            remaining_cycles = max(0, int((capacity_ah - 1.4) / 0.01))  # Rough estimate
            st.write(f"**Estimated Cycles Completed:** ~{int(estimated_cycles)}")
            st.write(f"**Estimated Remaining Cycles:** ~{remaining_cycles}")
            
            # Visualization: SOH Gauge
            st.header("Battery Health Visualization")
            fig, ax = plt.subplots()
            ax.bar(['Battery Health (SOH)'], [input_df['soh'].iloc[0] / 100], 
                   color='green' if input_df['soh'].iloc[0] > 70 else 'red')
            ax.set_ylim(0, 1)
            ax.set_ylabel("State of Health (%)")
            st.pyplot(fig)
            
            # Visualization: Failure Probability
            fig, ax = plt.subplots()
            ax.bar(['Failure Risk'], [ensemble_prob], 
                   color='orange' if ensemble_prob < 0.5 else 'red')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability of Failure")
            st.pyplot(fig)
            
            # Save prediction
            try:
                result_df = pd.DataFrame({
                    'estimated_cycles': [int(estimated_cycles)],
                    'battery_id': ['User Input'],
                    'actual_failure': [None],
                    'ensemble_predicted_failure': [ensemble_pred],
                    'ensemble_prob': [ensemble_prob],
                    'soh': [input_df['soh'].iloc[0]]
                })
                result_df.to_csv(os.path.join(PREDICTIONS_DIR, "manual_prediction.csv"), index=False)
                st.download_button(
                    label="Download Report",
                    data=result_df.to_csv(index=False).encode('utf-8'),
                    file_name="battery_health_report.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error saving prediction: {str(e)}")
                logger.error(f"Error saving prediction: {str(e)}")

# File upload
else:
    st.header("Upload Battery Data (CSV)")
    st.markdown("Upload a CSV with columns: capacity_mah, voltage, battery_age_months, charge_frequency, temperature. Optional: current, time, internal_resistance.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['capacity_mah', 'voltage', 'battery_age_months', 'charge_frequency', 'temperature']
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain columns: {', '.join(required_columns)}")
            else:
                # Add default values for optional columns
                for col, default in [('current', -1.0), ('time', 3600.0), ('internal_resistance', 0.1)]:
                    if col not in df.columns:
                        df[col] = default
                
                scaled_data, input_df = preprocess_input(df, scaler)
                if scaled_data is None or input_df is None:
                    st.stop()
                
                # Predictions
                xgb_prob = xgb_model.predict_proba(scaled_data)[:, 1]
                xgb_pred = (xgb_prob >= 0.5).astype(int)
                
                svm_pred = svm_model.predict(scaled_data)
                svm_pred = np.where(svm_pred == -1, 1, 0)
                svm_prob = np.where(svm_pred == 1, 0.9, 0.1)
                
                # LSTM predictions
                lstm_prob = []
                for i in range(len(scaled_data) - sequence_length + 1):
                    seq = create_lstm_sequences(scaled_data[i:i+sequence_length], sequence_length)
                    if seq is None:
                        st.stop()
                    lstm_prob.append(lstm_model.predict(seq, verbose=0)[0][0])
                lstm_prob = np.array(lstm_prob)
                lstm_pred = (lstm_prob >= 0.2).astype(int)
                
                # Pad LSTM predictions
                lstm_prob = np.pad(lstm_prob, (sequence_length - 1, 0), mode='constant', constant_values=0)
                lstm_pred = np.pad(lstm_pred, (sequence_length - 1, 0), mode='constant', constant_values=0)
                
                # Ensemble predictions
                ensemble_prob = 0.5 * lstm_prob + 0.3 * xgb_prob + 0.2 * svm_prob
                ensemble_pred = (ensemble_prob >= 0.5).astype(int)
                
                # Results DataFrame
                result_df = pd.DataFrame({
                    'estimated_cycles': input_df['cycle'],
                    'battery_id': df.get('battery_id', ['Unknown'] * len(df)),
                    'actual_failure': df.get('failure', [None] * len(df)),
                    'ensemble_predicted_failure': ensemble_pred,
                    'ensemble_prob': ensemble_prob,
                    'soh': input_df['soh']
                })
                
                # Display results
                st.header("Battery Health Report")
                st.dataframe(result_df)
                
                # Visualizations
                st.header("Visualizations")
                fig, ax = plt.subplots()
                sns.lineplot(data=input_df, x='cycle', y='soh', hue='battery_id', ax=ax)
                ax.set_title("Battery Health (SOH) Over Estimated Cycles")
                ax.set_xlabel("Estimated Cycles")
                ax.set_ylabel("State of Health (%)")
                st.pyplot(fig)
                
                fig, ax = plt.subplots()
                sns.scatterplot(data=input_df, x='soh', y='internal_resistance', hue=ensemble_pred, size=ensemble_pred, ax=ax)
                ax.set_title("Internal Resistance vs. SOH")
                ax.set_xlabel("State of Health (%)")
                ax.set_ylabel("Internal Resistance (Ohms)")
                st.pyplot(fig)
                
                # Download results
                try:
                    st.download_button(
                        label="Download Report",
                        data=result_df.to_csv(index=False).encode('utf-8'),
                        file_name="battery_health_report.csv",
                        mime="text/csv"
                    )
                    result_df.to_csv(os.path.join(PREDICTIONS_DIR, "uploaded_predictions.csv"), index=False)
                except Exception as e:
                    st.error(f"Error saving CSV: {str(e)}")
                    logger.error(f"Error saving CSV: {str(e)}")
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"Error processing CSV: {str(e)}")

# Instructions
st.markdown("""
### Instructions
- **Enter Details**: Provide the battery’s capacity (mAh), voltage, age (months), and charge frequency (e.g., 3 times per week). Leave optional fields as default if unknown.
- **Upload CSV**: Use a CSV with columns: capacity_mah, voltage, battery_age_months, charge_frequency, temperature. Optional: current, time, internal_resistance.
- **Output**: View the battery’s health (SOH), failure risk, and estimated remaining cycles. Download the report as a CSV.
- **Troubleshooting**: Ensure all model files and scaler.joblib are in the models/ directory. Check Streamlit Cloud logs (Manage app > Logs) for errors.
""")