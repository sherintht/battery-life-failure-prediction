from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import load_model
import joblib
import logging
import traceback
import sys

# Configure logging at the top
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load preprocessed data for scaler fitting
try:
    df = pd.read_csv('D:/Battery_Failure_Prediction/nasa_battery_data_preprocessed.csv')
    features = ['cycle', 'voltage', 'current', 'temperature', 'capacity', 'time', 'internal_resistance']
    X = df[features].values
    scaler = MinMaxScaler()
    scaler.fit(X)
    logger.info("Scaler fitted successfully from preprocessed data.")
except Exception as e:
    logger.error(f"Error loading data or fitting scaler: {e}\n{traceback.format_exc()}")
    sys.exit(1)

xgb = None
oc_svm = None
lstm = None
try:
    xgb = XGBClassifier()
    try:
        xgb.load_model('D:/Battery_Failure_Prediction/models/xgboost_model_tuned.json')
        logger.info("XGBoost model loaded successfully from .json.")
    except Exception as e:
        logger.warning(f"Failed to load XGBoost .json: {e}. Trying .joblib.")
        xgb = joblib.load('D:/Battery_Failure_Prediction/models/xgboost_model_tuned.joblib')
        logger.info("XGBoost model loaded successfully from .joblib.")
except Exception as e:
    logger.error(f"Failed to load XGBoost model: {e}\n{traceback.format_exc()}")
    sys.exit(1)

try:
    oc_svm = joblib.load('D:/Battery_Failure_Prediction/models/one_class_svm_model_tuned.joblib')
    logger.info("One-Class SVM model loaded successfully.")
except Exception as e:
    logger.warning(f"Failed to load One-Class SVM model: {e}\n{traceback.format_exc()}. Proceeding with XGBoost only.")

try:
    lstm = load_model('D:/Battery_Failure_Prediction/models/lstm_model_tuned.h5')
    logger.info("LSTM model loaded successfully.")
except Exception as e:
    logger.warning(f"Failed to load LSTM model: {e}\n{traceback.format_exc()}. Proceeding without LSTM.")

sequence_length = 20 if lstm is not None else 1

@app.route('/')
def home():
    return "Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug("Received prediction request.")
        data = request.get_json(force=True)
        if not data or 'data' not in data:
            logger.error("No data provided in request.")
            return jsonify({'error': 'No data provided'}), 400
        input_data = np.array(data['data'])
        logger.debug(f"Input data: {input_data[:5]}...")  # Log first 5 rows for debugging
        
        if input_data.shape[1] != len(features):
            logger.error(f"Expected {len(features)} features, got {input_data.shape[1]}")
            return jsonify({'error': f'Expected {len(features)} features, got {input_data.shape[1]}'}), 400
        if lstm is not None and len(input_data) < sequence_length:
            logger.error(f"Input data has {len(input_data)} samples, needs at least {sequence_length} for LSTM")
            return jsonify({'error': f'Input data must have at least {sequence_length} samples for LSTM'}), 400

        input_scaled = scaler.transform(input_data)

        result = []
        if xgb is not None:
            xgb_pred_prob = xgb.predict_proba(input_scaled)[:, 1]
            xgb_pred = (xgb_pred_prob > 0.5).astype(int)
            for i, pred in enumerate(xgb_pred):
                entry = {
                    'cycle': int(input_data[i, 0]),  # Ensure cycle is taken from input_data
                    'predicted_failure': int(pred),
                    'xgb_prob': float(xgb_pred_prob[i])
                }
                result.append(entry)

        if oc_svm is not None and lstm is not None:
            X_seq = []
            for i in range(len(input_scaled) - sequence_length + 1):
                X_seq.append(input_scaled[i:i + sequence_length])
            X_seq = np.array(X_seq)

            oc_svm_pred = oc_svm.predict(input_scaled)
            oc_svm_pred_prob = np.where(oc_svm_pred == -1, 0.9, 0.1)
            lstm_pred_prob = lstm.predict(X_seq, verbose=0).flatten()

            xgb_test_prob = xgb_pred_prob[sequence_length - 1:]
            oc_svm_test_prob = oc_svm_pred_prob[sequence_length - 1:]
            ensemble_prob = 0.5 * lstm_pred_prob + 0.3 * xgb_test_prob + 0.2 * oc_svm_test_prob
            ensemble_pred = (ensemble_prob > 0.5).astype(int)  # Lowered threshold to 0.5

            # Update result with ensemble predictions
            for i, pred in enumerate(ensemble_pred):
                cycle_idx = sequence_length - 1 + i
                result[cycle_idx].update({
                    'ensemble_predicted_failure': int(pred),
                    'ensemble_prob': float(ensemble_prob[i])
                })
                logger.debug(f"Cycle {cycle_idx}: Ensemble prob {ensemble_prob[i]}")

            # Add default ensemble values for earlier cycles (before sequence_length)
            for i in range(sequence_length - 1):
                result[i].update({
                    'ensemble_predicted_failure': result[i]['predicted_failure'],  # Use XGBoost prediction
                    'ensemble_prob': float(xgb_pred_prob[i])  # Use XGBoost probability
                })

        if not result:
            logger.error("No predictions generated.")
            return jsonify({'error': 'No predictions generated'}), 500

        logger.info(f"Prediction successful for {len(result)} cycles.")
        return jsonify({'predictions': result})

    except Exception as e:
        logger.error(f"Error in prediction: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    use_reloader = 'IPYTHON' not in dir() and 'JUPYTER' not in dir()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=use_reloader)