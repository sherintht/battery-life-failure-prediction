import os
import scipy.io
import numpy as np
import pandas as pd
from scipy import interpolate

# Define paths
data_dir = 'D:/Battery_Failure_Prediction/dataset'
output_dir = 'D:/Battery_Failure_Prediction'
output_file = os.path.join(output_dir, 'nasa_battery_data_combined.csv')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of battery files to process
battery_files = ['B0005.mat', 'B0055.mat', 'B0056.mat']
combined_data = []

for battery_file in battery_files:
    # Load the .mat file
    mat_data = scipy.io.loadmat(os.path.join(data_dir, battery_file))
    battery_id = battery_file.split('.')[0]
    battery_data = mat_data[battery_id]['cycle'][0][0][0]  # Access the 'cycle' struct

    # Set ambient temperature based on battery ID
    if battery_id == 'B0005':
        ambient_temperature = 24  # Room temperature (24°C)
    else:  # B0055, B0056
        ambient_temperature = 4   # Low temperature (4°C)

    # Process each cycle
    for cycle_idx, cycle in enumerate(battery_data):
        # Check if the cycle is a discharge cycle
        if cycle['type'][0] != 'discharge':
            continue

        # Extract data arrays
        voltage_data = cycle['data']['Voltage_measured'][0][0][0].flatten()
        current_data = cycle['data']['Current_measured'][0][0][0].flatten()
        temp_data = cycle['data']['Temperature_measured'][0][0][0].flatten()
        time_data = cycle['data']['Time'][0][0][0].flatten()
        capacity_data = cycle['data']['Capacity'][0][0][0].flatten()

        # Ensure arrays are not empty
        if not (len(voltage_data) > 0 and len(current_data) > 0 and len(temp_data) > 0 and len(time_data) > 0):
            continue

        # Calculate cycle number (adding 1 to match typical indexing)
        cycle_number = cycle_idx + 1

        # Calculate averages
        voltage = np.mean(voltage_data)
        current = np.mean(current_data)
        temperature = np.mean(temp_data)
        capacity = capacity_data[0] if len(capacity_data) > 0 else np.nan

        # Calculate time duration
        first_time = time_data[0]
        last_time = time_data[-1]
        time_value = last_time - first_time
        time = time_value if time_value > 0 else np.nan

        # Calculate internal resistance using max voltage drop over max discharge current
        if len(voltage_data) > 1 and len(current_data) > 1:
            voltage_drop = np.max(voltage_data) - np.min(voltage_data)
            discharge_current = np.min(current_data)  # Min current (most negative during discharge)
            if discharge_current < -0.01:  # Threshold to avoid division by near-zero currents
                internal_resistance = abs(voltage_drop / discharge_current)
            else:
                internal_resistance = 0.1  # Default value if current is too small
        else:
            internal_resistance = 0.1

        # Calculate SOC (State of Charge) based on voltage
        soc = (voltage - 3.0) / (4.2 - 3.0)  # Assuming 3.0V is 0% SOC, 4.2V is 100% SOC
        soc = max(0.0, min(1.0, soc))  # Clamp between 0 and 1

        # Calculate SOH (State of Health) based on capacity
        initial_capacity = 2.0  # Assuming initial capacity is 2.0 Ah
        soh = (capacity / initial_capacity) * 100  # In percentage

        # Determine failure (capacity < 1.4 Ah)
        failure = 1 if capacity < 1.4 else 0

        # Append to combined data
        combined_data.append({
            'battery_id': battery_id,
            'cycle': cycle_number,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'capacity': capacity,
            'time': time,
            'failure': failure,
            'ambient_temperature': ambient_temperature,
            'soc': soc,
            'soh': soh,
            'internal_resistance': internal_resistance
        })

# Convert to DataFrame and save to CSV
combined_df = pd.DataFrame(combined_data)
try:
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
except PermissionError as e:
    print(f"PermissionError: Unable to write to {output_file}. {str(e)}")
    print("Please ensure the file is not open in another program and try again.")
except Exception as e:
    print(f"Failed to save combined data: {str(e)}")