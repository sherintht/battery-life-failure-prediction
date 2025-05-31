import os
import pandas as pd
from scipy.io import loadmat

data_dir = 'D:/Battery_Failure_Prediction/battery_data'
output_dir = 'D:/Battery_Failure_Prediction/processed_all'
os.makedirs(output_dir, exist_ok=True)

all_battery_dfs = []

for filename in os.listdir(data_dir):
    if filename.endswith('.mat'):
        file_path = os.path.join(data_dir, filename)
        battery_id = os.path.splitext(filename)[0]

        print(f"📥 Processing {battery_id}...")
        try:
            mat = loadmat(file_path)
            battery_data = mat[battery_id][0, 0]
            cycles = battery_data['cycle'][0]

            battery_records = []

            for cycle_num, cycle in enumerate(cycles, start=1):
                try:
                    data = cycle['data'][0, 0]
                    voltage = data['Voltage_measured'].flatten()
                    current = data['Current_measured'].flatten()
                    time = data['Time'].flatten()

                    for i in range(len(time)):
                        battery_records.append({
                            'battery_id': battery_id,
                            'cycle_id': cycle_num,
                            'time': time[i],
                            'voltage': voltage[i],
                            'current': current[i]
                        })
                except Exception as e:
                    print(f"⚠️ Skipped cycle {cycle_num} in {battery_id}: {e}")

            # Save individual battery CSV
            df_battery = pd.DataFrame(battery_records)
            output_path = os.path.join(output_dir, f'{battery_id}_all_cycles.csv')
            df_battery.to_csv(output_path, index=False)
            print(f"✅ Saved: {output_path}")

            all_battery_dfs.append(df_battery)

        except Exception as e:
            print(f"❌ Error processing {battery_id}: {e}")

# Optionally combine all batteries into one CSV
df_all = pd.concat(all_battery_dfs, ignore_index=True)
combined_path = os.path.join(output_dir, 'all_batteries_combined.csv')
df_all.to_csv(combined_path, index=False)
print(f"\n📦 All data combined into:\n{combined_path}")