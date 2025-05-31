import streamlit as st
import numpy as np

st.title("🔋 Battery Cycle Life Estimator")

st.markdown("""
This tool provides a simplified estimate of battery cycle life based on user inputs.
""")

# User Inputs
capacity_mah = st.number_input("Battery Capacity (mAh)", min_value=100, max_value=10000, value=2000)
voltage = st.number_input("Nominal Voltage (V)", min_value=2.5, max_value=4.5, value=3.7)
daily_use_time = st.number_input("Daily Use Duration (hours)", min_value=0.1, max_value=24.0, value=2.0)
avg_current = st.number_input("Average Current Draw (A)", min_value=0.1, max_value=5.0, value=0.5)
charging_days_per_week = st.slider("Number of Charging Days per Week", min_value=1, max_value=7, value=5)

# Default assumptions
initial_capacity_Ah = capacity_mah / 1000.0
end_of_life_capacity_Ah = initial_capacity_Ah * 0.7  # 70% of original
soh_drop_per_cycle = 0.05  # % drop per cycle

# Estimate energy used per cycle
energy_per_day = avg_current * voltage * daily_use_time  # in Wh
energy_per_cycle = initial_capacity_Ah * voltage  # Full cycle energy in Wh
cycles_per_day = energy_per_day / energy_per_cycle if energy_per_cycle > 0 else 0.0

# Cycle life estimation
total_cycles = (initial_capacity_Ah - end_of_life_capacity_Ah) / (initial_capacity_Ah * soh_drop_per_cycle / 100)
estimated_life_days = total_cycles / cycles_per_day if cycles_per_day > 0 else 0

# Charging estimation
charges_per_week = cycles_per_day * charging_days_per_week
estimated_weeks = total_cycles / charges_per_week if charges_per_week > 0 else 0

# Output
st.subheader("📈 Estimated Results")
st.write(f"Estimated Battery Cycle Life: **{int(total_cycles)} cycles**")
st.write(f"Expected Operational Duration: **{int(estimated_life_days)} days**")
st.write(f"Charging Occurrences per Week: **{charges_per_week:.2f} times**")
st.write(f"Estimated Weeks Until End of Life: **{int(estimated_weeks)} weeks**")

# Explanation about inactivity
st.subheader("🔋 What if the Battery Isn't Charged for a While?")
st.markdown("""
- If the battery is not charged for a prolonged period, especially when it's already low, it may **deep discharge**, leading to:
  - Permanent capacity loss
  - Internal resistance increase
  - Possible safety issues

- To preserve battery health:
  - Store at ~50% charge if unused for long.
  - Avoid keeping it at 0% or 100% for extended times.
  - Recharge at least once every few weeks.
""")
