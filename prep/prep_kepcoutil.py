import PLANiT_PPA.KEPCOutils as _kepco
import pandas as pd

# Example usage
filepath = "database/KEPCO.xlsx"  # Replace with the actual file path
year = 2023  # User-provided year
selected_sheet = "HV_C_I"  # User-selected rates sheet

temporal_df, contract_fee = _kepco.process_kepco_data(filepath, year, selected_sheet) # rate krw/kWh

# Example usage
num_years = 28  # Number of years to generate
rate_increase = 0.05  # 5% annual increase

multiyearusagefees_df, contractfees_df = _kepco.multiyear_pricing(temporal_df, contract_fee, year, num_years, rate_increase)

# Define variables again for this environment
capacity_factor = 0.8
contract_capacity = 3000000  # 3 GW in kW
flat_demand = contract_capacity * capacity_factor  # kWh per hour
annual_generation = contract_capacity * capacity_factor * 8760

contractfees_df['rate'] * contract_capacity
df = pd.DataFrame(multiyearusagefees_df['rate'] * flat_demand)
df['year'] = df.index.year
df = df.groupby('year').sum().astype(float).round(0)

levelised_fees = (df / annual_generation).round(2)