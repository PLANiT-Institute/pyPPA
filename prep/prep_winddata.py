import pandas as pd
import geopandas as gpd
import os
import sqlite3
import PLANiT_PPA_utils.costutils as _cost
import PLANiT_PPA_utils.solarutils as _solar
import PLANiT_PPA_utils.windutils as _wind

"""
Wind Data Preprocessing
"""

plot = False
rewrite = False
rec_lt = [0, 80000]

folder_path = 'winddata/'
dataframes = []

# Read all CSVs, rename columns, concatenate
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, encoding='cp949')
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)

# Filter rows for specified regions
concatenated_df = concatenated_df[concatenated_df['col_2'].str.contains('인천광역시|인천 광역시|경기도|충청남도')]

# Drop unnecessary columns
concatenated_df.drop(columns=['col_3'], inplace=True)

# Rename
concatenated_df.rename(
    columns={'col_0': 'date', 'col_1': 'time', 'col_2': 'region', 'col_4': 'wind_speed'},
    inplace=True
)

# Adjust time
concatenated_df['time'] = concatenated_df['time'] - 1
concatenated_df['time'] = pd.to_numeric(concatenated_df['time'], errors='coerce').fillna(0).astype(int).clip(0, 23)

# Convert to datetime
concatenated_df['datetime'] = pd.to_datetime(
    concatenated_df['date'] + ' ' + concatenated_df['time'].astype(str).str.zfill(2) + ':00',
    format='%Y-%m-%d %H:%M',
    errors='coerce'
)
concatenated_df.dropna(subset=['datetime'], inplace=True)

# Convert wind_speed to numeric
concatenated_df['wind_speed'] = (
    concatenated_df['wind_speed'].astype(str).str.strip()
)
concatenated_df['wind_speed'] = pd.to_numeric(
    concatenated_df['wind_speed'], errors='coerce'
).fillna(0)

# Extract year and month
concatenated_df['year'] = concatenated_df['datetime'].dt.year
concatenated_df['month_of_year'] = concatenated_df['datetime'].dt.month

# Compute monthly max by (year, month, region)
concatenated_df['monthly_max'] = (
    concatenated_df
    .groupby(['region', 'year', 'month_of_year'])['wind_speed']
    .transform('max')
)

# Normalize by monthly max
concatenated_df['normalized_wind_speed'] = (
    concatenated_df['wind_speed'] / concatenated_df['monthly_max']
).fillna(0)

# (Optional) if you want a single pattern, group by day_of_year, hour, etc.
concatenated_df['day_of_year'] = concatenated_df['datetime'].dt.strftime('%m-%d')
concatenated_df['hour'] = concatenated_df['datetime'].dt.hour

annual_pattern = (
    concatenated_df
    .groupby(['region', 'day_of_year', 'hour'])
    .agg({'normalized_wind_speed': 'mean'})
    .reset_index()
)

pivoted_annual_pattern = annual_pattern.pivot(
    index=['day_of_year', 'hour'],
    columns='region',
    values='normalized_wind_speed'
).reset_index()

# Create a 'datetime' index using a placeholder year (e.g. 2024)
pivoted_annual_pattern['datetime'] = pd.to_datetime(
    '2024-' + pivoted_annual_pattern['day_of_year'] + ' ' +
    pivoted_annual_pattern['hour'].astype(str).str.zfill(2) + ':00',
    format='%Y-%m-%d %H:%M'
)
pivoted_annual_pattern.set_index('datetime', inplace=True)
pivoted_annual_pattern.drop(columns=['day_of_year', 'hour'], inplace=True)

# Rename columns for clarity
pivoted_annual_pattern.rename(
    columns={'경기도': 'Gyeonggi-do', '충청남도': 'Chungcheongnam-do'},
    inplace=True
)

for col in pivoted_annual_pattern.columns:
    pivoted_annual_pattern[col] = pivoted_annual_pattern[col]/pivoted_annual_pattern[col].max()
    print(f"Capacity Factor: {pivoted_annual_pattern[col].sum() / 8760}")

print(pivoted_annual_pattern.head())

target_dict = _solar.create_linear_target_dict(2023,
                                               2050,
                                               0.23,
                                               0.4,
                                               4)

extended_df = _wind.extend_hourly_pattern_to_target_years(pivoted_annual_pattern, start_year=2023, end_year=2050, plot = plot)


# Adjust the pattern to match target capacity factors
_, adjusted_pattern = _wind.adjust_hourly_pattern_to_targets(extended_df, target_dict, plot=plot)

# Save adjusted pattern to SQLite database
conn = sqlite3.connect('database/wind_patterns.db')
adjusted_pattern.to_sql('wind_patterns', conn, if_exists='replace')
conn.close()

gdf = gpd.read_file('gisdata/wind_feasible.gpkg')
gdf['LCOE'] = gdf['LCOE'] * 1000

dt = {}

for rec in rec_lt:
    wind_df = pd.DataFrame(gdf.drop(columns='geometry'))
    wind_df['capacity'] = 5
    wind_df['LCOE+REC'] = wind_df['LCOE'] + wind_df['rec_weight'] * rec

    # Initialize DataFrames to store results
    windLCOE_dfs = []
    windLCOEREC_dfs = []

    # Parameters for the analysis
    min_capacity = 0.1  # Minimum capacity threshold
    bin_size = 0.1  # Bin size for grouping
    plot = plot  # Set to True to generate plots

    # Perform the cost analysis for LCOE+REC
    windcost_dt = {}

    # Analyze costs for each region
    windcost_dt = {
        region: _cost.analyze_cost_data(
            cost_df=wind_df[wind_df['admin_boundaries'] == region],
            minimum_capacity=0.1,
            total_cost_column='LCOE+REC',
            bin=0.1,
            plot=plot
        )
        for region in wind_df['admin_boundaries'].unique()
    }

    # Store results in the dictionary
    result_df = pd.concat(windcost_dt).reset_index().rename(columns={'level_0': 'region'}).drop(columns='level_1')
    result_df['REC'] = rec  # Add REC value to the DataFrame
    dt[rec] = result_df

# Combine all results into a single DataFrame
windoutput_df = pd.concat(dt).reset_index().drop(columns=['level_0', 'level_1'])
windoutput_df['annualised_cost'] = windoutput_df['max_cost'].copy()

