# Execution
import pandas as pd
import PLANiT_PPA.solarutils as _solar
import sqlite3
"""
Import ASOS data 
"""
# service_key = "FDqMWIRBHMKkx89cHjXmYQxRUf+Y1BJ1ezlwHO4BEkF2NaWo7iS3lV0VdHUERtGTxZdwpj+dIZTdUiinc07UUQ=="
# stations = [
#     "108", "112", "114", "115", "119", "121", "127", "129", "130", "131", "133",
#     "135", "136", "137", "138", "140", "143", "146", "152", "155", "156", "159",
#     "162", "165", "168", "169", "170", "172", "174", "177", "184", "185", "188",
#     "189", "192", "201", "202", "203", "211", "212", "216", "217", "221", "226",
#     "232", "235", "236", "238", "239", "243", "244", "245", "247", "248", "251",
#     "252", "253", "254", "255", "257", "258", "259", "260", "261", "262", "263",
#     "264", "266", "268", "271", "272", "273", "276", "277", "278", "279", "281",
#     "283", "284", "285", "288", "289", "294", "295"
# ]
# start_date = "20210101"
# end_date = "20231231"
# db_file = "ASOS/asos_data.db"
#
# _solar.process_and_save_to_database(service_key, stations, start_date, end_date, db_file)

"""
Calculate statistics
"""
print("Import data and compute statistics")
stats = _solar.compute_stats(
    "ASOS/asos_data.db",
    quantiles=[0.9, 0.99],  # Include these in stats
    plot=True,
    plot_quantiles=[(0.9, 0.99)]  # Plot only these
)

"""
Generate the yearly pattern
"""
print("Generate yearly pattern")
years = range(2023, 2051)
column_name = 'q99'
yearly_pattern_df = _solar.generate_yearly_pattern(stats, column_name=column_name, years=years, normalize=True, plot=False)
yearly_pattern_df.to_sql('solar_patterns', sqlite3.connect('database/solar_patterns.db'), index=False, if_exists='replace')

"""
Add noise to fir the target capacity factor
"""
print("Adjust values to the target capacity factor")

days = _solar.get_days_by_year_range(2023, 2051)
current = yearly_pattern_df[column_name].sum() / (sum(days.values()) * 24)

# Example usage
target_dict = _solar.create_linear_target_dict(2023,
                                               2050,
                                               0.17,
                                               current,
                                               4)

_, adjusted_df = _solar.adjust_column_to_target(
    yearly_pattern_df,
    column_name="q99",
    target=target_dict,
    plot=False
)
print(adjusted_df)


