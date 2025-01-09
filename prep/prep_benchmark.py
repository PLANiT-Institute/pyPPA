import pandas as pd
import PLANiT_PPA_utils.costutils as _cost

# Example Usage
file_path = './data/global_benchmark_IRENA.xlsx'

# Aggregation methods and percentiles
agg_methods = ['min', 'median', 'max']
# percentiles = [0.25, 0.75]
exclude_filter = {'country': ['China']}
result_df = _cost.process_and_plot_capacity_data(
    file_path=file_path,
    x='capacity_gw',
    y='capital_cost_usd_kw',
    agg_methods=agg_methods,
    n=1,  # Bin width
    group_by_columns=['fuel'],  # Dynamic grouping
    include=None,  # No include filter
    exclude=exclude_filter,  # Exclude filter
    plot=True,
    plot_mv=True,
    moving_avg_window=10,  # Moving average with a window of 10
    percentiles=None,
    axis_limits={'x': [0, 100], 'y': [0, 10000]}
)

# Display the processed DataFrame
print(result_df.head())

