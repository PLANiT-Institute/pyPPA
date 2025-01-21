# 공시지가 * factor
# 거리에 따른 송전비용 = 개별 격자로부터 centroid (사업지)
# buffer area = 설치 가능 지역 (교집합)
# 넓이로 용량 계산 (600W / 5m2) - 알리오 기반 업데이트
# capital cost of each cell
# capital cost per 1GW (or given range)
# capacity factor from generation pattern
# get LCOE of each MWh (assumed lifespan)

"""
CAPEX calculator by Sanghyun
Date: 2024 Dec. 1st

"""
import pandas as pd
import geopandas as gpd
import numpy as np
import PLANiT_PPA.costutils as _cost

grid = _cost.process_grid_site_data(
    buffer_distance=25000,
    site_path="../gisdata/clusterpolygon.gpkg",
    grid_path="../gisdata/db_semippa.gpkg",
    output_path="../gisdata/grids_within_modified_buffer.gpkg"
)

# Call the function
# For loop to make annual change
cost_df = _cost.calculate_capex_by_grid(
    grid=grid,
    dist=['distance_0', 'distance_1'],
    capex=250_000,                                # KRW / MW
    connection_fees=5_000,                        # KRW / MW / km
    landcost_factor=2,                            # Factor (x landprice)
    power_density=120,                            # W / m^2
    availability=0.5,                             # Available land area factor (x landarea)
    others=[10, 20],                               # Other costs, KRW / MW
    digits=0
)

stats = _cost.analyze_cost_data(
    cost_df=cost_df,
    minimum_capacity=0.1,           # Replace with your minimum capacity
    total_cost_column='total_cost_distance_0',  # Replace with your total cost column
    plot=False  # Set to False if you don't want to generate a plot
)


# Example capacity factors and other inputs
capacity_factors = [0.17, 0.18, 0.19]  # Varying capacity factors over 3 years
capex_values = stats['max_cost']  # CAPEX values from the DataFrame
lifetime = 3  # Lifetime in years
discount_rate = 0.05  # 5% discount rate

# Calculate annualized capital costs
annualised_costs = _cost.annualise_capital_cost(capacity_factors, capex_values, lifetime, discount_rate)
stats['annualised_cost'] = annualised_costs

