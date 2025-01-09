import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle


def process_and_interpolate_annual_data(path, scenario_name):
    """
    Processes an Excel file, extracts data for a specified scenario,
    and interpolates values to fill in annual data.

    Args:
        path (str): Path to the Excel file.
        scenario_name (str): Name of the scenario to process.

    Returns:
        pd.DataFrame: A DataFrame with interpolated annual data.
    """
    # Read the Excel file
    # path = 'database/NGFS_carbonprice.xlsx'
    # scenario_name = 'Net Zero 2050'
    df = pd.read_excel(path, index_col=0)
    df = df[scenario_name]

    # Generate a range of all annual years
    annual_years = pd.DataFrame({'Year': range(int(df.index.min()), int(df.index.max()) + 1)})
    annual_years.set_index('Year', inplace=True)

    # Merge df with annual_years using the index
    df = pd.DataFrame(df)  # Convert Series to DataFrame
    df.columns = ['value']  # Name the column
    interpolated_data = annual_years.join(df, how='left')

    # Interpolate the missing values
    interpolated_data['value'] = interpolated_data['value'].interpolate(method='linear')

    return interpolated_data


def process_grid_site_data(buffer_distance, site_path, grid_path, output_path=None):
    """
    Processes grid and site GeoDataFrames to calculate distances between centroids
    of grid polygons and buffer areas around site polygons.

    Parameters:
    - buffer_distance (float): Buffer distance in meters around site polygons.
    - site_path (str): Path to the site polygon GeoDataFrame file.
    - grid_path (str): Path to the grid polygon GeoDataFrame file.
    - output_path (str, optional): Path to save the resulting GeoDataFrame. If None, the result is not saved.

    Returns:
    - grid_within_buffer (GeoDataFrame): GeoDataFrame with grid polygons and distance columns to each site centroid.
    """

    # Load GeoDataFrames
    site_poly = gpd.read_file(site_path)
    grid_poly = gpd.read_file(grid_path)

    # Drop rows with missing geometries from grid_poly
    grid_poly = grid_poly.dropna(subset=["geometry"]).reset_index(drop=True)

    # Ensure both GeoDataFrames use the same CRS
    if site_poly.crs != grid_poly.crs:
        site_poly = site_poly.to_crs(grid_poly.crs)

    # Create a buffer around each site polygon
    buffer_poly = site_poly.copy()
    buffer_poly["geometry"] = buffer_poly.geometry.buffer(buffer_distance)

    # Remove buffer overlaps with the original site polygons
    buffer_poly["geometry"] = buffer_poly.geometry.difference(site_poly.union_all())

    # Perform spatial join: Find grid polygons within the modified buffer
    grid_within_buffer = gpd.sjoin(
        grid_poly,
        buffer_poly[["geometry"]],  # Use only geometry from buffer_poly
        how="inner",
        predicate="intersects",
    )

    # Drop any columns that may have been added from the right GeoDataFrame
    columns_to_drop = [col for col in grid_within_buffer.columns if col not in grid_poly.columns]
    grid_within_buffer = grid_within_buffer.drop(columns=columns_to_drop)

    # Reset index
    grid_within_buffer.reset_index(drop=True, inplace=True)

    # Calculate centroids for site and grid polygons
    site_centroids = site_poly.geometry.centroid
    grid_centroids = grid_within_buffer.geometry.centroid

    # Calculate the distance for each site centroid to all grid centroids
    for site_idx, site_cent in tqdm(site_centroids.items(), desc="Calculating distances for each site"):
        # Calculate distances for the current site
        distances = grid_centroids.distance(site_cent)

        # Add the distances as a new column to grid_within_buffer
        grid_within_buffer[f"distance_{site_idx}"] = distances.round(0)

    # Optionally save the result to a file
    if output_path:
        grid_within_buffer.to_file(output_path, driver="GPKG")

    return grid_within_buffer

def calculate_capex_by_grid(grid, dist_cols, capex, connection_fees, landcost_factor, power_density, availability, area = 'area', others=None, digits=None):
    """
    Calculate CAPEX components by grid.

    Parameters:
    - grid (pd.DataFrame): DataFrame containing grid data, including area and distances.
    - dist_cols (list): List of column names representing distances.
    - capex (float): CAPEX cost per MW (KRW).
    - connection_fees (float): Connection fees per MW per km (KRW).
    - landcost_factor (float): Factor applied to land cost.
    - power_density (float): Power density (W/m^2).
    - availability (float): Availability factor for land area.
    - others (list or float): Other costs per MW.
    - digits (int): Number of decimal places to round the results.

    Returns:
    - pd.DataFrame: A DataFrame containing capacity and total capital costs for each distance.
    """

    # Calculate capacity (MW)
    capacity = (grid[area] * availability * power_density / 1e6)  # Convert W to MW

    # Calculate equipment cost (KRW)
    equipment_cost = capacity * capex

    # Extract distances
    distances = grid[dist_cols]

    # Ensure capacity is properly reshaped for broadcasting
    capacity_expanded = capacity.values.reshape(-1, 1)  # Shape: (n_rows, 1)

    # Calculate connection cost (KRW) for each distance
    connection_cost = capacity_expanded * connection_fees * distances.values  # (n_rows, len(dist_cols))

    # Calculate land cost (KRW)
    land_cost = grid['wavgprice'] * landcost_factor * grid[area]

    # Calculate total other costs (KRW)
    if others is not None:
        if isinstance(others, (list, tuple)):
            total_other_cost = capacity * sum(others)
        else:
            total_other_cost = capacity * others
    else:
        total_other_cost = 0  # Default to 0 if "others" is not provided

    # Initialize result DataFrame
    result = pd.DataFrame({
        'capacity': capacity,
        'land_cost': land_cost,
        'equipment_cost': equipment_cost,
        'other_cost': total_other_cost,
    })

    # Add connection cost and total cost for each distance
    for i, col in enumerate(dist_cols):
        result[f'connection_cost_{col}'] = connection_cost[:, i]
        result[f'total_cost_{col}'] = (
            equipment_cost +
            land_cost +
            total_other_cost +
            connection_cost[:, i]
        )

    if digits is not None:
        result = result.round(digits)
    return result

# LCOE 계산하기
import pandas as pd
import plotly.express as px

def annualise_capital_cost(capacity_factors, capex_values, opex_values, lifetime, discount_rate=0.05, rec_cost = 0, min_smp = None, digits=2, plot=False):
    """
    Calculates the annualized Levelized Cost of Electricity (LCOE) including CAPEX, OPEX, and optional REC costs.

    Parameters:
    - capacity_factors (float, list, or np.array):
      Single capacity factor (as a decimal, e.g., 0.25 for 25%) or
      a list/array of yearly capacity factors over the plant's lifetime.
    - capex_values (float or list):
      Single CAPEX value (float) or multiple CAPEX values (list) with associated annualization.
    - opex_values (float or list):
      Single OPEX value (KRW per kW-year) or multiple OPEX values (list).
    - lifetime (int): Total plant lifetime in years.
    - discount_rate (float): Discount rate as a decimal (default is 5%).
    - rec_cost (float): Renewable Energy Certificate cost in KRW per MWh (default is 0).
    - digits (int): Number of decimal places to round the results (default is 2).
    - plot (bool): If True, generates a bar plot of the results.

    Returns:
    - annualized_costs (list): List of annualized LCOE values (KRW per MWh).
    """
    # Handle single value for capacity factor
    if isinstance(capacity_factors, (float, int)):
        capacity_factors = [capacity_factors] * lifetime

    if len(capacity_factors) != lifetime:
            raise ValueError("The length of capacity_factors must match the lifetime of the plant.")

    # Handle single or multiple CAPEX values
    if isinstance(capex_values, (float, int)):
        capex_values = [capex_values]

    # Handle single OPEX value and apply it to all CAPEX values
    if isinstance(opex_values, (float, int)):
        opex_values = [opex_values] * len(capex_values)

    if len(opex_values) != len(capex_values):
        raise ValueError("The number of OPEX values must match the number of CAPEX values or be a single value.")

    # Discount factors for each year
    discount_factors = [(1 / (1 + discount_rate) ** t) for t in range(1, lifetime + 1)]

    # Total discounted generation (in MWh)
    total_discounted_generation = sum(
        cf * 8760 * discount_factors[t] for t, cf in enumerate(capacity_factors)
    )

    # Calculate annualized LCOE (CAPEX + OPEX) for each pair of CAPEX and OPEX values
    annualized_costs = []
    for capex, opex in zip(capex_values, opex_values):
        # Annualized CAPEX
        annualized_capex = capex * (discount_rate * (1 + discount_rate) ** lifetime) / ((1 + discount_rate) ** lifetime - 1)

        # Convert OPEX from kW-year to MW-year
        opex_mw_year = opex * 1000

        # Total discounted OPEX
        total_discounted_opex = sum(opex_mw_year * discount_factors[t] for t in range(lifetime))

        # Total discounted costs (CAPEX + OPEX + REC)
        total_discounted_costs = annualized_capex * lifetime + total_discounted_opex + (rec_cost * total_discounted_generation)

        # LCOE calculation
        lcoe = round(total_discounted_costs / total_discounted_generation, ndigits=digits)

        if min_smp is not None:
            lcoe = min_smp if lcoe < min_smp else lcoe

        annualized_costs.append(lcoe)

    # Optional: Generate a bar plot
    if plot:
        # Create a DataFrame for plotting
        stats = pd.DataFrame({
            'CAPEX Value': capex_values,
            'OPEX Value (kW-year)': opex_values,
            'LCOE (KRW/MWh)': annualized_costs
        })

        # Create a bar chart
        fig = px.bar(
            stats,
            x=stats.index,  # Use the index for fixed step x-axis
            y='LCOE (KRW/MWh)',
            title='Levelized Cost of Electricity (LCOE)',
            labels={
                'index': 'Fixed Steps (CAPEX-OPEX Order)',
                'LCOE (KRW/MWh)': 'LCOE (KRW per MWh)'
            }
        )

        fig.update_layout(
            xaxis_title='Fixed Steps',
            yaxis_title='LCOE (KRW per MWh)',
            template='plotly_white'
        )
        fig.show()

    return annualized_costs

def analyze_cost_data(cost_df, minimum_capacity, total_cost_column, bin = 1, plot=False, annualised_cost = False):
    """
    Processes cost data to calculate cumulative capacity bins and associated cost statistics.

    Parameters:
    - cost_df (pd.DataFrame): DataFrame containing cost and capacity data.
    - minimum_capacity (float): Minimum capacity threshold to filter the data.
    - total_cost_column (str): Column name in cost_df representing the total cost.
    - plot (bool): If True, generates an interactive plot using Plotly. Default is False.

    Returns:
    - stats (pd.DataFrame): DataFrame containing max, mean, median, and std of costs per capacity bin.
    """
    # Ensure the total_cost_column exists in the DataFrame
    if total_cost_column not in cost_df.columns:
        raise ValueError(f"The column '{total_cost_column}' does not exist in the DataFrame.")

    # Filter the DataFrame based on the minimum capacity
    cost_df_filtered = cost_df[cost_df['capacity'] >= minimum_capacity].copy()

    if cost_df_filtered.empty:
        raise ValueError("No data left after filtering by minimum capacity.")

    # Calculate the cost per unit capacity
    if annualised_cost:
        cost_df_filtered['cost'] = cost_df_filtered[total_cost_column]
    else:
        cost_df_filtered['cost'] = cost_df_filtered[total_cost_column] / cost_df_filtered['capacity']

    # Sort the DataFrame by 'cost' in ascending order
    cost_df_sorted = cost_df_filtered.sort_values(by='cost').reset_index(drop=True)

    # Calculate capacity in GW
    cost_df_sorted['capacity_gw'] = cost_df_sorted['capacity'] / 1e3

    # Calculate cumulative capacity in GW
    cost_df_sorted['cumulative_capacity_gw'] = cost_df_sorted['capacity_gw'].cumsum()

    # Define bin edges at every 1 GW of cumulative capacity
    max_cumulative_capacity = cost_df_sorted['cumulative_capacity_gw'].max()
    bin_edges = np.arange(0, np.ceil(max_cumulative_capacity) + 1, bin)  # Bins at every 1 GW

    # Assign bins based on cumulative capacity
    cost_df_sorted['capacity_bin'] = pd.cut(
        cost_df_sorted['cumulative_capacity_gw'],
        bins=bin_edges,
        right=False
    )

    # Group by cumulative capacity bins and calculate statistics
    stats = cost_df_sorted.groupby('capacity_bin').agg({
        'cost': ['max', 'mean', 'median', 'std'],
        'capacity_gw': 'sum'
    }).reset_index()

    # Flatten MultiIndex columns
    stats.columns = ['capacity_bin', 'max_cost', 'mean_cost', 'median_cost', 'std_cost', 'bin_capacity_gw']

    # Optionally plot the results
    if plot:
        # Convert capacity_bin to a string representation for plotting
        stats['bin_label'] = stats['capacity_bin'].astype(str)

        fig = px.bar(
            stats,
            x='bin_label',
            y='max_cost',
            labels={'bin_label': 'Cumulative Capacity Bin (GW)', 'max_cost': 'Maximum Cost (KRW per MW)'},
            title='Maximum Cost per Accumulated Capacity Bin'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white')
        fig.show()

    return stats


def process_and_plot_capacity_data(file_path, x, y, sheet_name='Countries', agg_methods=None, n=1, group_by_columns=None, include=None, exclude=None, plot=False, moving_avg_window=None, percentiles=None, plot_mv=False, axis_limits=None):
    """
    Processes data from the sheet, creates bins for x-axis,
    and aggregates y-axis by the specified methods.
    Optionally plots the results dynamically based on grouping.

    Parameters:
        file_path (str): Path to the Excel file.
        x (str): Column to use for the x-axis (e.g., capacity_gw).
        y (str): Column to aggregate for the y-axis (e.g., capital_cost_usd_kw).
        sheet_name (str): Name of the sheet to extract data from.
        agg_methods (list): List of aggregation methods to apply (e.g., ['min', 'median', 'mean', 'max']).
        n (int): Bin width for capacity.
        group_by_columns (list): List of columns to group by.
        include (dict): Dictionary specifying columns and values to include (e.g., {'fuel': ['PV - utility']}).
        exclude (dict): Dictionary specifying columns and values to exclude. Used only if include is None.
        plot (bool): Whether to plot the results.
        moving_avg_window (int): Window size for moving average. If None, no moving average is calculated.
        percentiles (list): List of percentiles to calculate (e.g., [0.25, 0.75]).
        plot_mv (bool): Whether to plot only the moving average when plot=True.
        axis_limits (dict): Dictionary specifying axis limits for the plot (e.g., {'x': [0, 100], 'y': [0, 2000]}).

    Returns:
        pd.DataFrame: Processed DataFrame with binned x and aggregated y for all aggregation methods.
    """
    import pandas as pd
    from itertools import cycle
    import plotly.graph_objects as go

    # Load data
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Check required columns
    required_columns = group_by_columns + [x, y] if group_by_columns else [x, y]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Apply include or exclude filters
    if include:
        for key, values in include.items():
            if key not in df.columns:
                raise ValueError(f"Column '{key}' specified in include is not in the DataFrame.")
            df = df[df[key].isin(values)]
    if exclude:
        for key, values in exclude.items():
            if key not in df.columns:
                raise ValueError(f"Column '{key}' specified in exclude is not in the DataFrame.")
            df = df[~df[key].isin(values)]

    # Create bins for x-axis
    df['capacity_bin'] = pd.cut(df[x], bins=range(0, int(df[x].max()) + 2, n))

    # Group and aggregate data
    group_by_columns = group_by_columns or []
    agg_methods = agg_methods or ['mean']  # Default aggregation method is 'mean'
    percentiles = percentiles or []

    # Create a dictionary of aggregation methods
    agg_dict = {method: pd.NamedAgg(column=y, aggfunc=method) for method in agg_methods}

    # Add custom percentiles
    for p in percentiles:
        agg_dict[f'percentile_{int(p * 100)}'] = pd.NamedAgg(column=y, aggfunc=lambda x: x.quantile(p))

    grouped = df.groupby(group_by_columns + ['capacity_bin'], dropna=True).agg(**agg_dict).reset_index()

    # Calculate moving average for each method if requested
    if moving_avg_window:
        for method in agg_methods + [f'percentile_{int(p * 100)}' for p in percentiles]:
            grouped[f'{method}_moving_avg'] = grouped.groupby(group_by_columns)[method].transform(
                lambda series: series.rolling(window=moving_avg_window, min_periods=1).mean()
            )

    # Plot if requested
    if plot:
        if not group_by_columns:
            raise ValueError("At least one column must be specified for grouping to create plots.")

        groups = grouped[group_by_columns[0]].unique()  # Assume the first column for grouping plots

        # Generate a consistent color for each aggregation method
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])  # Dynamic color palette
        method_colors = {method: next(colors) for method in agg_methods + [f'percentile_{int(p * 100)}' for p in percentiles]}

        for group in groups:
            subset = grouped[grouped[group_by_columns[0]] == group]

            fig = go.Figure()

            for method in agg_methods + [f'percentile_{int(p * 100)}' for p in percentiles]:
                if not plot_mv:
                    color = method_colors[method]  # Assign color based on method
                    fig.add_trace(go.Scatter(
                        x=subset['capacity_bin'].astype(str),
                        y=subset[method],
                        mode='lines+markers',
                        line=dict(color=color),
                        name=f"{method} ({group})"
                    ))
                if moving_avg_window:
                    color = method_colors[method]
                    fig.add_trace(go.Scatter(
                        x=subset['capacity_bin'].astype(str),
                        y=subset[f'{method}_moving_avg'],
                        mode='lines',
                        line=dict(color=color, dash='dot'),
                        name=f"{method} (Moving Avg - {group})"
                    ))

            # Apply axis limits if specified
            if axis_limits:
                fig.update_xaxes(range=axis_limits.get('x', None))
                fig.update_yaxes(range=axis_limits.get('y', None))

            fig.update_layout(
                title=f"{y} vs {x} ({group_by_columns[0]}: {group})",
                xaxis_title="Capacity Bin (GW)",
                yaxis_title=y.capitalize(),
                legend_title="Methods",
                xaxis=dict(tickangle=45),
                template="plotly_white"
            )

            fig.show()

    return grouped
