import pandas as pd

def process_kepco_data(filepath, year, selected_sheet):
    """
    Process KEPCO Excel data to generate a DataFrame with datetime as index
    and temporal values (based on season and timezone).

    Parameters:
    - filepath (str): Path to the Excel file.
    - year (int): The year to generate the datetime index.
    - selected_sheet (str): The sheet name for the selected rates (e.g., "HV_C_I").

    Returns:
    - pd.DataFrame: Temporal DataFrame with datetime index and corresponding rates.
    - float: Contract fee (krw/kw).
    """
    # Load necessary sheets
    timezone_df = pd.read_excel(filepath, sheet_name="timezone")
    season_df = pd.read_excel(filepath, sheet_name="season")
    contract_df = pd.read_excel(filepath, sheet_name="contract", index_col=0)
    rates_df = pd.read_excel(filepath, sheet_name=selected_sheet, index_col=0)

    # Validate user input
    if selected_sheet not in contract_df.index:
        raise ValueError(f"Selected sheet {selected_sheet} is not valid.")

    # Generate datetime index for the entire year
    date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h")

    # Map seasons based on months
    month_to_season = season_df.set_index("month")["season"].to_dict()
    month_to_int = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5,
        "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    datetime_season = pd.Series(index=date_range, dtype="object")
    for month, season in month_to_season.items():
        month_index = month_to_int[month]
        datetime_season[date_range.month == month_index] = season

    # Map timezones based on hours and months (keep timezone values in uppercase)
    timezone_mapping = timezone_df.set_index("hours")
    datetime_timezone = pd.Series(index=date_range, dtype="object")
    for month in timezone_mapping.columns:
        month_index = month_to_int[month]
        for hour in range(24):
            datetime_timezone[(date_range.month == month_index) & (date_range.hour == hour)] = timezone_mapping.loc[hour, month]

    # Create a DataFrame with datetime index and map rates based on season and timezone
    temporal_df = pd.DataFrame(index=date_range)
    temporal_df["season"] = datetime_season
    temporal_df["timezone"] = datetime_timezone

    # Map rates from the selected sheet
    temporal_df["rate"] = temporal_df.apply(
        lambda row: rates_df.loc[row["timezone"], row["season"]],
        axis=1
    )

    # Extract contract fee
    contract_fee = contract_df.loc[selected_sheet, "fees"]

    return temporal_df, contract_fee

def multiyear_pricing(temporal_df, contract_fee, start_year, num_years, rate_increase, annualised_contract=False):
    """
    Generate a long DataFrame with hourly data for multiple years and cumulative rate increase,
    optionally annualizing the contract fee.

    Parameters:
    - temporal_df (pd.DataFrame): The base DataFrame with hourly rates for one year.
    - contract_fee (float): The contract fee (krw/kw).
    - start_year (int): Starting year for the data.
    - num_years (int): Number of years to include in the data.
    - rate_increase (float): Annual rate increase (e.g., 0.05 for 5%).
    - annualised_contract (bool): If True, annualize contract fees across hourly data.

    Returns:
    - pd.DataFrame: Long DataFrame with datetime index, escalated rates, and contract fees (if annualized).
    - float: Updated contract fee for the final year.
    """
    all_years_df = []
    preset_df = temporal_df.copy()
    preset_df.index = preset_df.index.strftime("%m-%d %H:%M")

    # Add February 29 to the preset_df if it doesn't already exist
    if not any(preset_df.index.str.startswith("02-29")):
        feb_28_data = preset_df.loc["02-28 00:00":"02-28 23:00"].copy()
        feb_29_data = feb_28_data.copy()
        feb_29_data.index = feb_29_data.index.str.replace("02-28", "02-29")
        preset_df = pd.concat([preset_df, feb_29_data])

    # Initialize a list to store contract fees for each year
    contract_fees = []

    # Process each year
    for year in range(start_year, start_year + num_years):
        # Generate the correct date range for the current year (accounts for leap years)
        date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="h")

        # Create an empty DataFrame for the current year
        current_year_df = pd.DataFrame(index=date_range, columns=temporal_df.columns)

        # Align indices and fill current_year_df with matching values from preset_df
        current_year_df.index = current_year_df.index.strftime("%m-%d %H:%M")
        matching_indices = current_year_df.index.intersection(preset_df.index)
        current_year_df.loc[matching_indices, :] = preset_df.loc[matching_indices, :].values

        # Adjust rates with annual rate increase
        current_year_df['rate'] = current_year_df['rate'] * (1 + rate_increase) ** (year - start_year)

        # Restore the full datetime index for the current year
        current_year_df.index = date_range

        # Calculate the contract fee for the current year
        current_year_contract_fee = contract_fee * (1 + rate_increase) ** (year - start_year)
        contract_fees.append({"year": year, "rate": current_year_contract_fee})

        # If annualized contract fees are needed, divide by the number of hours in the year
        if annualised_contract:
            hours_in_year = len(date_range)
            current_year_df['contract_fee'] = current_year_contract_fee / hours_in_year

        # Append the current year DataFrame to the list
        all_years_df.append(current_year_df)

    # Combine all years into a single DataFrame
    long_df = pd.concat(all_years_df)

    # Return the long DataFrame and contract fees as a DataFrame
    return long_df, pd.DataFrame(contract_fees)

def create_rec_grid(start_year, end_year, initial_rec, rate_increase):
    """
    Generate a DataFrame for REC values with annual increments.

    Parameters:
    - start_year (int): The starting year for the REC values.
    - end_year (int): The ending year for the REC values.
    - initial_rec (float): The initial REC value for the first year.
    - rate_increase (float): Annual rate increase (e.g., 0.05 for 5%).

    Returns:
    - pd.DataFrame: A DataFrame with REC values for each year.
    """
    rec_values = {
        year: initial_rec * (1 + rate_increase) ** (year - start_year)
        for year in range(start_year, end_year + 1)
    }
    return pd.DataFrame({"value": rec_values})
