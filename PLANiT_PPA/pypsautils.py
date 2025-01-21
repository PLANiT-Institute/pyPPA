import pandas as pd
import pypsa



def get_time_series_data(excel_data, sheet_name, snapshots, column_name=None):
    """
    Retrieve and align time series data with network snapshots.
    """
    sheet = excel_data.get(sheet_name)
    if sheet is None:
        raise ValueError(f"Time series sheet '{sheet_name}' not found.")

    # Strip whitespace from column names to avoid issues
    sheet.columns = sheet.columns.str.strip()

    # Check if 'datetime' is already the index
    if sheet.index.name != 'datetime':
        if 'datetime' not in sheet.columns:
            raise ValueError(f"'datetime' column not found in sheet '{sheet_name}'.")

        # Ensure 'datetime' is in datetime format and set as index
        sheet['datetime'] = pd.to_datetime(sheet['datetime'])
        sheet.set_index('datetime', inplace=True)

    # Reindex to match snapshots, filling missing values with zeros
    time_series = sheet.reindex(snapshots).fillna(0)

    if column_name:
        # Check if the specified column exists in the sheet
        if column_name in time_series.columns:
            print(f"Retrieving time series for column '{column_name}' in sheet '{sheet_name}'")
            return time_series[column_name]
        else:
            raise ValueError(f"Column '{column_name}' not found in sheet '{sheet_name}'.")
    else:
        # If no column name is provided, return the first column if only one exists
        if len(time_series.columns) == 1:
            return time_series.iloc[:, 0]
        else:
            raise ValueError(f"Multiple columns found in sheet '{sheet_name}', please specify 'column_name'.")

def resolve_time_series_reference(value, excel_data, snapshots, component_name=None):
    """
    Resolve 'ts_' prefixed values to time series data.
    """
    if isinstance(value, str) and value.startswith('ts_'):
        parts = value.split('.')
        sheet_name = parts[0]  # e.g., 'ts_load_data'
        column_name = parts[1] if len(parts) > 1 else component_name
        return get_time_series_data(excel_data, sheet_name, snapshots, column_name)
    return value

def add_components_from_excel(network, file_path):
    """
    Add components to a PyPSA network from an Excel file.
    Each sheet represents a component type, and time series sheets start with 'ts_'.
    """
    excel_data = pd.read_excel(file_path, sheet_name=None)

    if "Bus" not in excel_data:
        raise ValueError("The 'Bus' sheet is missing in the Excel file.")

    # Process 'Bus' first, then other component sheets, excluding time series sheets
    component_sheets = ["Bus"] + [s for s in excel_data if s != "Bus" and not s.startswith('ts_')]

    for sheet_name in component_sheets:
        df = excel_data[sheet_name]
        component_type = sheet_name.capitalize()

        print(f"Adding components of type '{component_type}' from sheet '{sheet_name}'.")

        for idx, row in df.iterrows():
            row_dict = row.dropna().to_dict()
            name = row_dict.pop('name', f"{sheet_name}_{idx}")

            static_attrs = {}
            time_series_attrs = {}

            for attr, value in row_dict.items():
                # Resolve time series references
                resolved_value = resolve_time_series_reference(value, excel_data, network.snapshots, name)

                if isinstance(resolved_value, pd.Series):
                    time_series_attrs[attr] = resolved_value
                else:
                    static_attrs[attr] = resolved_value

            # Add the component with static attributes
            network.add(component_type, name, **static_attrs)

            # Assign time-varying attributes
            for attr, series in time_series_attrs.items():
                components_t_df = getattr(network, f"{component_type.lower()}s_t", None)

                if components_t_df is not None:
                    if attr not in components_t_df:
                        components_t_df[attr] = pd.DataFrame(index=network.snapshots)

                    components_t_df[attr][name] = series
                else:
                    raise ValueError(
                        f"Component '{component_type}' does not have time-varying attribute '{attr}'."
                    )

def equivalent_annual_cost(capex, rate, periods):
    """
    Calculate the equivalent annual cost of a capital expenditure (CapEx).

    Parameters:
    - capex: The total capital cost (present value).
    - rate: The discount rate per period (as a decimal, e.g., 0.05 for 5%).
    - periods: The total number of periods.

    Returns:
    - The equivalent annual cost.
    """
    if rate == 0:
        return capex / periods
    else:
        annuity_factor = (1 - (1 + rate) ** -periods) / rate
        annual_cost = capex / annuity_factor
        return annual_cost