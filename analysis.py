import os
import pandas as pd

def run_analysis(input_directory):
    """
    Reads Excel files, extracts scenario details, and merges data by Scenario Group and Scenario.

    :param input_directory: Path to the directory containing Excel files.
    :return: Dictionary of merged DataFrames per sheet.
    """
    # Get all Excel files in the input directory
    files = [f for f in os.listdir(input_directory) if f.endswith(".xlsx")]

    # Dictionary to store data by sheet names
    sheets_dict = {}

    for file in files:
        filepath = os.path.join(input_directory, file)

        # Extract scenario group and scenario name
        if "_" in file:
            scenario_group, scenario = file.split("_", 1)
        else:
            scenario_group, scenario = file.split(".")[0], "REF"
        scenario = scenario.replace(".xlsx", "")

        # Read all sheets from the Excel file
        xls = pd.ExcelFile(filepath)

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Ensure "Scenario Group" and "Scenario" columns exist before inserting
            if "Scenario Group" not in df.columns:
                df.insert(0, "Scenario Group", scenario_group)

            if "Scenario" not in df.columns:
                df.insert(1, "Scenario", scenario)

            # Rename unnamed numeric columns to 'year'
            df.columns = [col if not str(col).startswith("Unnamed") else "year" for col in df.columns]

            # Ensure required column "Value" exists
            if "Value" not in df.columns:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if "year" in numeric_cols:
                    numeric_cols.remove("year")  # Exclude "year"

                if numeric_cols:
                    df["Value"] = df[numeric_cols].sum(axis=1)
                else:
                    continue  # Skip if no numeric data

            # Store data for merging later
            if sheet_name in sheets_dict:
                sheets_dict[sheet_name].append(df)
            else:
                sheets_dict[sheet_name] = [df]

    # Merge all data for each sheet
    merged_sheets = {sheet: pd.concat(data, ignore_index=True) for sheet, data in sheets_dict.items()}

    return merged_sheets  # Dictionary of merged DataFrames per sheet

def calculate_npv(group, col, discount_rate):
    # Determine the initial year for discounting in this group
    initial_year = group["year"].min()
    # Compute discount factor for each row
    group = group.copy()
    group["discount_factor"] = (1 + discount_rate) ** (group["year"] - initial_year)
    # Compute NPV for the column
    npv_value = (group[col] / group["discount_factor"]).sum()
    return npv_value

merged_results = run_analysis(input_directory)

# Create copies of the relevant dataframes
cost_df = merged_results["marginal cost (KRW)"].copy()
rec_df = merged_results["rec payment (KRW)"].copy()
ess_df = merged_results["ESS annualized cost (KRW per y)"].copy()
gen_df = merged_results["generation (GWh)"].copy()

# Rename the last column of each dataframe to the desired name
cost_df.rename(columns={cost_df.columns[-1]: "Cost (KRW)"}, inplace=True)
rec_df.rename(columns={rec_df.columns[-1]: "REC (KRW)"}, inplace=True)
ess_df.rename(columns={ess_df.columns[-1]: "ESS (KRW)"}, inplace=True)
gen_df.rename(columns={gen_df.columns[-1]: "Generation (GWh)"}, inplace=True)

# Keep only the key columns and the renamed value column
cost_df = cost_df[["Scenario Group", "Scenario", "year", "Cost (KRW)"]]
rec_df = rec_df[["Scenario Group", "Scenario", "year", "REC (KRW)"]]
ess_df = ess_df[["Scenario Group", "Scenario", "year", "ESS (KRW)"]]
gen_df = gen_df[["Scenario Group", "Scenario", "year", "Generation (GWh)"]]

# Merge the four dataframes side by side using an outer join
merged_total = cost_df.merge(rec_df, on=["Scenario Group", "Scenario", "year"], how="outer") \
                     .merge(ess_df, on=["Scenario Group", "Scenario", "year"], how="outer") \
                     .merge(gen_df, on=["Scenario Group", "Scenario", "year"], how="outer")

# Replace NaN values with 0
merged_total.fillna(0, inplace=True)

merged_total['Total Cost (KRW)'] = merged_total['Cost (KRW)'] + merged_total['REC (KRW)'] + merged_total['ESS (KRW)']
merged_total['Cost (KRW/MWh)'] = merged_total['Total Cost (KRW)'] / merged_total['Generation (GWh)'] / 1e3

# List the columns for which you want to calculate NPV
npv_columns = ["Cost (KRW)", "REC (KRW)", "ESS (KRW)", "Generation (GWh)", "Total Cost (KRW)"]

# Group merged_total by "Scenario Group" and "Scenario" and calculate NPVs
npv_df_list = []
grouped = merged_total.groupby(["Scenario Group", "Scenario"])

for (scenario_group, scenario), group in grouped:
    npv_values = {"Scenario Group": scenario_group, "Scenario": scenario}
    for col in npv_columns:
        npv_values[col] = calculate_npv(group, col, 0.05)
    npv_df_list.append(npv_values)

npv_df = pd.DataFrame(npv_df_list)
npv_df['Cost (KRW/MWh)'] = npv_df['Total Cost (KRW)'] / npv_df['Generation (GWh)']/1e3