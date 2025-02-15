import os
import pandas as pd
import streamlit as st


class PPAAnalysis:
    def __init__(self, input_directory, discount_rate=0.05):
        """
        Initialize the analysis with the input directory and discount rate.
        """
        self.input_directory = input_directory
        self.discount_rate = discount_rate

    def run_analysis(self):
        """
        Reads Excel files, extracts scenario details, and merges data by Scenario Group and Scenario.
        Returns a dictionary of merged DataFrames per sheet.
        """
        files = [f for f in os.listdir(self.input_directory) if f.endswith(".xlsx")]
        sheets_dict = {}

        for file in files:
            filepath = os.path.join(self.input_directory, file)
            # Extract scenario group and scenario name
            if "_" in file:
                scenario_group, scenario = file.split("_", 1)
            else:
                scenario_group, scenario = file.split(".")[0], "REF"
            scenario = scenario.replace(".xlsx", "")
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
                        numeric_cols.remove("year")
                    if numeric_cols:
                        df["Value"] = df[numeric_cols].sum(axis=1)
                    else:
                        continue
                # Store data for merging later
                sheets_dict.setdefault(sheet_name, []).append(df)

        merged_sheets = {sheet: pd.concat(data, ignore_index=True) for sheet, data in sheets_dict.items()}
        return merged_sheets

    @staticmethod
    def calculate_npv(group, col, discount_rate):
        """
        Calculate the NPV for a given column in the provided group.
        """
        initial_year = group["year"].min()
        group = group.copy()
        group["discount_factor"] = (1 + discount_rate) ** (group["year"] - initial_year)
        npv_value = (group[col] / group["discount_factor"]).sum()
        return npv_value

    def run(self):
        """
        Runs the analysis:
          1. Merges Excel files using run_analysis().
          2. Processes key sheets (marginal cost, rec payment, ESS annualized cost, generation).
          3. Merges these side by side into merged_total.
          4. Calculates additional columns (Total Cost and Cost per MWh).
          5. Groups data by Scenario Group and Scenario to compute NPVs.

        Returns:
          merged_total: DataFrame with merged totals.
          npv_df: DataFrame with NPVs by scenario.
        """
        merged_results = self.run_analysis()

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
        merged_total.fillna(0, inplace=True)

        # Calculate additional columns
        merged_total['Total Cost (KRW)'] = (merged_total['Cost (KRW)'] +
                                            merged_total['REC (KRW)'] +
                                            merged_total['ESS (KRW)'])
        merged_total['Cost (KRW/MWh)'] = merged_total['Cost (KRW)']/merged_total['Generation (GWh)']/1e3
        merged_total['REC (KRW/MWh)'] = merged_total['REC (KRW)']/merged_total['Generation (GWh)']/1e3
        merged_total['ESS (KRW/MWh'] = merged_total['ESS (KRW)']/merged_total['Generation (GWh)']/1e3
        merged_total['Total Cost (KRW/MWh)'] = merged_total['Total Cost (KRW)'] / merged_total['Generation (GWh)'] / 1e3

        # List the columns for which to calculate NPV
        npv_columns = ["Cost (KRW)", "REC (KRW)", "ESS (KRW)", "Generation (GWh)", "Total Cost (KRW)"]

        npv_df_list = []
        grouped = merged_total.groupby(["Scenario Group", "Scenario"])
        for (scenario_group, scenario), group in grouped:
            npv_values = {"Scenario Group": scenario_group, "Scenario": scenario}
            for col in npv_columns:
                npv_values[col] = self.calculate_npv(group, col, self.discount_rate)
            # Derived Cost (KRW/MWh) NPV
            cost_npv = npv_values["Cost (KRW)"]
            rec_npv = npv_values["REC (KRW)"]
            ess_npv = npv_values["ESS (KRW)"]
            gen_npv = npv_values.get("Generation (GWh)", None)
            tot_npv = npv_values.get("Total Cost (KRW)", None)

            npv_values['Cost (KRW/MWh)'] = cost_npv / (gen_npv * 1e3)
            npv_values['REC (KRW/MWh)'] = rec_npv / (gen_npv * 1e3)
            npv_values['ESS (KRW/MWh)'] = ess_npv / (gen_npv * 1e3)
            npv_values["Total Cost (KRW/MWh)"] = tot_npv / (gen_npv * 1e3)

            npv_df_list.append(npv_values)
        npv_df = pd.DataFrame(npv_df_list)

        return merged_results, merged_total, npv_df