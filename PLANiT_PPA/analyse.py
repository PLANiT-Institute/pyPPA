import pandas as pd

class PPAAnalysis:
    def __init__(self, input_directory, discount_rate=0.05):
        """
        Initialize the analysis with the input directory and discount rate.
        """
        self.input_directory = input_directory
        self.discount_rate = discount_rate

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
          1. Uses _analyse.run_analysis to merge Excel files.
          2. Creates a merged_total DataFrame from key sheets.
          3. Calculates additional columns (Total Cost and Cost per MWh).
          4. Groups data and computes NPVs.

        Returns:
          merged_total: DataFrame with merged totals.
          npv_df: DataFrame with NPVs by scenario.
        """
        # Run analysis using the _analyse module
        merged_results = _analyse.run_analysis(self.input_directory)

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

        # Calculate additional columns
        merged_total['Total Cost (KRW)'] = (merged_total['Cost (KRW)'] +
                                            merged_total['REC (KRW)'] +
                                            merged_total['ESS (KRW)'])
        merged_total['Cost (KRW/MWh)'] = merged_total['Total Cost (KRW)'] / merged_total['Generation (GWh)'] / 1e3

        # List the columns for which to calculate NPV
        npv_columns = ["Cost (KRW)", "REC (KRW)", "ESS (KRW)", "Generation (GWh)", "Total Cost (KRW)"]

        # Group merged_total by "Scenario Group" and "Scenario" and calculate NPVs
        npv_df_list = []
        grouped = merged_total.groupby(["Scenario Group", "Scenario"])
        for (scenario_group, scenario), group in grouped:
            npv_values = {"Scenario Group": scenario_group, "Scenario": scenario}
            for col in npv_columns:
                npv_values[col + " NPV"] = self.calculate_npv(group, col, self.discount_rate)
            # Also calculate the derived cost per MWh NPV
            gen_npv = npv_values.get("Generation (GWh) NPV", None)
            tot_npv = npv_values.get("Total Cost (KRW) NPV", None)
            if gen_npv is not None and gen_npv != 0:
                npv_values["Cost (KRW/MWh) NPV"] = tot_npv / (gen_npv * 1e3)
            else:
                npv_values["Cost (KRW/MWh) NPV"] = 0
            npv_df_list.append(npv_values)
        npv_df = pd.DataFrame(npv_df_list)

        return merged_total, npv_df