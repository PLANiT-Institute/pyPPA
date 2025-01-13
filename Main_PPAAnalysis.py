import streamlit as st
import pandas as pd
import PLANiT_PPA_utils.ppamodule as _ppa

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Disable scientific notation
pd.set_option('display.float_format', '{:,.2f}'.format)

def main():
    st.title("PPA Model Configuration")

    # Place the button at the top
    if st.button("Run PPA Model"):
        run_model = True
    else:
        run_model = False

    col1, col2, col3 = st.columns(3)

    with col1:
        # Section: Scenario Information
        st.subheader("Scenario Information")
        scenario_name = st.text_input("Scenario Name", value="Default Scenario")
        output_file_path = st.text_input("Specify Output File Path (absolute, e.g., C:/path/to/output.xlsx)", value="C:/path/to/output.xlsx")

        # Section: Load Configurations
        st.subheader("Load Configurations")
        loads_config = {
            "SK": st.number_input("Load for SK (MW)", min_value=0, value=3000, step=100),
            "Samsung": st.number_input("Load for Samsung (MW)", min_value=0, value=3000, step=100),
        }

        # Slider for max grid share in percentage
        # Checkbox for enabling/disabling the slider
        st.subheader("Maximum Grid Share")
        use_max_grid_share = st.checkbox("Enable Max Grid Share?")

        # Slider for max grid share in percentage (only shown if checkbox is checked)
        if use_max_grid_share:
            max_grid_share_percent = st.slider(
                "Max Grid Share (%)",
                min_value=0, max_value=100, value=50, step=1
            )
        else:
            max_grid_share_percent = False  # Return False if checkbox is not checked

        # Convert the percentage to a decimal for internal calculations
        max_grid_share = None if use_max_grid_share is False else max_grid_share_percent / 100

        sense = st.selectbox("Grid Share Condition", options=["==", "<=", ">="])

        # Section: Time Settings
        st.subheader("Time")
        initial_year = st.number_input("Initial Year", min_value=2000, value=2023, step=1)
        end_year = st.number_input("Analysis Target Year", min_value=2000, value=2050, step=1)
        model_year = st.number_input("Model Year", min_value=2000, value=2030, step=1)

    with col2:

        # Section: Rate and SMP Settings
        st.subheader("Grid Rate")
        rate_increase = st.number_input("Rate Increase (% per year)", min_value=0.0, value=0.05, step=0.01)
        selected_sheet = st.selectbox(
            "Select a Sheet",
            options=["HV_C_I", "HV_C_II", "HV_C_III"],  # List of dropdown options
            index=2  # Default selection index (e.g., HV_C_III)
        )

        st.subheader("Renewable Cost")
        smp_limit = st.checkbox("Renewable LCOE Limit (>= SMP + REC)?", value=True)
        smp_init = st.number_input("Initial SMP (KRW/MWh)", min_value=0, value=167000, step=1000)
        smp_increase = st.number_input("SMP Annual Increase (%)", min_value=0.0, value=0.05, step=0.01)

        # Section: REC Settings
        st.subheader("Carbon Price")
        # Checkbox for Carbon Price Scenario
        use_ngfs_carbon_price = st.checkbox("Use NGFS Carbon Price Scenario?", value=True)

        # Input for custom carbon price if NGFS is not selected
        if use_ngfs_carbon_price:
            carbonprice_init = "NGFS"
            carbonprice_rate = 0
        else:
            carbonprice_init = st.number_input("Custom Carbon Price Initial Value (KRW/kgCO2)", min_value=0.0, value=0.0,
                                               step=0.01)
            carbonprice_rate = st.number_input("Custom Carbon Price Annual Increase (%)", min_value=0.0, value=0.0, step=0.01)

        st.subheader("REC")
        rec_grid_init = st.number_input("Initial REC Price (KRW/MWh)", min_value=0, value=80000, step=1000)
        rec_reduction = st.checkbox("REC Reduction to 0?", value=True)
        rec_increase = st.number_input("REC Price Annual Increase (%)", min_value=0.0, value=0.0, step=0.01)
        rec_include = st.checkbox("Does REC include in PPA fees (True: REC is not recognized)?", value=True)

        # Section: Currency and Grid Settings
        st.subheader("Currency and Grid Settings")
        currency_exchange = st.number_input("Currency Exchange Rate (KRW/USD)", min_value=900, max_value=2000,
                                            value=1400, step=1)

    with col3:

        # Section: Battery Settings
        st.subheader("Battery")
        battery_include = st.checkbox("Include Battery?", value=False)

        # Section: Battery Parameters
        st.subheader("Battery Parameters")
        battery_parameters = {
            "include": battery_include,
            "capital_cost_per_mw": st.number_input("Battery Capital Cost per MW (KRW)", min_value=0, value=1_000_000_000, step=100_000_000),
            "capital_cost_per_mwh": st.number_input("Battery Capital Cost per MWh (KRW)", min_value=0, value=250_000_000, step=10_000_000),
            "efficiency_store": st.slider("Battery Storage Efficiency (%)", min_value=0.0, max_value=1.0, value=0.9, step=0.01),
            "efficiency_dispatch": st.slider("Battery Dispatch Efficiency (%)", min_value=0.0, max_value=1.0, value=0.9, step=0.01),
            "max_hours": st.number_input("Battery Max Hours", min_value=0, value=4, step=1),
        }

        # Section: Default Parameters
        st.subheader("Default Parameters (Do not change unless you know what you are doing)")
        default_params = {
            'buffer': st.number_input("Buffer (m)", min_value=0, value=25000, step=1000),
            'bin_size': st.number_input("Bin Size", min_value=0, value=1, step=1),
            'min_capacity': st.number_input("Minimum Capacity (MW)", min_value=0.0, value=0.1, step=0.01),
            'connection_fees': st.number_input("Connection Fees (KRW)", min_value=0, value=0, step=1000),
            'power_density': st.number_input("Power Density (W/m²)", min_value=0, value=120, step=10),
            'digits': None,
            'dist_cols': ['distance_0'],
            'max_distace': st.number_input("Max Distance (m)", min_value=0, value=20000, step=1000),
            'wind_cap': st.number_input("Wind Capacity (MW)", min_value=0, value=5, step=1),
        }

    if run_model:
        # Instantiate the class
        ppa_model = _ppa.PPAModel(
            loads_config,
            battery_include,
            currency_exchange,
            max_grid_share,
            sense,
            rate_increase,
            selected_sheet,
            carbonprice_init,
            carbonprice_rate,
            rec_grid_init,
            rec_reduction,
            rec_include,
            rec_increase,
            smp_limit,
            smp_init,
            smp_increase,
            initial_year,
            end_year,
            model_year,
            # Parameters
            default_params,
            # Battery Parameters
            battery_parameters
        )

        # Collect parameters into a dictionary for the summary
        parameters_summary = {
            "Parameter Name": [
                "Scenario Name",
                "Load for SK (MW)",  # Unit: MW
                "Load for Samsung (MW)",  # Unit: MW
                "Include Battery",
                "Currency Exchange Rate (KRW/USD)",  # Unit: KRW/USD
                "Max Grid Share (%)",  # Unit: %
                "Grid Share Condition",
                "Rate Increase (% per year)",  # Unit: %
                "Selected Sheet",
                "SMP Limit",
                "Initial SMP (KRW/MWh)",  # Unit: KRW/MWh
                "SMP Annual Increase (%)",  # Unit: %
                "Carbon Price Scenario",
                "Carbon Price Rate",
                "Initial REC Price (KRW/MWh)",  # Unit: KRW/MWh
                "REC Reduction",
                "REC Price Annual Increase (%)",  # Unit: %
                "REC Include in PPA Fees",
                "Initial Year",
                "Analysis Target Year",
                "Modelling Year",
                "Battery Capital Cost per MW (KRW)",  # Unit: KRW
                "Battery Capital Cost per MWh (KRW)",  # Unit: KRW
                "Battery Storage Efficiency (%)",  # Unit: %
                "Battery Dispatch Efficiency (%)",  # Unit: %
                "Battery Max Hours (hours)",  # Unit: hours
                "Buffer (m)",  # Unit: meters
                "Bin Size",
                "Minimum Capacity (MW)",  # Unit: MW
                "Connection Fees (KRW)",  # Unit: KRW
                "Power Density (W/m²)",  # Unit: W/m²
                "Max Distance (m)",  # Unit: meters
                "Wind Capacity (MW)",  # Unit: MW
            ],
            "Value": [
                scenario_name, loads_config["SK"], loads_config["Samsung"],
                battery_include, currency_exchange, max_grid_share_percent,
                sense, rate_increase, selected_sheet,
                smp_limit, smp_init, smp_increase, carbonprice_init, carbonprice_rate,
                rec_grid_init, rec_reduction, rec_increase,
                rec_include, initial_year, end_year,
                model_year, battery_parameters["capital_cost_per_mw"],
                battery_parameters["capital_cost_per_mwh"], battery_parameters["efficiency_store"] * 100,
                                                            battery_parameters["efficiency_dispatch"] * 100,
                battery_parameters["max_hours"], default_params['buffer'],
                default_params['bin_size'], default_params['min_capacity'], default_params['connection_fees'],
                default_params['power_density'], default_params['max_distace'], default_params['wind_cap']
            ]
        }

        parameters_df = pd.DataFrame(parameters_summary)
        output = ppa_model.run_model()
        st.subheader("Output File Selection")
        st.write(output)

        # File input selection box for the output file

        try:
            # Ensure parameters_df is string-only and safe for Excel
            # parameters_df = parameters_df.astype(str)
            parameters_df = parameters_df.applymap(
                lambda x: f"'{x}" if isinstance(x, str) and x.startswith(('=', '+', '-', '@')) else x
            )

            # Save the output to an Excel file
            with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                # Write Parameters Summary
                parameters_df.to_excel(writer, sheet_name="Parameters Summary", index=False)

                # Save each model output as a separate sheet
                for sheet_name, data in output.items():
                    if isinstance(data, str):
                        from io import StringIO
                        df = pd.read_csv(StringIO(data), delim_whitespace=True)
                    elif isinstance(data, pd.Series):
                        df = data.to_frame(name="Value")
                    elif isinstance(data, pd.DataFrame):
                        df = data
                    else:
                        st.warning(f"Unrecognized data format for sheet '{sheet_name}'. Skipping.")
                        continue

                    valid_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                    df.to_excel(writer, sheet_name=valid_sheet_name, index=True)

            st.success(f"Output successfully saved to {output_file_path}")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
