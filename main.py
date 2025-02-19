# pip install git+https://github.com/planit-institute/pyPPA.git@pip
#   Local URL: http://localhost:8501
#   Network URL: http://172.30.1.30:8501

import streamlit as st
import pandas as pd
import PLANiT_PPA.ppamodule as _ppa
import PLANiT_PPA.analyse as _analyse
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Disable scientific notation
pd.set_option('display.float_format', '{:,.2f}'.format)


# -------------------------------
# Helper functions to convert CSV values
# -------------------------------
def get_bool(d, key, default):
    val = d.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def get_int(d, key, default):
    try:
        return int(float(d.get(key, default)))
    except Exception:
        return default


def get_float(d, key, default):
    try:
        return float(d.get(key, default))
    except Exception:
        return default


# -------------------------------
# Function to run a single scenario from the CSV defaults
# -------------------------------
def run_scenario(scenario, scenario_defaults, base_output_path):
    # Scenario Info & Output Path
    scenario_name = scenario_defaults.get("Scenario Name", "Default Scenario")
    # Use the provided base output file name and append the scenario name
    base, ext = os.path.splitext(base_output_path)
    scenario_file_path = scenario_defaults.get("Output File Path", base_output_path)

    print(scenario_file_path)

    # Load Configurations
    loads_config = {
        "SK": get_int(scenario_defaults, "Load for SK (MW)", 3000),
        "Samsung": get_int(scenario_defaults, "Load for Samsung (MW)", 3000)
    }

    # Maximum Grid Share
    use_max_grid_share = get_bool(scenario_defaults, "Enable Max Grid Share", False)
    if use_max_grid_share:
        max_grid_share_percent = get_int(scenario_defaults, "Max Grid Share (%)", 50)
    else:
        max_grid_share_percent = False
    max_grid_share = None if not use_max_grid_share else max_grid_share_percent / 100
    sense = scenario_defaults.get("Grid Share Condition", "==")

    # Time Settings
    initial_year = get_int(scenario_defaults, "Initial Year", 2023)
    end_year = get_int(scenario_defaults, "Analysis Target Year", 2050)
    model_year = get_int(scenario_defaults, "Model Year", 2030)

    # PPA Payment Settings
    ppa_payment_type = scenario_defaults.get("PPA Payment Type", "Current")

    # Rate and SMP Settings
    rate_increase = get_float(scenario_defaults, "Rate Increase (% per year)", 0.05)
    sheet_options = ["HV_C_I", "HV_C_II", "HV_C_III"]
    default_sheet = scenario_defaults.get("Selected Sheet", "HV_C_III")
    selected_sheet = default_sheet if default_sheet in sheet_options else "HV_C_III"
    smp_limit = get_bool(scenario_defaults, "SMP Limit", True)
    smp_init = get_int(scenario_defaults, "Initial SMP (KRW/MWh)", 167000)
    smp_increase = get_float(scenario_defaults, "SMP Annual Increase (%)", 0.05)

    # Carbon Price / REC Settings
    use_ngfs_carbon_price = get_bool(scenario_defaults, "Use NGFS Carbon Price Scenario", True)
    if use_ngfs_carbon_price:
        carbonprice_init = "NGFS"
        carbonprice_rate = 0
    else:
        carbonprice_init = get_float(scenario_defaults, "Custom Carbon Price Initial Value (KRW/kgCO2)", 0.0)
        carbonprice_rate = get_float(scenario_defaults, "Custom Carbon Price Annual Increase (%)", 0.0)
    rec_grid_init = get_int(scenario_defaults, "Initial REC Price (KRW/MWh)", 80000)
    rec_reduction = get_bool(scenario_defaults, "REC Reduction", True)
    rec_increase = get_float(scenario_defaults, "REC Price Annual Increase (%)", 0.0)
    rec_include = get_bool(scenario_defaults, "REC Include in PPA Fees", True)

    # Currency and Grid Settings
    currency_exchange = get_int(scenario_defaults, "Currency Exchange Rate (KRW/USD)", 1400)

    # Battery Settings
    battery_include = get_bool(scenario_defaults, "Include Battery", False)
    battery_parameters = {
        "include": battery_include,
        "capital_cost_per_mw": get_int(scenario_defaults, "Battery Capital Cost per MW (KRW)", 1_000_000_000),
        "capital_cost_per_mwh": get_int(scenario_defaults, "Battery Capital Cost per MWh (KRW)", 250_000_000),
        "efficiency_store": get_float(scenario_defaults, "Battery Storage Efficiency (%)", 0.9),
        "efficiency_dispatch": get_float(scenario_defaults, "Battery Dispatch Efficiency (%)", 0.9),
        "max_hours": get_int(scenario_defaults, "Battery Max Hours (hours)", 4),
        "lifespan": get_int(scenario_defaults, "Battery Lifespan (years)", 20)
    }

    # Default Parameters (hardcoded keys 'digits' and 'dist_cols' remain as before)
    default_params = {
        'discount_rate': get_float(scenario_defaults, "Discount Rate (%)", 0.05),
        'irr': get_float(scenario_defaults, "Internal Rate of Return (%)", 0.05),
        'buffer': get_int(scenario_defaults, "Buffer (m)", 25000),
        'bin_size': get_int(scenario_defaults, "Bin Size", 1),
        'min_capacity': get_float(scenario_defaults, "Minimum Capacity (MW)", 0.1),
        'connection_fees': get_int(scenario_defaults, "Connection Fees (KRW)", 0),
        'power_density': get_int(scenario_defaults, "Power Density (W/m2)", 120),
        'digits': None,
        'dist_cols': ['distance_0'],
        'max_distace': get_int(scenario_defaults, "Max Distance (m)", 20000),
        'wind_cap': get_int(scenario_defaults, "Wind Capacity (MW)", 5),
    }

    # Instantiate the PPA model using these defaults
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
        default_params,
        battery_parameters,
        ppa_payment_type
    )

    # Prepare a summary of the parameters (for record keeping)
    parameters_summary = {
        "Parameter Name": [
            "Scenario Name",
            "Load for SK (MW)",
            "Load for Samsung (MW)",
            "Include Battery",
            "Currency Exchange Rate (KRW/USD)",
            "Max Grid Share (%)",
            "Grid Share Condition",
            "Rate Increase (% per year)",
            "Selected Sheet",
            "PPA Payment Type",
            "SMP Limit",
            "Initial SMP (KRW/MWh)",
            "SMP Annual Increase (%)",
            "Carbon Price Scenario",
            "Carbon Price Rate",
            "Initial REC Price (KRW/MWh)",
            "REC Reduction",
            "REC Price Annual Increase (%)",
            "REC Include in PPA Fees",
            "Initial Year",
            "Analysis Target Year",
            "Modelling Year",
            "Battery Capital Cost per MW (KRW)",
            "Battery Capital Cost per MWh (KRW)",
            "Battery Storage Efficiency (%)",
            "Battery Dispatch Efficiency (%)",
            "Battery Max Hours (hours)",
            "Battery Lifespan (years)",
            "Discount rate (%)",
            "Internal Rate of Return (%)",
            "Buffer (m)",
            "Bin Size",
            "Minimum Capacity (MW)",
            "Connection Fees (KRW)",
            "Power Density (W/m2)",
            "Max Distance (m)",
            "Wind Capacity (MW)",
        ],
        "Value": [
            scenario_name,
            loads_config["SK"],
            loads_config["Samsung"],
            battery_include,
            currency_exchange,
            max_grid_share_percent if use_max_grid_share else "Not enabled",
            sense,
            rate_increase,
            selected_sheet,
            ppa_payment_type,
            smp_limit,
            smp_init,
            smp_increase,
            carbonprice_init,
            carbonprice_rate,
            rec_grid_init,
            rec_reduction,
            rec_increase,
            rec_include,
            initial_year,
            end_year,
            model_year,
            battery_parameters["capital_cost_per_mw"],
            battery_parameters["capital_cost_per_mwh"],
            battery_parameters["efficiency_store"] * 100,
            battery_parameters["efficiency_dispatch"] * 100,
            battery_parameters["max_hours"],
            battery_parameters["lifespan"],
            default_params['discount_rate'],
            default_params['irr'],
            default_params['buffer'],
            default_params['bin_size'],
            default_params['min_capacity'],
            default_params['connection_fees'],
            default_params['power_density'],
            default_params['max_distace'],
            default_params['wind_cap'],
        ]
    }

    parameters_df = pd.DataFrame(parameters_summary)


    # Run the model
    output = ppa_model.run_model()

    # Save the output if the model did not return "Infeasible"
    if output != "Infeasible":
        try:
            parameters_df = parameters_df.applymap(
                lambda x: f"'{x}" if isinstance(x, str) and x.startswith(('=', '+', '-', '@')) else x
            )

            with pd.ExcelWriter(scenario_file_path, engine='openpyxl') as writer:
                parameters_df.to_excel(writer, sheet_name="Parameters Summary", index=False)

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

            return f"Output successfully saved to {scenario_file_path}"
        except Exception as e:
            return f"An error occurred while saving the output for scenario {scenario}: {e}"
    else:
        return f"The model is infeasible for scenario {scenario}. Please check inputs."


# -------------------------------
# Main App Function
# -------------------------------
def main():
    st.title("PPA Model Configuration")

    # -------------------------------
    # Load scenario defaults CSV file
    # -------------------------------
    try:
        scenarios_df = pd.read_excel("scenario_defaults.xlsx", index_col=0)
        scenario_options = scenarios_df.columns.tolist()
    except Exception as e:
        scenarios_df = None
        scenario_options = []
        st.error("Could not load scenario defaults Scenario file. Please ensure 'scenario_defaults.xlsx' exists.")

    # -------------------------------
    # Buttons to run either a single scenario or all scenarios
    # -------------------------------
    # run_model = st.button("Run PPA Model")
    # run_all = st.button("Run All Scenarios")

    # Create three columns for layout (for interactive single-scenario input)
    col1, col2, col3 = st.columns(3)

    with col1:
        run_model = st.button("Run PPA Model")

    with col2:
        run_all = st.button("Run All Scenarios")

    with col3:
        # --- Reload CSV button ---
        if st.button("Reload Scenario File", key="reload_csv"):
            try:
                # Re-read the CSV file and then force a rerun
                scenarios_df = pd.read_excel("scenario_defaults.xlsx", index_col=0)
                # st.success("CSV file reloaded successfully!")
            except Exception as e:
                st.error(f"Could not reload Scenario file: {e}")

    with col1:
        st.subheader("Scenario Selection")
        if scenario_options:
            selected_scenario = st.selectbox("Select Scenario", options=scenario_options)
            defaults = scenarios_df[selected_scenario].to_dict() if scenarios_df is not None else {}
        else:
            st.info("No scenario defaults available.")
            defaults = {}

        st.subheader("Scenario Information")
        scenario_name = st.text_input("Scenario Name", value=defaults.get("Scenario Name", "Default Scenario"))
        output_file_path = st.text_input("Specify Output File Path (absolute, e.g., C:/path/to/output.xlsx)",
                                         value=defaults.get("Output File Path", "C:/path/to/output.xlsx"))

        st.subheader("Load Configurations")
        loads_config = {
            "SK": st.number_input("Load for SK (MW)", min_value=0,
                                  value=int(defaults.get("Load for SK (MW)", 3000)), step=100),
            "Samsung": st.number_input("Load for Samsung (MW)", min_value=0,
                                       value=int(defaults.get("Load for Samsung (MW)", 3000)), step=100),
        }

        st.subheader("Maximum Grid Share")
        use_max_grid_share = st.checkbox("Enable Max Grid Share?", value=defaults.get("Enable Max Grid Share", False))
        if use_max_grid_share:
            max_grid_share_percent = st.slider("Max Grid Share (%)", min_value=0, max_value=100,
                                               value=int(defaults.get("Max Grid Share (%)", 50)), step=1)
        else:
            max_grid_share_percent = False
        max_grid_share = None if not use_max_grid_share else max_grid_share_percent / 100
        sense = st.selectbox("Grid Share Condition", options=["==", "<=", ">="],
                             index=["==", "<=", ">="].index(defaults.get("Grid Share Condition", "==")))

        st.subheader("Time")
        initial_year = st.number_input("Initial Year", min_value=2000,
                                       value=int(defaults.get("Initial Year", 2023)), step=1)
        end_year = st.number_input("Analysis Target Year", min_value=2000,
                                   value=int(defaults.get("Analysis Target Year", 2050)), step=1)
        model_year = st.number_input("Model Year", min_value=2000,
                                     value=int(defaults.get("Model Year", 2030)), step=1)

    with col2:
        st.subheader("PPA Payment Settings")
        ppa_payment_options = ["Current", "Levelised"]
        default_payment = defaults.get("PPA Payment Type", "Current")
        try:
            payment_index = ppa_payment_options.index(default_payment)
        except ValueError:
            payment_index = 0
        ppa_payment_type = st.selectbox("PPA Payment Type", options=ppa_payment_options, index=payment_index)

        st.subheader("Grid Rate")
        rate_increase = st.number_input("Rate Increase (% per year)", min_value=0.0,
                                        value=float(defaults.get("Rate Increase (% per year)", 0.05)), step=0.01)
        sheet_options = ["HV_C_I", "HV_C_II", "HV_C_III"]
        default_sheet = defaults.get("Selected Sheet", "HV_C_III")
        try:
            sheet_index = sheet_options.index(default_sheet)
        except ValueError:
            sheet_index = 2
        selected_sheet = st.selectbox("Select a Sheet", options=sheet_options, index=sheet_index)

        st.subheader("Renewable Cost")
        smp_limit = st.checkbox("Renewable LCOE Limit (>= SMP + REC)?", value=defaults.get("SMP Limit", True))
        smp_init = st.number_input("Initial SMP (KRW/MWh)", min_value=0,
                                   value=int(defaults.get("Initial SMP (KRW/MWh)", 167000)), step=1000)
        smp_increase = st.number_input("SMP Annual Increase (%)", min_value=0.0,
                                       value=float(defaults.get("SMP Annual Increase (%)", 0.05)), step=0.01)

        st.subheader("Carbon Price")
        use_ngfs_carbon_price = st.checkbox("Use NGFS Carbon Price Scenario?",
                                            value=defaults.get("Use NGFS Carbon Price Scenario", True))
        if use_ngfs_carbon_price:
            carbonprice_init = "NGFS"
            carbonprice_rate = 0
        else:
            carbonprice_init = st.number_input("Custom Carbon Price Initial Value (KRW/kgCO2)", min_value=0.0,
                                               value=float(
                                                   defaults.get("Custom Carbon Price Initial Value (KRW/kgCO2)", 0.0)),
                                               step=0.01)
            carbonprice_rate = st.number_input("Custom Carbon Price Annual Increase (%)", min_value=0.0,
                                               value=float(
                                                   defaults.get("Custom Carbon Price Annual Increase (%)", 0.0)),
                                               step=0.01)

        st.subheader("REC")
        rec_grid_init = st.number_input("Initial REC Price (KRW/MWh)", min_value=0,
                                        value=int(defaults.get("Initial REC Price (KRW/MWh)", 80000)), step=1000)
        rec_reduction = st.checkbox("REC Reduction to 0?", value=defaults.get("REC Reduction", True))
        rec_increase = st.number_input("REC Price Annual Increase (%)", min_value=0.0,
                                       value=float(defaults.get("REC Price Annual Increase (%)", 0.0)), step=0.01)
        rec_include = st.checkbox("Does REC include in PPA fees (True: REC is not recognized)?",
                                  value=defaults.get("REC Include in PPA Fees", True))

        st.subheader("Currency and Grid Settings")
        currency_exchange = st.number_input("Currency Exchange Rate (KRW/USD)", min_value=900, max_value=2000,
                                            value=int(defaults.get("Currency Exchange Rate (KRW/USD)", 1400)), step=1)

    with col3:
        st.subheader("Battery")
        battery_include = st.checkbox("Include Battery?", value=defaults.get("Include Battery", False))

        st.subheader("Battery Parameters")
        battery_parameters = {
            "include": battery_include,
            "capital_cost_per_mw": st.number_input("Battery Capital Cost per MW (KRW)", min_value=0,
                                                   value=int(defaults.get("Battery Capital Cost per MW (KRW)",
                                                                          1_000_000_000)),
                                                   step=100_000_000),
            "capital_cost_per_mwh": st.number_input("Battery Capital Cost per MWh (KRW)", min_value=0,
                                                    value=int(defaults.get("Battery Capital Cost per MWh (KRW)",
                                                                           250_000_000)),
                                                    step=10_000_000),
            "efficiency_store": st.slider("Battery Storage Efficiency (%)", min_value=0.0, max_value=1.0,
                                          value=float(defaults.get("Battery Storage Efficiency (%)", 0.9)), step=0.01),
            "efficiency_dispatch": st.slider("Battery Dispatch Efficiency (%)", min_value=0.0, max_value=1.0,
                                             value=float(defaults.get("Battery Dispatch Efficiency (%)", 0.9)),
                                             step=0.01),
            "max_hours": st.number_input("Battery Max Hours", min_value=0,
                                         value=int(defaults.get("Battery Max Hours (hours)", 4)), step=1),
            "lifespan": st.number_input("Battery Lifespan", min_value=0,
                                        value=int(defaults.get("Battery Lifespan", 20)), step=1),
        }

        st.subheader("Default Parameters (Do not change unless you know what you are doing)")
        default_params = {
            'discount_rate': st.number_input("Discount Rate (%)", min_value=0.0, max_value=1.0,
                                             value=float(defaults.get("Discount Rate (%)", 0.05)), step=0.001),

            'irr': st.number_input("Internal Rate of Return (%)", min_value=0.0, max_value=1.0,
                                             value=float(defaults.get("Internal Rate of Return (%)", 0.075)), step=0.001),


            'buffer': st.number_input("Buffer (m)", min_value=0,
                                      value=int(defaults.get("Buffer (m)", 25000)), step=1000),
            'bin_size': st.number_input("Bin Size (GW)", min_value=0.0,
                                        value=float(defaults.get("Bin Size", 10)), step=0.01),
            'min_capacity': st.number_input("Minimum Capacity (MW)", min_value=0.0,
                                            value=float(defaults.get("Minimum Capacity (MW)", 0.1)), step=0.01),
            'connection_fees': st.number_input("Connection Fees (KRW)", min_value=0,
                                               value=int(defaults.get("Connection Fees (KRW)", 0)), step=1000),
            'power_density': st.number_input("Power Density (W/m2)", min_value=0,
                                             value=int(defaults.get("Power Density (W/m2)", 120)), step=10),
            'digits': None,
            'dist_cols': ['distance_0'],
            'max_distace': st.number_input("Max Distance (m)", min_value=0,
                                           value=int(defaults.get("Max Distance (m)", 20000)), step=1000),
            'wind_cap': st.number_input("Wind Capacity (MW)", min_value=0,
                                        value=int(defaults.get("Wind Capacity (MW)", 5)), step=1),
        }

    if run_all:
        if scenarios_df is None or len(scenario_options) == 0:
            st.error("No scenarios available to run all.")
        else:
            st.subheader("Running All Scenarios")
            base_output_path = output_file_path  # Base name from UI
            results = {}
            for scenario in scenario_options:
                scenario_defaults = scenarios_df[scenario].to_dict()
                st.write(f"Running scenario: **{scenario}**")
                result_message = run_scenario(scenario, scenario_defaults, base_output_path)
                results[scenario] = result_message
                st.write(result_message)

    input_directory = os.path.dirname(output_file_path)

    if st.button("Analyse the output", key="analyse_button"):
        if not os.path.exists(input_directory):
            st.error(f"Invalid directory: {input_directory}. Please check the output path.")
        else:
            st.info("Running analysis...")
            analysis = _analyse.PPAAnalysis(input_directory, discount_rate=0.05)
            merged_results, merged_total, npv_df = analysis.run()

            # Save all merged_results (each sheet), merged_total, and npv_df to an Excel file
            merged_output_path = os.path.join(input_directory, "merged_results.xlsx")
            with pd.ExcelWriter(merged_output_path, engine="openpyxl") as writer:
                # Save the merged total and NPV analysis
                merged_total.to_excel(writer, sheet_name="Merged Total", index=False)
                npv_df.to_excel(writer, sheet_name="NPV Analysis", index=False)

                # Save each sheet in merged_results
                for sheet_name, df in merged_results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            st.subheader("Merged Total Data")
            st.dataframe(merged_total)
            st.subheader("NPV Analysis")
            st.dataframe(npv_df)
            st.success(f"Analysis completed! Results saved to: {merged_output_path}")
            st.markdown(f"[Download Merged Results](sandbox://{merged_output_path})")

    elif run_model:
        st.write(f"\n\nOptimisation process start: {scenario_name}\n\n")

        # Instantiate the PPA model using these defaults
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
            default_params,
            battery_parameters,
            ppa_payment_type
        )

        parameters_summary = {
            "Parameter Name": [
                "Scenario Name",
                "Load for SK (MW)",
                "Load for Samsung (MW)",
                "Include Battery",
                "Currency Exchange Rate (KRW/USD)",
                "Max Grid Share (%)",
                "Grid Share Condition",
                "Rate Increase (% per year)",
                "Selected Sheet",
                "PPA Payment Type",
                "SMP Limit",
                "Initial SMP (KRW/MWh)",
                "SMP Annual Increase (%)",
                "Carbon Price Scenario",
                "Carbon Price Rate",
                "Initial REC Price (KRW/MWh)",
                "REC Reduction",
                "REC Price Annual Increase (%)",
                "REC Include in PPA Fees",
                "Initial Year",
                "Analysis Target Year",
                "Modelling Year",
                "Battery Capital Cost per MW (KRW)",
                "Battery Capital Cost per MWh (KRW)",
                "Battery Storage Efficiency (%)",
                "Battery Dispatch Efficiency (%)",
                "Battery Max Hours (hours)",
                "Battery Lifespan (years)",
                "Discount rate (%)",
                "Internal Rate of Return (%)",
                "Buffer (m)",
                "Bin Size",
                "Minimum Capacity (MW)",
                "Connection Fees (KRW)",
                "Power Density (W/m2)",
                "Max Distance (m)",
                "Wind Capacity (MW)",
            ],
            "Value": [
                scenario_name,
                loads_config["SK"],
                loads_config["Samsung"],
                battery_include,
                currency_exchange,
                max_grid_share_percent if use_max_grid_share else "Not enabled",
                sense,
                rate_increase,
                selected_sheet,
                ppa_payment_type,
                smp_limit,
                smp_init,
                smp_increase,
                carbonprice_init,
                carbonprice_rate,
                rec_grid_init,
                rec_reduction,
                rec_increase,
                rec_include,
                initial_year,
                end_year,
                model_year,
                battery_parameters["capital_cost_per_mw"],
                battery_parameters["capital_cost_per_mwh"],
                battery_parameters["efficiency_store"] * 100,
                battery_parameters["efficiency_dispatch"] * 100,
                battery_parameters["max_hours"],
                battery_parameters["lifespan"],
                default_params['discount_rate'],
                default_params['irr'],
                default_params['buffer'],
                default_params['bin_size'],
                default_params['min_capacity'],
                default_params['connection_fees'],
                default_params['power_density'],
                default_params['max_distace'],
                default_params['wind_cap']
            ]
        }

        parameters_df = pd.DataFrame(parameters_summary)
        output = ppa_model.run_model()


        if output != "Infeasible":
            st.subheader("Output File Selection")
            # st.write(output)

            try:
                parameters_df = parameters_df.applymap(
                    lambda x: f"'{x}" if isinstance(x, str) and x.startswith(('=', '+', '-', '@')) else x
                )

                with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                    parameters_df.to_excel(writer, sheet_name="Parameters Summary", index=False)

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
                st.error(f"An error occurred while saving the output: {e}")
        else:
            st.warning("The model is infeasible. Please check your inputs and try again.")


if __name__ == "__main__":
    main()
