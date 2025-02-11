# PPA Modeling and Optimization

## Overview
This repository contains an advanced Power Purchase Agreement (PPA) modeling and optimization framework utilizing **PyPSA** for energy system analysis. The model incorporates **renewable energy generation**, **grid electricity**, and **battery storage** to determine optimal energy procurement strategies.

## Features
- **Multi-Year PPA Analysis**: Evaluate long-term renewable energy procurement strategies.
- **Renewable Energy Cost Estimation**: Calculate LCOE for PV, agrivoltaics, and offshore wind.
- **KEPCO Grid Pricing Integration**: Incorporates official KEPCO tariffs and projections.
- **Battery Storage Modeling**: Assess the financial and operational impact of battery integration.
- **Carbon and REC Pricing**: Includes customizable carbon pricing scenarios (e.g., NGFS) and REC costs.
- **Scenario-based Analysis**: Supports multiple scenario runs with configurable parameters.
- **Streamlit Interface**: Web-based UI for configuring and running PPA simulations.

## Installation
To use this repository, clone it and install the required dependencies:

```sh
# Install the custom PLANiT_PPA package
pip install git+https://github.com/planit-institute/pyPPA.git@pip

# Download the necessary configuration files
download yaml and main.py from the repository and place them in the appropriate directory.
```

## Usage
### Running the Model
You can run the PPA model using the Streamlit interface:

```sh
streamlit run main.py
```

Alternatively, execute a single scenario from the command line:

```python
python main.py --scenario my_scenario
```

### Configuration
The model configuration is managed through an **Excel file (`scenario_defaults.xlsx`)**, which defines key parameters such as:
- Load profiles
- Grid share constraints
- Carbon and REC price scenarios
- Battery inclusion and specifications

## Key Components
### `ppamodule.py`
- **`PPAModel`**: Main class that runs PPA optimization.
- **Grid & REC Calculation**: Processes grid data, REC values, and KEPCO pricing.
- **Cost Analysis**: Estimates CAPEX, OPEX, and LCOE.
- **PyPSA Network Simulation**: Builds a network for energy dispatch and optimization.
- **Output Processing**: Generates key insights on energy mix, emissions, and cost breakdowns.

### `main.py`
- Streamlit-based UI for configuring and running simulations.
- Loads default parameters from `scenario_defaults.xlsx`.
- Supports running single or multiple scenarios.

## Example Output
Upon successful execution, results are saved as an **Excel file** containing:
- **LCOE Calculations**: Cost analysis for different energy sources.
- **Generation Mix**: Share of grid, PV, wind, and agrivoltaics.
- **Marginal Cost Analysis**: Energy cost breakdown.
- **Carbon and REC Analysis**: Emissions and compliance costs.
- **Battery Impact Assessment**: Storage requirements and cost evaluation.

## Contributing
We welcome contributions to improve this repository! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-new`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to your fork (`git push origin feature-new`).
5. Submit a pull request.

## License
This project is licensed under the **GNU General Public License v3.0**.

## Contact
For inquiries or collaboration, please contact: [sanghyun@planit.institute]
