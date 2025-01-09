import pandas as pd
import PLANiT_PPA_utils.ppamodule as _ppa

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Disable scientific notation
pd.set_option('display.float_format', '{:,.2f}'.format)

# ------------------------------
# EXAMPLE USAGE
# ------------------------------
if __name__ == "__main__":
    # Aggregate loads (SK, Samsung) into one total load
    loads_config = {
        "SK": 3000,
        "Samsung": 3000
    }

    # 배터리 포함여부
    battery_include = False

    # 환율
    currency_exchange = 1400  # KRW / USD

    # 전력망 공급 비중
    max_grid_share = 0.5
    sense = '=='

    # Scenario: 산업용전력요금
    # Reference: 연 5% 산업용 전력 요금 상승
    # High: 연 10% 산업용 전력 요금 상승
    rate_increase = 0.05  # 5% annual increase in electricity rates
    selected_sheet = "HV_C_III"

    # Scenario: 배출권거래제
    # Reference: 11,000 (고정)
    # NGFS: 11,000 (2023) ~ NGFS 시나리오 중 Net Zero 2050 (환율: 1400 원)
    carbonprice_init = 'NGFS'  # 11000    # KRW/ton # Select carbon price

    # Scenario: 전력망 REC
    # Reference: REC 가격 유지 (80,000 원/REC)
    # 가격하락: REC 가격 하락 (80,000 원/REC → 0 원/REC 2050)
    rec_grid_init = 80000  # KRW/MWh
    rec_reduction = True  # True: REC 감소 to 0
    rec_increase = 0.00  # 5% annual increase in REC price, if rec_reduction = False

    # 재생에너지 비용 SMP 기준 (SMP vs 가격 시장)
    # Reference: 직접 PPA 계약 비용 >= SMP + REC
    # Alternative: 직접 PPA 계약비용 == 재생에너지 LCOE
    # Model selection
    smp_limit = True  # True: 직접PPA가격 >= SMP + REC
    smp_init = 167000  # KRW/MWh (2023, EPSIS)
    smp_increase = 0.05

    # Scenario: 직접 PPA REC 불인정
    # Reference: 직접 PPA 시 REC 소각
    # REC 인정: 직접 PPA 시에도 REC 인정. 즉, 직접 PPA 계약금액에 REC 미반영
    rec_include = False  # True: 직접 PPA REC 불인정

    # Define start and end year
    initial_year = 2023
    analysis_target_year = 2050
    start_year = 2030
    end_year = 2030

    default_params = {
        'buffer': 25000,
        'bin_size': 1,
        'min_capacity': 0.1,
        'connection_fees': 0,
        'power_density': 120,
        'digits': None,
        'dist_cols': ['distance_0'],
        'max_distace': 20000,  # m
        'wind_cap': 5,  # MW
    }

    battery_parameters = {
        "include": battery_include,
        "capital_cost_per_mw": 1_000_000_000,
        "capital_cost_per_mwh": 250_000_000,
        "efficiency_store": 0.9,
        "efficiency_dispatch": 0.9,
        "max_hours": 4,
    }

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
        rec_grid_init,
        rec_reduction,
        rec_include,
        rec_increase,
        smp_limit,
        smp_init,
        smp_increase,
        initial_year,
        analysis_target_year,
        start_year,
        end_year,
        # Parameters
        default_params,
        # Battery Parameters
        battery_parameters
    )

    # Run the model
    output = ppa_model.run_model()
