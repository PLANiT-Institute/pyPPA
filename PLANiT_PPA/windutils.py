import pandas as pd
import plotly.express as px
import numpy as np

def adjust_hourly_pattern_to_targets(
    df: pd.DataFrame,
    target: float or dict,
    rand: tuple = None,
    max_adj_percent_diff: float or None = None,
    plot: bool = False
):
    """
    Adjust values in all numeric columns of a DataFrame to achieve a target annual mean
    (within ±0.1%) for each year or for a single target across all years.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex and numeric columns.
    target : float or dict
        - If float (or int), the same target (0–1) is applied to all years.
        - If dict, keys are years (int) and values are the annual mean targets.
    rand : tuple or None
        (low, high) for uniform random distribution used in the gap-increment logic.
        Defaults to (0, 1) if None.
    max_adj_percent_diff : float or None
        If set, adjacent random increments differ by no more than this fraction.
        If None, no constraint is applied.
    plot : bool
        If True, plot each column before vs. after adjustment (matplotlib).

    Returns
    -------
    year_ratios : dict of dict
        Nested dict mapping [column][year] -> final mean
        (i.e., sum of adjusted column / total rows for that year).
    df_adjusted : pd.DataFrame
        The DataFrame with adjusted values in all numeric columns.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DateTimeIndex.")

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns

    # Exclude 'year' column from numeric columns to adjust
    numeric_cols = numeric_cols.drop('year', errors='ignore')

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to adjust.")

    # Extract year from index
    df = df.copy()
    df['year'] = df.index.year

    # Parse target into a dict {year: value}
    if isinstance(target, (float, int)):
        unique_years = df['year'].unique()
        target_dict = {year: float(target) for year in unique_years}
    elif isinstance(target, dict):
        target_dict = target
    else:
        raise ValueError("`target` must be a float/int or dict {year: target_value}.")

    # Default random range
    if rand is None:
        rand = (0, 1)

    year_ratios = {}  # To store final means per column-year

    for year, target_value in target_dict.items():
        df_year_mask = df['year'] == year
        nrows = df_year_mask.sum()
        if nrows == 0:
            continue

        for col in numeric_cols:
            current_vals = df.loc[df_year_mask, col].copy()
            current_mean = current_vals.mean()

            if current_vals.isna().all():
                continue

            gap = (target_value - current_mean) * nrows
            random_numbers = np.random.uniform(rand[0], rand[1], size=nrows)

            if max_adj_percent_diff is not None:
                for i in range(1, nrows):
                    low = random_numbers[i - 1] * (1 - max_adj_percent_diff)
                    high = random_numbers[i - 1] * (1 + max_adj_percent_diff)
                    low = max(low, 0)
                    high = min(high, 1)
                    random_numbers[i] = min(max(random_numbers[i], low), high)

            if gap > 0:
                max_possible_increase = 1 - current_vals
                if max_possible_increase.sum() < gap:
                    raise ValueError(f"Year {year}, col '{col}': Not enough headroom.")
                increments = max_possible_increase * random_numbers
                scaling_factor = gap / increments.sum()
                df.loc[df_year_mask, col] = current_vals + increments * scaling_factor
            elif gap < 0:
                gap = abs(gap)
                max_possible_decrease = current_vals
                if max_possible_decrease.sum() < gap:
                    raise ValueError(f"Year {year}, col '{col}': Not enough capacity.")
                reductions = max_possible_decrease * random_numbers
                scaling_factor = gap / reductions.sum()
                df.loc[df_year_mask, col] = current_vals - reductions * scaling_factor

            df.loc[df_year_mask, col] = df.loc[df_year_mask, col].clip(0, 1)

            final_mean = df.loc[df_year_mask, col].mean()
            if abs(final_mean - target_value) > abs(target_value) * 0.001:
                raise ValueError(
                    f"Year {year}, col '{col}': Adjusted mean {final_mean:.5f} "
                    f"differs from target {target_value:.5f} by >0.1%."
                )

    df_adjusted = df.drop(columns='year')
    for col in numeric_cols:
        year_ratios[col] = {
            year: df[df['year'] == year][col].mean()
            for year in target_dict
        }

    if plot:
        import matplotlib.pyplot as plt
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[col], label="Original", alpha=0.5)
            plt.plot(df.index, df_adjusted[col], label="Adjusted", linestyle="--")
            plt.title(f"Adjustment of '{col}' to Target(s)")
            plt.xlabel("Datetime")
            plt.ylabel(col)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return year_ratios, df_adjusted

def extend_hourly_pattern_to_target_years(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    plot: bool = False
) -> pd.DataFrame:
    """
    Extends an hourly pattern DataFrame to cover a specified year range,
    copying and repeating data as needed, and handling leap years.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex containing hourly data.
    start_year : int
        Start year of the target period.
    end_year : int
        End year of the target period.
    plot : bool, optional
        If True, generates interactive Plotly plots of the extended data.

    Returns
    -------
    extended_df : pd.DataFrame
        Extended DataFrame with hourly data for all years in the target range.
    """
    from datetime import date

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The input DataFrame must have a DatetimeIndex.")

    # Ensure all columns are numeric, converting if necessary
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Generate the full target hourly index
    target_index = pd.date_range(
        start=f"{start_year}-01-01 00:00:00",
        end=f"{end_year}-12-31 23:00:00",
        freq="h"
    )

    # Prepare an empty DataFrame with the target index
    extended_df = pd.DataFrame(index=target_index, columns=df.columns)

    # Helper function to check if a year is a leap year
    def is_leap_year(year):
        return date(year, 3, 1).toordinal() - date(year, 2, 1).toordinal() == 29

    # Loop through each year in the target range
    for year in range(start_year, end_year + 1):
        # Check if the year is already in the original dataset
        original_years = df.index.year.unique()
        if year in original_years:
            year_data = df[df.index.year == year]
        else:
            # Reuse data from the first available year
            year_data = df.copy()

            # Adjust dates to match the current target year
            def adjust_date(dt):
                try:
                    return dt.replace(year=year)
                except ValueError:
                    # Handle February 29 in non-leap years
                    if dt.month == 2 and dt.day == 29:
                        return dt.replace(year=year, day=28)
                    raise

            year_data.index = year_data.index.map(adjust_date)

            # Handle leap years: Add February 29 if the target year is a leap year
            if is_leap_year(year):
                feb_28 = pd.Timestamp(f"{year}-02-28")
                feb_29 = pd.Timestamp(f"{year}-02-29")
                if feb_28 in year_data.index and feb_29 not in year_data.index:
                    # Copy values from February 28 to February 29
                    leap_day_data = year_data.loc[year_data.index.date == feb_28.date()].copy()
                    leap_day_data.index = leap_day_data.index.map(lambda dt: dt.replace(day=29))
                    year_data = pd.concat([year_data, leap_day_data])

        # Assign data for this year to the extended DataFrame
        extended_df.loc[year_data.index, :] = year_data.values

    # Ensure all columns in extended DataFrame are numeric
    for col in extended_df.columns:
        extended_df[col] = pd.to_numeric(extended_df[col], errors="coerce")

    # Interpolate numeric columns only
    extended_df = extended_df.interpolate(limit_direction="both")

    # Optional interactive plotting
    if plot:
        fig = px.line(
            extended_df.reset_index(),
            x="index",
            y=extended_df.columns,
            labels={"index": "Datetime", "value": "Values"},
            title="Extended Hourly Patterns"
        )
        fig.update_layout(template="plotly_white")
        fig.show()

    return extended_df
