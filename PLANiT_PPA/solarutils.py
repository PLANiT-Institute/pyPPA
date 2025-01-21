import sqlite3
import requests
import time
from datetime import datetime, timedelta
import calendar
import pandas as pd
import plotly.express as px
import numpy as np

# Function to fetch data for a given date range and station
def fetch_ghi_data(service_key, station_id, start_date, end_date):
    base_url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    params = {
        "serviceKey": service_key,
        "numOfRows": "999",  # Maximum rows per page
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "stnIds": station_id,
        "startDt": start_date,
        "startHh": "00",
        "endDt": end_date,
        "endHh": "23",
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for station {station_id} from {start_date} to {end_date}: {e}")
        return None

# Function to split the date range into manageable chunks (e.g., monthly)
def date_range_chunks(start_date, end_date, chunk_size=30):
    chunks = []
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date_dt = datetime.strptime(end_date, "%Y%m%d")
    while current_date <= end_date_dt:
        chunk_end_date = current_date + timedelta(days=chunk_size - 1)
        if chunk_end_date > end_date_dt:
            chunk_end_date = end_date_dt
        chunks.append((current_date.strftime("%Y%m%d"), chunk_end_date.strftime("%Y%m%d")))
        current_date = chunk_end_date + timedelta(days=1)
    return chunks

# Function to create SQLite database and tables
def create_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create `stations` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stations (
        station_id INTEGER PRIMARY KEY,
        station_name TEXT NOT NULL
    );
    """)

    # Create `ghi_data` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ghi_data (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        icsr REAL,
        FOREIGN KEY (station_id) REFERENCES stations (station_id)
    );
    """)

    conn.commit()
    return conn

# Function to insert station metadata into SQLite
def insert_stations(conn, stations):
    cursor = conn.cursor()
    station_records = [{"station_id": station_id, "station_name": f"Station {station_id}"} for station_id in stations]
    cursor.executemany("""
    INSERT OR IGNORE INTO stations (station_id, station_name)
    VALUES (:station_id, :station_name)
    """, station_records)
    conn.commit()

# Function to insert GHI data into SQLite
def insert_ghi_data(conn, df):
    # Ensure the timestamp column is in ISO 8601 string format
    df['timestamp'] = pd.to_datetime(df['tm']).dt.strftime('%Y-%m-%d %H:%M:%S')
    ghi_records = df[['stnId', 'timestamp', 'icsr']].rename(columns={'stnId': 'station_id'}).to_dict('records')
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO ghi_data (station_id, timestamp, icsr)
    VALUES (:station_id, :timestamp, :icsr)
    """, ghi_records)
    conn.commit()

# Main function to process data and save to SQLite
def process_and_save_to_database(service_key, stations, start_date, end_date, db_file):
    conn = create_database(db_file)
    insert_stations(conn, stations)

    for station in stations:
        print(f"Fetching data for station {station}...")
        date_chunks = date_range_chunks(start_date, end_date)

        for start, end in date_chunks:
            print(f"  Date range: {start} to {end}")
            data = fetch_ghi_data(service_key, station, start, end)
            if data and "response" in data and "body" in data["response"] and "items" in data["response"]["body"]:
                items = data["response"]["body"]["items"]["item"]
                if items:
                    df = pd.DataFrame(items)
                    # df = df[['tm', 'stnId', 'icsr']]  # Filter relevant columns
                    insert_ghi_data(conn, df)
                    print(f"  {len(df)} records inserted for station {station}.")
                else:
                    print(f"  No data found for station {station} from {start} to {end}.")
            else:
                print(f"  No valid data found for station {station} in range {start} to {end}.")

            time.sleep(1)  # Respect API rate limits

    conn.close()

def compute_stats(file_name, quantiles=None, plot=False, plot_quantiles=None):
    """
    Compute statistics from a SQLite database, including mean and custom quantiles.
    Optionally, plot hourly statistics for specified quantiles by month.

    Parameters:
    - file_name (str): Path to the SQLite database.
    - quantiles (list of float): List of desired quantile values (e.g., [0.1, 0.2, 0.3] for 10%, 20%, 30%).
    - plot (bool): Whether to plot hourly statistics for each month. Default is False.
    - plot_quantiles (list of float): List of quantiles to plot if plot=True. Must be a subset of `quantiles`.

    Returns:
    - DataFrame: DataFrame containing computed statistics grouped by month and hour.
    """
    import sqlite3
    import pandas as pd
    import matplotlib.pyplot as plt

    # Connect to SQLite database
    conn = sqlite3.connect(file_name)

    # Query to fetch GHI data grouped by month and hour
    query = """
    SELECT 
        strftime('%m', timestamp) AS month,
        strftime('%H', timestamp) AS hour,
        icsr
    FROM 
        ghi_data
    ORDER BY 
        month, hour;
    """

    # Load data into a DataFrame
    ghi_data = pd.read_sql_query(query, conn)

    # Convert month and hour to integers
    ghi_data['month'] = ghi_data['month'].astype(int)
    ghi_data['hour'] = ghi_data['hour'].astype(int)

    # Close database connection
    conn.close()

    # Ensure 'icsr' is numeric
    ghi_data['icsr'] = pd.to_numeric(ghi_data['icsr'], errors='coerce')
    ghi_data['icsr'] = ghi_data['icsr'].fillna(0)

    # Default quantiles if none are provided
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # Default percentiles

    # Define a function to calculate quantiles
    def quantile_func(q):
        return lambda x: x.quantile(q)

    # Create the aggregation dictionary
    agg_dict = {
        'mean': ('icsr', 'mean')  # Add mean
    }

    # Add quantiles dynamically
    for q in quantiles:
        agg_dict[f'q{int(q * 100)}'] = ('icsr', lambda x, q=q: x.quantile(q))

    # Compute statistics
    stats = ghi_data.groupby(['month', 'hour']).agg(**agg_dict).reset_index()

    # Flatten the column names
    stats.columns = ['month', 'hour'] + list(agg_dict.keys())

    # If plotting is enabled, create the plot with bands
    if plot:
        plt.figure(figsize=(12, 8))

        for month in range(1, 13):  # Loop through months (1 to 12)
            subset = stats[stats['month'] == month]

            # Plot quantile bands if specified
            if plot_quantiles:
                for lower_q, upper_q in plot_quantiles:
                    lower_label = f'q{int(lower_q * 100)}'
                    upper_label = f'q{int(upper_q * 100)}'

                    if lower_label in stats.columns and upper_label in stats.columns:
                        plt.fill_between(
                            subset['hour'],
                            subset[lower_label],
                            subset[upper_label],
                            alpha=0.2,
                            label=f"Month {month:02} ({lower_q*100:.0f}%-{upper_q*100:.0f}%)"
                        )

            # Plot mean or median as a line
            if 'mean' in stats.columns:
                plt.plot(subset['hour'], subset['mean'], label=f"Month {month:02} Mean")

        # Customize the plot
        plt.title("Hourly GHI Pattern by Month (Mean and Quantile Bands)")
        plt.xlabel("Hour of Day")
        plt.ylabel("GHI (icsr)")
        plt.xticks(range(0, 24))  # Ensure all 24 hours are visible
        plt.legend(title="Month")
        plt.grid(True)
        plt.show()

    return stats

def generate_yearly_pattern(df, column_name='mean', years=range(datetime.now().year, datetime.now().year + 1), normalize=True, plot=False):
    """
    Generate yearly patterns (8760 or 8784 hours per year) by repeating daily patterns for each month
    and optionally plot the results.

    Parameters:
    - df (DataFrame): DataFrame containing aggregated statistics grouped by month and hour.
    - column_name (str): The column to use for generating the pattern.
    - years (iterable): A range or list of years for which to generate the patterns.
    - normalize (bool): Whether to normalize the values between 0 and 1 for each year.
    - plot (bool): Whether to plot the resulting yearly patterns.

    Returns:
    - DataFrame: A DataFrame with the yearly patterns and an interactive plot if `plot=True`.
    """
    all_yearly_data = []

    for year in years:
        is_leap_year = calendar.isleap(year)
        total_hours = 8784 if is_leap_year else 8760

        # Prepare the hourly pattern for each month
        hourly_pattern = df.groupby(['month', 'hour'])[column_name].mean().reset_index()

        # Create the yearly pattern for the current year
        yearly_data = []
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                for hour in range(24):
                    value = hourly_pattern.loc[
                        (hourly_pattern['month'] == month) & (hourly_pattern['hour'] == hour),
                        column_name
                    ].values[0]
                    yearly_data.append({
                        'year': year,
                        'month': month,
                        'day': day,
                        'hour': hour,
                        column_name: value
                    })

        # Create DataFrame for the current year's pattern
        yearly_df = pd.DataFrame(yearly_data)

        # Adjust to match the total hours for leap or regular year
        yearly_df = yearly_df.iloc[:total_hours].reset_index(drop=True)

        # Normalize values if the normalize argument is True
        if normalize:
            max_value = yearly_df[column_name].max()
            yearly_df[column_name] = yearly_df[column_name] / max_value if max_value != 0 else yearly_df[column_name]

        # Append to the list of all yearly data
        all_yearly_data.append(yearly_df)

    # Combine all yearly DataFrames into one
    combined_df = pd.concat(all_yearly_data, ignore_index=True)

    # Create a datetime column
    combined_df['datetime'] = pd.to_datetime({
        'year': combined_df['year'],
        'month': combined_df['month'],
        'day': combined_df['day'],
        'hour': combined_df['hour']
    })

    # Plot the data if requested
    if plot:
        fig = px.line(
            combined_df,
            x='datetime',
            y=column_name,
            title=f"Yearly {column_name.upper()} Patterns",
            labels={
                'datetime': 'Date',
                column_name: f"GHI ({column_name})"
            }
        )

        # Customize the layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=f"GHI ({column_name})",
            template='plotly_white',
            hovermode='x unified'
        )

        # Show the plot
        fig.show()

    return combined_df

def adjust_column_to_target(dataframe, column_name, target, rand=None, max_adj_percent_diff=None, plot=False):
    """
    Adjust the values in a specified column to achieve the target mean value within 0.1% difference
    for each year or a single target value across all years, and return the ratio of total sum
    to total hours (rows) for each year.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the column to adjust and a 'year' column.
    - column_name (str): The name of the column to adjust.
    - target (dict or float): A dictionary where keys are years and values are the target mean values for each year,
      or a single float value to apply the same target to all years.
    - rand (tuple or None): A tuple (low, high) for the range of random numbers. If None, defaults to (0,1).
    - max_adj_percent_diff (float or None): The maximum allowed percentage difference between
      two adjacent random increments. For example, 0.2 means no two adjacent increments differ by more than 20%.
      If None, this constraint is ignored.
    - plot (bool): Whether to plot the column before and after adjustment. Default is False.

    Returns:
    - pd.DataFrame: The DataFrame with the adjusted column values.
    - dict: A dictionary with year as keys and (total sum / total hours of that year) as values.
    """

    df = dataframe.copy()

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if 'year' not in df.columns:
        raise ValueError("DataFrame must contain a 'year' column.")

    # If target is a single value, create a dictionary for all years
    if isinstance(target, (int, float)):
        unique_years = df['year'].unique()
        target_dict = {year: target for year in unique_years}
    elif isinstance(target, dict):
        target_dict = target
    else:
        raise ValueError("Target must be a dictionary or a single numeric value.")

    # Default rand range if None
    if rand is None:
        rand = (0, 1)
    else:
        rand = (rand[0], rand[1])

    adjusted_dfs = []
    year_ratios = {}  # To store total sum / total hours per year

    for year, target_value in target_dict.items():
        df_year = df[df['year'] == year].copy()
        nrows = len(df_year)
        if nrows == 0:
            continue  # No data for this year

        current_mean = df_year[column_name].mean()
        gap = (target_value - current_mean) * nrows  # Total amount to adjust

        # Generate random numbers
        random_numbers = np.random.uniform(rand[0], rand[1], size=nrows)

        # Apply max_adj_percent_diff constraints if required
        if max_adj_percent_diff is not None:
            for i in range(1, nrows):
                allowed_min = random_numbers[i-1] * (1 - max_adj_percent_diff)
                allowed_max = random_numbers[i-1] * (1 + max_adj_percent_diff)
                allowed_min = max(allowed_min, 0)
                allowed_max = min(allowed_max, 1)
                if random_numbers[i] < allowed_min:
                    random_numbers[i] = allowed_min
                elif random_numbers[i] > allowed_max:
                    random_numbers[i] = allowed_max

        if gap >= 0:
            # Increase values
            max_possible_increase = 1 - df_year[column_name]
            total_possible_increase = max_possible_increase.sum()
            if total_possible_increase < gap:
                raise ValueError(f"Not enough capacity to increase values to reach the target for year {year}.")

            possible_increments = max_possible_increase * random_numbers
            scaling_factor = gap / possible_increments.sum()
            per_row_adjustment = possible_increments * scaling_factor
            df_year[column_name] += per_row_adjustment
        else:
            # Decrease values
            gap = -gap  # Make gap positive for reduction
            max_possible_reduction = df_year[column_name]
            total_possible_reduction = max_possible_reduction.sum()
            if total_possible_reduction < gap:
                raise ValueError(f"Not enough capacity to reduce values to reach the target for year {year}.")

            possible_reductions = max_possible_reduction * random_numbers
            scaling_factor = gap / possible_reductions.sum()
            per_row_adjustment = possible_reductions * scaling_factor
            df_year[column_name] -= per_row_adjustment

        # Ensure values are within [0, 1]
        df_year[column_name] = df_year[column_name].clip(0, 1)

        # Verify the adjustment
        adjusted_mean = df_year[column_name].mean()
        if abs(adjusted_mean - target_value) > target_value * 0.001:
            raise ValueError(
                f"Unable to achieve the target value within 0.1% difference for year {year}. Adjusted mean: {adjusted_mean:.5f}."
            )

        # Compute total sum / total hours (rows) for this year
        year_total = df_year[column_name].sum()
        ratio = year_total / nrows
        year_ratios[year] = ratio

        adjusted_dfs.append(df_year)

    # Concatenate adjusted dataframes
    adjusted_df = pd.concat(adjusted_dfs) if adjusted_dfs else pd.DataFrame(columns=df.columns)
    # Merge back any data for years not in target_dict
    remaining_df = df[~df['year'].isin(target_dict.keys())]
    result_df = pd.concat([adjusted_df, remaining_df]).sort_index()

    # Optional plotting
    if plot:
        if 'datetime' not in df.columns:
            raise ValueError("To plot, the DataFrame must contain a 'datetime' column.")
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError("Plotly must be installed to use the plotting feature.")

        plot_data = pd.DataFrame({
            "Index": df['datetime'],
            "Original": df[column_name],
            "Adjusted": result_df.loc[df.index, column_name]
        })
        fig = px.line(plot_data, x="Index", y=["Original", "Adjusted"],
                      labels={"value": "Value", "variable": "Type"})
        fig.update_layout(legend_title_text="Series",
                          template='plotly_white')
        fig.show()

    return year_ratios, result_df


def create_linear_target_dict(start_year, end_year, start_value, end_value, digits = 2):
    """
    Creates a dictionary with years as keys and values linearly increasing from start_value to end_value.

    :param start_year: The first year in the dictionary (inclusive).
    :param end_year: The last year in the dictionary (inclusive).
    :param start_value: The starting value for the first year.
    :param end_value: The ending value for the last year.
    :return: A dictionary with years as keys and linearly increasing values.
    """
    num_years = end_year - start_year
    step = (end_value - start_value) / num_years
    return {year: round(start_value + step * (year - start_year), digits) for year in range(start_year, end_year + 1)}

def get_days_by_year_range(start_year, end_year):
    """
    Returns a dictionary where the keys are years and the values are the number of days in each year.

    :param start_year: The starting year (inclusive).
    :param end_year: The ending year (inclusive).
    :return: A dictionary with years as keys and the number of days as values.
    """
    # Generate a date range covering the full years
    date_range = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='D')

    # Group by year and count days
    days_per_year = date_range.to_series().groupby(date_range.year).size().to_dict()

    return days_per_year