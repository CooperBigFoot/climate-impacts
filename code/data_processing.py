import pandas as pd
import scipy.io
import os


# Define the custom order of months
MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def _map_numeric_to_month(month_numeric: int) -> str:
    """
    Map a numeric month to its corresponding string representation.

    Args:
        month_numeric (int): The numeric representation of the month.

    Returns:
        str: The string representation of the month or 'Invalid Month'.
    """
    if 1 <= month_numeric <= 12:
        return MONTH_ORDER[month_numeric - 1]
    return "Invalid Month"


def _is_input_data_mat_file(file_path: str) -> bool:
    """
    Check if the file is the Input_data.mat file.

    Args:
        file_path (str): The path of the file.

    Returns:
        bool: True if the file is the Input_data.mat file, False otherwise.
    """

    return os.path.basename(file_path) == "Input_data.mat"


def add_year_column(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Add a Year column to the DataFrame based on the file path.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The DataFrame with the added Year column.
    """
    if _is_input_data_mat_file(file_path):
        df["Year"] = (df.index // 365) + 1986
    else:
        df["Year"] = (df.index // 365) + 1980
    return df


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Date column to the DataFrame based on the Year column

    Args:
        df: The DataFrame to transform

    Returns:
        The DataFrame with the added Date column
    """
    start_date = pd.to_datetime(f'{df["Year"].iloc[0]}-01-01')
    df["Date"] = start_date + pd.to_timedelta(df.index % 365, unit="D")
    return df


def handle_leap_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust dates in the DataFrame for leap years to align dates from March 1st onwards with non-leap years.

    This function identifies all leap years within the DataFrame's 'Date' column. For each leap year identified,
    it shifts all dates from March 1st and onwards by one day forward.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Date' column in datetime format.

    Returns:
        pd.DataFrame: Modified DataFrame with adjusted dates for leap years.

    Note:
        Only adjusts dates from March 1st onwards in leap years, earlier dates in leap years are not affected.
    """
    leap_years = df["Date"].dt.year[df["Date"].dt.is_leap_year].unique()
    for year in leap_years:
        start_date = pd.to_datetime(f"{year}-03-01")
        mask = (df["Date"].dt.year == year) & (df["Date"].dt.month >= 3)
        df.loc[mask, "Date"] += pd.DateOffset(days=1)
    return df


def skip_feb_29(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skip February 29 in the Date column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The DataFrame with February 29 skipped.
    """
    df.loc[
        (df["Date"].dt.month == 2) & (df["Date"].dt.day == 29), "Date"
    ] += pd.DateOffset(days=1)
    return df


def add_month_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Month and Day columns to the DataFrame based on the Date column.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The DataFrame with the added Month and Day columns.
    """
    df["Month"] = df["Date"].dt.month.apply(_map_numeric_to_month)
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date"], inplace=True)
    return df


def transform_dates(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Transform the DataFrame by adding Year, Month, and Day columns.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    return (
        df.pipe(add_year_column, file_path)
        .pipe(add_date_column)
        .pipe(handle_leap_years)
        .pipe(skip_feb_29)
        .pipe(add_month_day_columns)
    )


def process_mat_file(file_path: str) -> pd.DataFrame:
    """
    Process the .mat file and return its content as a DataFrame.

    Args:
        file_path (str): The path of the .mat file.

    Returns:
        pd.DataFrame: The content of the .mat file as a DataFrame.
    """

    mat_data = scipy.io.loadmat(file_path)

    if _is_input_data_mat_file(file_path):
        df = pd.DataFrame(
            {key: mat_data[key].flatten() for key in ["P", "Tmax", "Tmin"]}
        )
        df.columns = ["Precipitation", "T_max", "T_min"]
        df["T_avg"] = (df["T_max"] + df["T_min"]) / 2  # Add T_avg column
        return transform_dates(df, file_path)

    all_realizations_df = pd.DataFrame()

    for realization in mat_data["TS"]:

        df = pd.DataFrame(
            realization[0][:, 1:], columns=["Precipitation", "T_max", "T_min"]
        )
        df["T_avg"] = (df["T_max"] + df["T_min"]) / 2  # Add T_avg column
        all_realizations_df = pd.concat(
            [all_realizations_df, transform_dates(df, file_path)], ignore_index=True
        )

        all_realizations_df["Simulation"] = (
            all_realizations_df.index // (30 * 365)
        ) + 1

    return all_realizations_df
