import pandas as pd


def preprocess_data(processed_mat_df: pd.DataFrame) -> pd.DataFrame | None:
    """This function processes the DataFrame from the processed_mat_file function to a format that can be used for the BucketModel.

    Parameters:
    - processed_mat_df (pd.DataFrame): The DataFrame containing the data from the processed_mat_file function.

    Returns:
    - pd.DataFrame | None: A DataFrame containing the data in a format that can be used for the BucketModel. If the DataFrame contains the "Simulations" column, the function will return None.
    """

    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    if "Simulation" in processed_mat_df.columns:
        print(
            "Simulations column detected. This function is not designed to handle simulations data. \nConsider looping through the simulations and using this function on each simulation."
        )
        return None
    df = processed_mat_df.copy()

    df["Month"] = df["Month"].map(months)
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    df = df.set_index("date")

    df["P_mix"] = df["Precipitation"]

    keep_columns = ["P_mix", "T_max", "T_min"]
    df = df[keep_columns]

    return df


def train_validate_split(data: pd.DataFrame, train_size: float) -> tuple:
    """This function splits the data into training and validating sets.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - train_size (float): The proportion of the data to use for training. This is a value between 0 and 1.

    Returns:
    - tuple: A tuple containing the training and testing DataFrames.
    """

    train_size = int(len(data) * train_size)
    train_data = data.iloc[:train_size]
    validate_data = data.iloc[train_size:]

    return train_data, validate_data


def main() -> None:
    """
    This is an example of how you can use the preprocess_data function. You need to change the path_to_file and output_destination to your own paths.
    Alternatively you can import this function into another script and use it there. See example_run.ipynb for more information.
    """

    path_to_file = "/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.txt"
    output_destination = "/Users/cooper/Desktop/bucket-model/data/GSTEIGmeteo.csv"
    catchment_area = 384.2  # km^2
    data = preprocess_data(path_to_file, output_destination, catchment_area)
    print(data)


if __name__ == "__main__":
    main()
