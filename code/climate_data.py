from dataclasses import dataclass, field
import pandas as pd
import scipy.io
import os
from typing import Optional, Tuple, Dict
import copy


@dataclass
class ClimateData:
    """
    A class to process and manage climate data from .mat files.

    This class provides methods for reading .mat files, preprocessing the data,
    and preparing it for use with the BucketModel. It handles both single dataset
    and multiple simulation datasets.

    Attributes:
    - _data (Optional[pd.DataFrame]): The processed climate data.
    - file_path (Optional[str]): Path to the .mat file.

    Class Attributes:
    - MONTH_ORDER (Dict[str, int]): The order of months used for sorting.

    Methods:
    - read_mat_file: Read and process a .mat file.
    - preprocess_for_bucket_model: Prepare data for use with the BucketModel.
    - train_validate_split: Split the data into training and validation sets.
    - create_datetime_index: Create a DateTime index for the data.
    - copy: Create a deep copy of the ClimateData object.
    """

    _data: Optional[pd.DataFrame] = field(default=None, init=False)
    file_path: Optional[str] = field(default=None, init=False)

    MONTH_ORDER: Dict[str, int] = field(
        default_factory=lambda: {
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
    )

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data

    def read_mat_file(self, file_path: str) -> None:
        """
        Read and process a .mat file containing climate data.

        This method reads the specified .mat file, processes its contents,
        and stores the result in the _data attribute. It handles both
        Input_data.mat files and files containing multiple simulations.

        Parameters:
        - file_path (str): The path to the .mat file to be read.

        Raises:
        - FileNotFoundError: If the specified file does not exist.
        - ValueError: If the file format is not as expected.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        self.file_path = file_path
        mat_data = scipy.io.loadmat(file_path)

        if self._is_input_data_mat_file(file_path):
            df = pd.DataFrame(
                {key: mat_data[key].flatten() for key in ["P", "Tmax", "Tmin"]}
            )
            df.columns = ["Precipitation", "T_max", "T_min"]
        elif "TS" in mat_data:
            all_realizations = []
            for i, realization in enumerate(mat_data["TS"]):
                df = pd.DataFrame(
                    realization[0][:, 1:], columns=["Precipitation", "T_max", "T_min"]
                )
                df["Simulation"] = i + 1
                all_realizations.append(df)
            df = pd.concat(all_realizations, ignore_index=True)
        else:
            raise ValueError("Unexpected file format. Unable to process the data.")

        df["T_avg"] = (df["T_max"] + df["T_min"]) / 2
        self._data = self._transform_dates(df)

    def preprocess_for_bucket_model(self) -> pd.DataFrame:
        """
        Prepare the climate data for use with the BucketModel.

        This method processes the data stored in the class to make it
        compatible with the BucketModel. It includes steps such as
        converting the date information to a datetime index and selecting
        the relevant columns.

        Returns:
        - pd.DataFrame: A DataFrame containing the preprocessed data ready for the BucketModel.

        Raises:
        - ValueError: If no data has been loaded yet.
        """
        if self._data is None:
            raise ValueError("No data loaded. Please use read_mat_file() first.")

        df = self._data.copy()

        df["Month"] = df["Month"].map(self.MONTH_ORDER)
        df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df = df.set_index("date")

        df["P_mix"] = df["Precipitation"]

        keep_columns = ["P_mix", "T_max", "T_min"]
        if "Simulation" in df.columns:
            keep_columns.append("Simulation")

        return df[keep_columns]

    def train_validate_split(
        self, train_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the preprocessed data into training and validation sets.

        This method splits the data prepared by preprocess_for_bucket_model()
        into training and validation sets based on the specified train_size.

        Parameters:
        - train_size (float): The proportion of the data to use for training (0 to 1).

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation DataFrames.

        Raises:
        - ValueError: If no data has been loaded or if train_size is not between 0 and 1.
        """
        if self._data is None:
            raise ValueError("No data loaded. Please use read_mat_file() first.")

        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")

        df = self.preprocess_for_bucket_model()
        train_size = int(len(df) * train_size)
        train_data = df.iloc[:train_size]
        validate_data = df.iloc[train_size:]

        return train_data, validate_data

    def create_datetime_index(self) -> "ClimateData":
        """
        Create a DateTime index for the data.

        This method adds a 'Date' column to the data, converts it to a DateTime index,
        and sorts the data by this index. It returns a new ClimateData object with the modified data.

        Returns:
        - ClimateData: A new ClimateData object with the DateTime index.

        Raises:
        - ValueError: If no data has been loaded yet.
        """
        if self._data is None:
            raise ValueError("No data loaded. Please use read_mat_file() first.")

        new_climate_data = self.copy()
        df = new_climate_data._data

        # Convert month names to numbers if necessary
        if df["Month"].dtype == "object":
            df["Month"] = df["Month"].map(self.MONTH_ORDER)

        # Create the datetime index
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        new_climate_data._data = df
        return new_climate_data

    def copy(self) -> "ClimateData":
        """
        Create a deep copy of the ClimateData object.

        Returns:
        - ClimateData: A new ClimateData object with a copy of the data.
        """
        return copy.deepcopy(self)

    @staticmethod
    def _is_input_data_mat_file(file_path: str) -> bool:
        """
        Check if the file is the Input_data.mat file.

        Parameters:
        - file_path (str): The path of the file to check.

        Returns:
        - bool: True if the file is Input_data.mat, False otherwise.
        """
        return os.path.basename(file_path) == "Input_data.mat"

    def _add_year_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a Year column to the DataFrame based on the file path.

        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The DataFrame with the added Year column.
        """
        start_year = 1986 if self._is_input_data_mat_file(self.file_path) else 1980
        df["Year"] = (df.index // 365) + start_year
        return df

    def _add_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a Date column to the DataFrame based on the Year column.

        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The DataFrame with the added Date column.
        """
        start_date = pd.to_datetime(f'{df["Year"].iloc[0]}-01-01')
        df["Date"] = start_date + pd.to_timedelta(df.index % 365, unit="D")
        return df

    def _handle_leap_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust dates in the DataFrame for leap years.

        This method shifts all dates from March 1st onwards by one day forward
        in leap years to align with non-leap years.

        Parameters:
        - df (pd.DataFrame): The DataFrame to adjust.

        Returns:
        - pd.DataFrame: The DataFrame with adjusted dates for leap years.
        """
        leap_years = df["Date"].dt.year[df["Date"].dt.is_leap_year].unique()
        for year in leap_years:
            mask = (df["Date"].dt.year == year) & (df["Date"].dt.month >= 3)
            df.loc[mask, "Date"] += pd.DateOffset(days=1)
        return df

    def _skip_feb_29(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Skip February 29 in the Date column of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The DataFrame with February 29 skipped.
        """
        df.loc[
            (df["Date"].dt.month == 2) & (df["Date"].dt.day == 29), "Date"
        ] += pd.DateOffset(days=1)
        return df

    def _add_month_day_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Month and Day columns to the DataFrame based on the Date column.

        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The DataFrame with added Month and Day columns.
        """
        df["Month"] = df["Date"].dt.strftime("%b")
        df["Month"] = pd.Categorical(
            df["Month"], categories=list(self.MONTH_ORDER.keys()), ordered=True
        )
        df["Day"] = df["Date"].dt.day
        df.drop(columns=["Date"], inplace=True)
        return df

    def _transform_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by adding Year, Month, and Day columns.

        This method applies a series of transformations to add and adjust
        date-related columns in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.

        Returns:
        - pd.DataFrame: The transformed DataFrame with added date columns.
        """
        return (
            df.pipe(self._add_year_column)
            .pipe(self._add_date_column)
            .pipe(self._handle_leap_years)
            .pipe(self._skip_feb_29)
            .pipe(self._add_month_day_columns)
        )
