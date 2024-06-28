from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import numpy as np


@dataclass
class UncertaintyAnalysis:
    """
    A class to analyze different types of uncertainties in climate data.

    Attributes:
    - data (pd.DataFrame): The input climate data.

    Methods:
    - calculate_tu: Calculate Total Uncertainty.
    - calculate_eu: Calculate Emission Scenario Uncertainty.
    - calculate_cmu: Calculate Climate Model Uncertainty.
    - calculate_su: Calculate Stochastic Uncertainty.
    """

    data: pd.DataFrame

    def calculate_tu(self, column: str) -> float:
        """
        Calculate Total Uncertainty (TU) for a given column.

        Parameters:
        - column (str): The column to calculate TU for.

        Returns:
        - float: The calculated Total Uncertainty.
        """
        lower_percentile = self.data[column].quantile(0.05)
        upper_percentile = self.data[column].quantile(0.95)
        tu = upper_percentile - lower_percentile
        return round(tu, 2)

    def calculate_eu(self, column: str, tu: float) -> Tuple[float, float]:
        """
        Calculate Emission Scenario Uncertainty (EU) and its partition.

        Parameters:
        - column (str): The column to calculate EU for.
        - tu (float): The Total Uncertainty value.

        Returns:
        - Tuple[float, float]: The calculated EU and its partition.
        """
        rcp85_data = self.data[self.data["Scenario"] == "RCP8.5"]
        rcp45_data = self.data[self.data["Scenario"] == "RCP4.5"]

        median_rcp85 = rcp85_data[column].median()
        median_rcp45 = rcp45_data[column].median()

        eu = abs(median_rcp85 - median_rcp45)
        partition = eu / tu

        return round(eu, 2), round(partition, 2)

    def calculate_cmu(self, column: str, tu: float) -> Tuple[float, float]:
        """
        Calculate Climate Model Uncertainty (CMU) and its partition.

        Parameters:
        - column (str): The column to calculate CMU for.
        - tu (float): The Total Uncertainty value.

        Returns:
        - Tuple[float, float]: The calculated CMU and its partition.
        """
        model_means = (
            self.data.groupby(["Climate_Model", "Scenario"])[column]
            .mean()
            .reset_index()
        )
        rcp_ranges = model_means.groupby("Scenario")[column].apply(
            lambda x: x.max() - x.min()
        )
        cmu = rcp_ranges.mean()
        partition = cmu / tu

        return round(cmu, 2), round(partition, 2)

    def calculate_su(self, column: str, tu: float) -> Tuple[float, float]:
        """
        Calculate Stochastic Uncertainty (SU) and its partition.

        Parameters:
        - column (str): The column to calculate SU for.
        - tu (float): The Total Uncertainty value.

        Returns:
        - Tuple[float, float]: The calculated SU and its partition.
        """
        simulation_ranges = (
            self.data.groupby(["Climate_Model", "Scenario"])[column]
            .apply(lambda x: x.max() - x.min())
            .reset_index()
        )
        rcp_averages = simulation_ranges.groupby("Scenario")[column].mean()
        su = rcp_averages.mean()
        partition = su / tu

        return round(su, 2), round(partition, 2)
