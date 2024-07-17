import pandas as pd
import numpy as np
from typing import Tuple


class UncertaintyAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        # Update scenario names if needed
        self.data["Scenario"] = self.data["Scenario"].replace(
            {"RCP4": "RCP4.5", "RCP8": "RCP8.5"}
        )

    def calculate_tu(self, column: str) -> float:
        """
        Calculate Total Uncertainty (TU) for a given column.

        Args:
            column (str): The column name.

        Returns:
            float: The Total Uncertainty value.
        """
        lower_percentile = self.data[column].quantile(0.05)
        upper_percentile = self.data[column].quantile(0.95)
        tu = upper_percentile - lower_percentile
        return round(tu, 2)

    def calculate_eu(self, column: str, tu: float) -> Tuple[float, float]:
        """Calculate Emission Scenario Uncertainty (EU) and its partition."""
        scenario_medians = self.data.groupby("Scenario")[column].median()
        eu = abs(scenario_medians["RCP8.5"] - scenario_medians["RCP4.5"])
        partition = eu / tu
        return round(eu, 2), round(partition, 2)

    def calculate_cmu(self, column: str, tu: float) -> Tuple[float, float]:
        """
        Calculate Climate Model Uncertainty (CMU) and its partition.

        Args:
            column (str): The column name.
            tu (float): Total Uncertainty value.

        Returns:
            Tuple[float, float]: Climate Model Uncertainty and its partition."""

        def compute_cmu_for_scenario(scenario):
            model_means = (
                self.data[self.data["Scenario"] == scenario]
                .groupby("Climate_Model")[column]
                .mean()
            )
            return model_means.max() - model_means.min()

        cmu_rcp45 = compute_cmu_for_scenario("RCP4.5")
        cmu_rcp85 = compute_cmu_for_scenario("RCP8.5")

        cmu = (cmu_rcp45 + cmu_rcp85) / 2
        partition = cmu / tu
        return round(cmu, 2), round(partition, 2)

    def calculate_su(self, column: str, tu: float) -> Tuple[float, float]:
        """
        Calculate Stochastic Uncertainty (SU) and its partition.

        Args:
            column (str): The column name.
            tu (float): Total Uncertainty value.

        Returns:
            Tuple[float, float]: Stochastic Uncertainty and its partition.
        """

        def compute_su_for_scenario(scenario):
            # Find the median climate model for the scenario
            model_medians = (
                self.data[self.data["Scenario"] == scenario]
                .groupby("Climate_Model")[column]
                .median()
            )
            overall_median = model_medians.median()
            median_model = (model_medians - overall_median).abs().idxmin()

            # Compute 5-95th percentile range for the median model
            median_model_data = self.data[
                (self.data["Scenario"] == scenario)
                & (self.data["Climate_Model"] == median_model)
            ][column]
            return median_model_data.quantile(0.95) - median_model_data.quantile(0.05)

        su_rcp45 = compute_su_for_scenario("RCP4.5")
        su_rcp85 = compute_su_for_scenario("RCP8.5")

        su = (su_rcp45 + su_rcp85) / 2
        partition = su / tu
        return round(su, 2), round(partition, 2)
