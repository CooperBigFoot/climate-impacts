import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme, ks_2samp
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
import seaborn as sns
from typing import Tuple, Optional, List, Union, Dict


@dataclass
class ClimateExtreme:
    """
    A class to analyze and compare extreme values in climate data.

    This class provides methods for fitting extreme value distributions,
    performing statistical tests, and visualizing comparisons between datasets.

    Attributes:
    - data (pd.DataFrame): The input climate data.
    - extreme (np.ndarray): Extracted extreme values.
    - fit_results (dict): Results of distribution fitting.

    Methods:
    - fit_genextreme: Fit a Generalized Extreme Value distribution.
    - plot_fit_and_ci: Plot the fitted distribution with confidence intervals.
    - truncated_ks_test: Perform a Kolmogorov-Smirnov test on extreme values.
    - plot_extreme_comparison: Plot a comparison of extreme value distributions.
    """

    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)
    fit_results: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Create a deep copy of the data to prevent modifying the original
        self.data = self.data.copy()

    def fit_genextreme(
        self, column: str, quantile: float, n_bootstrap: int = 1000, ci: float = 0.95
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Fit a Generalized Extreme Value distribution to the data and compute confidence intervals.

        Parameters:
        - column (str): The column to fit the distribution to.
        - quantile (float): The quantile to use for the threshold.
        - n_bootstrap (int): Number of bootstrap samples for CI estimation.
        - ci (float): Confidence interval level.

        Returns:
        - Tuple[float, float, float, np.ndarray, np.ndarray]:
          The parameters of the fitted distribution (c, loc, scale) and their confidence intervals.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        quantile_value = self.data[column].quantile(quantile)
        self.extreme = self.data[self.data[column] > quantile_value][column].values

        c, loc, scale = genextreme.fit(self.extreme)

        # Bootstrap for confidence intervals
        bootstrap_params = np.array(
            [
                genextreme.fit(
                    np.random.choice(self.extreme, size=len(self.extreme), replace=True)
                )
                for _ in range(n_bootstrap)
            ]
        )

        ci_lower = np.percentile(bootstrap_params, (1 - ci) / 2 * 100, axis=0)
        ci_upper = np.percentile(bootstrap_params, (1 + ci) / 2 * 100, axis=0)

        self.fit_results[column] = {
            "distribution": "genextreme",
            "parameters": {"c": c, "loc": loc, "scale": scale},
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

        return c, loc, scale, ci_lower, ci_upper

    def plot_fit_and_ci(
        self, column: str, units: str, output_destination: Optional[str] = None
    ) -> None:
        """
        Plot the histogram of the data against the fitted Generalized Extreme Value distribution with confidence intervals.

        Parameters:
        - column (str): The column to plot.
        - units (str): The units of the data.
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed instead of saving.

        Raises:
        - ValueError: If the Generalized Extreme Value distribution has not been fitted yet.
        """
        if self.extreme is None or column not in self.fit_results:
            raise ValueError(
                "You must fit the Generalized Extreme Value distribution first."
            )

        fit_params = self.fit_results[column]["parameters"]
        ci_lower = self.fit_results[column]["ci_lower"]
        ci_upper = self.fit_results[column]["ci_upper"]

        bins = int(np.sqrt(len(self.extreme)))

        plt.figure(figsize=(10, 6))
        plt.hist(
            self.extreme,
            bins=bins,
            density=True,
            alpha=0.6,
            color="#189AB4",
            label=f"Data: {column}",
        )

        x = np.linspace(self.extreme.min(), self.extreme.max(), 1000)
        y = genextreme.pdf(x, **fit_params)
        plt.plot(x, y, "r-", label="Fitted GEV")

        # Plot confidence intervals
        y_lower = genextreme.pdf(x, *ci_lower)
        y_upper = genextreme.pdf(x, *ci_upper)
        plt.fill_between(x, y_lower, y_upper, color="r", alpha=0.2, label="95% CI")

        plt.title(f"{column} vs Generalized Extreme Value with 95% CI")
        sns.despine()
        plt.grid(linestyle="-", alpha=0.2, color="black")
        plt.xlabel(column + f" ({units})")
        plt.ylabel("Density")
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()

    def compare_ci(
        self,
        column: str,
        other: "ClimateExtreme",
        self_name: str = "Self",
        other_name: str = "Other",
    ) -> pd.DataFrame:
        """
        Compare the confidence intervals of this ClimateExtreme object with another for a specific column.

        Parameters:
        - column (str): The column to compare.
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - self_name (str): Name to use for this object in the output DataFrame.
        - other_name (str): Name to use for the other object in the output DataFrame.

        Returns:
        - pd.DataFrame: A DataFrame containing the fitted parameters and confidence intervals for the two objects.

        Raises:
        - ValueError: If the column is not found in the fit_results of either object.
        """
        result = pd.DataFrame(index=["c", "loc", "scale"])

        for name, stats in [(self_name, self), (other_name, other)]:
            if column not in stats.fit_results:
                print(f"Column {column} not found in fit_results for {name}")
                continue

            fit_data = stats.fit_results[column]
            params = fit_data["parameters"]
            ci_lower = fit_data["ci_lower"]
            ci_upper = fit_data["ci_upper"]

            result[f"{name}_value"] = pd.Series({k: v for k, v in params.items()})
            result[f"{name}_CI_lower"] = pd.Series(
                dict(zip(["c", "loc", "scale"], ci_lower))
            )
            result[f"{name}_CI_upper"] = pd.Series(
                dict(zip(["c", "loc", "scale"], ci_upper))
            )

        return result

    def truncated_ks_test(
        self, column: str, other: "ClimateExtreme", quantile: float = 0.99
    ) -> Tuple[float, float]:
        """
        Perform a truncated Kolmogorov-Smirnov test on the extreme values of two datasets.

        Parameters:
        - column (str): The column to compare.
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - quantile (float): The quantile threshold for defining extreme values.

        Returns:
        - Tuple[float, float]: The KS statistic (a measure of difference) and the p-value.

        Raises:
        - ValueError: If the column is not found in one or both datasets.
        """
        if column not in self.data.columns or column not in other.data.columns:
            raise ValueError(f"Column '{column}' not found in one or both datasets.")

        # Define thresholds
        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        # Extract extreme values
        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        # Perform KS test
        ks_statistic, p_value = ks_2samp(extremes_self, extremes_other)

        return ks_statistic, p_value

    def plot_extreme_comparison(
        self,
        column: str,
        units: str,
        other: "ClimateExtreme",
        quantile: float = 0.95,
        output_destination: Optional[str] = None,
    ) -> None:
        """
        Plot the extreme value distributions of two datasets for comparison.

        Parameters:
        - column (str): The column to compare.
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - quantile (float): The quantile threshold for defining extreme values.
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.

        Raises:
        - ValueError: If the column is not found in one or both datasets.
        """
        sns.set_context("paper", font_scale=1.4)
        
        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        plt.figure(figsize=(10, 6))
        sns.kdeplot(extremes_self, label="Generated Data", shade=True)
        sns.kdeplot(extremes_other, label="Observed Data", shade=True)

        plt.title(f"Comparison of Extreme Values ({quantile:.2%} quantile)")
        plt.xlabel(column + f" ({units})")
        plt.ylabel("Density")
        sns.despine()
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()
