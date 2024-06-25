import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme, ks_2samp
from dataclasses import dataclass, field
import seaborn as sns
from typing import Tuple, Optional, List


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
    - qq_plot: Create a Quantile-Quantile plot for extreme values.
    - block_maxima: Apply the Block Maxima method.
    - compare_block_maxima: Compare and plot block maxima between datasets.
    """

    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)
    fit_results: dict = field(init=False, default_factory=dict)

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
        self, column: str, other: "ClimateExtreme", quantile: float = 0.95
    ) -> Tuple[float, float]:
        """
        Perform a truncated Kolmogorov-Smirnov test on the extreme values of two datasets.

        Parameters:
        - column (str): The column to compare.
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - quantile (float): The quantile threshold for defining extreme values.

        Returns:
        - Tuple[float, float]: The KS statistic and p-value.
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
        """
        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        plt.figure(figsize=(10, 6))
        sns.kdeplot(extremes_self, label="Generated Data", shade=True)
        sns.kdeplot(extremes_other, label="Observed Data", shade=True)

        plt.title(f"Comparison of Extreme Values ({quantile:.2%} quantile)")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()

    def qq_plot(
        self,
        column: str,
        other: "ClimateExtreme",
        quantile: float = 0.95,
        output_destination: Optional[str] = None,
    ) -> None:
        """
        Create a Quantile-Quantile plot comparing the extreme values of two datasets.

        Parameters:
        - column (str): The column to compare.
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - quantile (float): The quantile threshold for defining extreme values.
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.
        """
        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        # Ensure the arrays have the same length
        min_length = min(len(extremes_self), len(extremes_other))
        extremes_self = np.sort(extremes_self)[:min_length]
        extremes_other = np.sort(extremes_other)[:min_length]

        plt.figure(figsize=(8, 8))
        plt.scatter(extremes_other, extremes_self, alpha=0.5)
        plt.plot(
            [extremes_other.min(), extremes_other.max()],
            [extremes_other.min(), extremes_other.max()],
            "r--",
        )

        plt.xlabel("Observed Data Quantiles")
        plt.ylabel("Generated Data Quantiles")
        plt.title(f"Q-Q Plot of Extreme Values (>{quantile:.2%} quantile)")

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()

    # TODO: I need to deal with the DateTime index and somehow figure out how to deal with the many simulations
    def block_maxima(self, column: str, freq: str = "Y") -> pd.Series:
        """
        Apply the Block Maxima method to the data.

        Parameters:
        - column (str): The column to analyze.
        - freq (str): The frequency for defining blocks. Default is 'Y' for yearly.
                      Use 'M' for monthly, 'Q' for quarterly, etc.

        Returns:
        - pd.Series: A series of block maxima.
        """
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError(
                "The DataFrame index must be a DatetimeIndex for block maxima analysis."
            )

        return self.data.resample(freq)[column].max()

    def compare_block_maxima(
        self,
        column: str,
        other: "ClimateExtreme",
        freq: str = "Y",
        output_destination: Optional[str] = None,
    ) -> None:
        """
        Compare block maxima between two datasets and create a plot.

        Parameters:
        - column (str): The column to compare.
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - freq (str): The frequency for defining blocks.
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.
        """
        bm_self = self.block_maxima(column, freq)
        bm_other = other.block_maxima(column, freq)

        plt.figure(figsize=(12, 6))
        plt.plot(bm_self.index, bm_self.values, label="Generated Data", marker="o")
        plt.plot(bm_other.index, bm_other.values, label="Observed Data", marker="s")

        plt.xlabel("Time")
        plt.ylabel(f"Block Maxima ({column})")
        plt.title(f"Comparison of Block Maxima ({freq} frequency)")
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()
