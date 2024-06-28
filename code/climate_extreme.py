import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme, ks_2samp
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
import seaborn as sns
from typing import Tuple, Optional, List, Union, Dict


# TODO: Work with copy of data instead of modifying in place
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
    - ddf_results (Dict[int, Dict[int, float]]): Depth-Duration-Frequency data.

    Methods:
    - fit_genextreme: Fit a Generalized Extreme Value distribution.
    - plot_fit_and_ci: Plot the fitted distribution with confidence intervals.
    - truncated_ks_test: Perform a Kolmogorov-Smirnov test on extreme values.
    - plot_extreme_comparison: Plot a comparison of extreme value distributions.
    - compute_ddf: Compute Depth-Duration-Frequency data.
    - fit_ddf_exponential: Fit an exponential function to the DDF curve.
    - plot_ddf: Plot the Depth-Duration-Frequency curve.
    """

    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)
    fit_results: dict = field(init=False, default_factory=dict)
    ddf_results: Dict[int, Dict[int, float]] = field(init=False, default_factory=dict)

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
        - pd.DataFrame: A DataFrame containing the fitted parameters and confidence intervals.

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
        - Tuple[float, float]: The KS statistic and p-value.

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
        sns.despine()
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()

    def compute_ddf(
        self, column: str, durations: List[int], return_periods: List[int]
    ) -> None:
        """
        Compute the Depth-Duration-Frequency (DDF) data.

        Parameters:
        - column (str): The precipitation column to analyze.
        - durations (List[int]): List of durations (in days) to analyze.
        - return_periods (List[int]): List of return periods (in years) to compute.

        Raises:
        - ValueError: If the column is not found in the data.
        """
        # Create a datetime index
        df = self.data.copy()

        # Convert month names to numbers
        df["Month"] = pd.to_datetime(df["Month"], format="%b").dt.month

        # Create the datetime index
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df.set_index("Date", inplace=True)

        ams = {}
        for duration in durations:
            # Calculate rolling sum
            rolling_sum = df[column].rolling(window=duration).sum()

            # Resample to yearly maximum
            yearly_max = rolling_sum.groupby(rolling_sum.index.year).max()

            ams[duration] = yearly_max

        params = {
            duration: genextreme.fit(ams[duration].dropna()) for duration in durations
        }

        self.ddf_results = {
            duration: {
                rp: genextreme.ppf(1 - 1 / rp, *params[duration])
                for rp in return_periods
            }
            for duration in durations
        }

    def fit_ddf_sherman(self, return_period: int) -> Tuple[float, float, float]:
        """
        Fit the Sherman equation to the DDF curve for a specific return period.

        The Sherman equation is of the form: I = a / (t + b)^n
        Where I is intensity, t is duration, and a, b, n are parameters to be fitted.

        Parameters:
        - return_period (int): The return period to fit the Sherman equation for.

        Returns:
        - Tuple[float, float, float]: The fitted parameters (a, b, n) for the Sherman equation

        Raises:
        - ValueError: If the DDF data has not been computed.
        """
        if not self.ddf_results:
            raise ValueError(
                "DDF data has not been computed. Call compute_ddf() first."
            )

        durations = np.array(list(self.ddf_results.keys()))
        depths = np.array(
            [self.ddf_results[duration][return_period] for duration in durations]
        )

        # Convert depths to intensities
        intensities = depths / durations

        def sherman_eq(t, a, b, n):
            return a / (t + b) ** n

        # Initial guess for parameters
        p0 = [np.max(intensities) * np.min(durations), 1, 0.5]

        try:
            popt, _ = curve_fit(sherman_eq, durations, intensities, p0=p0, maxfev=10000)
            return tuple(popt)
        except RuntimeError:
            print(
                f"Warning: Sherman equation fitting failed for {return_period}-year return period. Using simple power law instead."
            )
            # Fall back to simple power law: I = a * t^(-n)
            a, n = np.polyfit(np.log(durations), np.log(intensities), 1)
            return (np.exp(n), 0, -a)  # Return in the form of (a, b, n) where b is 0

    def plot_ddf(
        self,
        fit_return_periods: Optional[Union[int, List[int]]] = None,
        output_destination: Optional[str] = None,
    ) -> None:
        """
        Plot the Depth-Duration-Frequency (DDF) curve, optionally with Sherman equation fits for specified return periods.

        Parameters:
        - fit_return_periods (Optional[Union[int, List[int]]]): If provided, plot the Sherman equation fit for these return period(s).
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.

        Raises:
        - ValueError: If the DDF data has not been computed.
        """
        if not self.ddf_results:
            raise ValueError(
                "DDF data has not been computed. Call compute_ddf() first."
            )

        durations = list(self.ddf_results.keys())
        return_periods = list(self.ddf_results[durations[0]].keys())

        plt.figure(figsize=(12, 8))

        # Plot DDF curves
        for rp in return_periods:
            depths = [self.ddf_results[duration][rp] for duration in durations]
            plt.plot(durations, depths, marker="o", label=f"{rp}-year return period")

        # Plot fits if specified
        if fit_return_periods is not None:
            if isinstance(fit_return_periods, int):
                fit_return_periods = [fit_return_periods]

            for rp in fit_return_periods:
                if rp in return_periods:
                    a, b, n = self.fit_ddf_sherman(rp)
                    x_fit = np.linspace(min(durations), max(durations), 100)
                    y_fit = a * x_fit / (x_fit + b) ** n
                    plt.plot(
                        x_fit,
                        y_fit,
                        "--",
                        label=f"Sherman Fit ({rp}-year): {a:.2f} * t / (t + {b:.2f})^{n:.2f}",
                    )
                else:
                    print(
                        f"Warning: {rp}-year return period not in computed data. Skipping fit."
                    )

        plt.xlabel("Duration (days)")
        plt.ylabel("Precipitation Depth (mm)")
        plt.title("Depth-Duration-Frequency Curve")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(linestyle="-", alpha=0.2, color="black")
        sns.despine()
        # plt.xscale("log")
        # plt.yscale("log")
        plt.tight_layout()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()
