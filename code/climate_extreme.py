import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from dataclasses import dataclass, field
import seaborn as sns


@dataclass
class ClimateExtreme:
    """
    A class to calculate climate statistics.

    Available distributions:
    - genextreme
    """

    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)
    fit_results: dict = field(init=False, default_factory=dict)

    def fit_genextreme(
        self, column: str, quantile: float
    ) -> tuple[float, float, float]:
        """
        Fit a Generalized Extreme Value distribution to the data.

        Parameters:
        - column (str): The column to fit the distribution to.
        - quantile (float): The quantile to use for the threshold.

        Returns:
        - tuple[float, float, float]: The parameters of the fitted distribution: c, loc, scale.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        quantile_value = self.data[column].quantile(quantile)
        self.extreme = self.data[self.data[column] > quantile_value][column].values

        c, loc, scale = genextreme.fit(self.extreme)

        self.fit_results[column] = {
            "distribution": "genextreme",
            "parameters": {"c": c, "loc": loc, "scale": scale},
        }
        return c, loc, scale

    def plot_hist_vs_genextreme(
        self, column: str, units: str, c: float, loc: float, scale: float, output_destination: str = None
    ) -> None:
        """
        Plot the histogram of the data against the Generalized Extreme Value distribution.

        Parameters:
        - column (str): The column to plot.
        - units (str): The units of the data.
        - c (float): The shape parameter of the distribution.
        - loc (float): The location parameter of the distribution.
        - scale (float): The scale parameter of the distribution.
        - output_destination (str): File path to save the figure. If None, the plot will be displayed instead of saving.
        """
        if self.extreme is None:
            raise ValueError(
                "You must fit the Generalized Extreme Value distribution first."
            )

        bins = int(np.sqrt(len(self.extreme)))

        plt.hist(
            self.extreme,
            bins=bins,
            density=True,
            alpha=0.6,
            color="#189AB4",
            label=f"Data: {column}",
        )

        x = np.linspace(self.extreme.min(), self.extreme.max(), 1000)
        y = genextreme.pdf(x, c, loc, scale)
        plt.plot(x, y, "r--", label="Fitted GEV")

        plt.title(f"{column} vs Generalized Extreme Value")
        sns.despine()
        plt.grid(linestyle="-", alpha=0.2, color="black")
        plt.xlabel(column + f" ({units})")
        plt.ylabel("Density")
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)

        plt.show()

    @staticmethod
    def compare_parameters(
        column: str,
        distribution: str,
        stats1: "ClimateExtreme",
        stats2: "ClimateExtreme",
        name1: str,
        name2: str,
    ) -> pd.DataFrame:
        """
        Compare the parameters of two ClimateExtreme objects for a specific column and distribution.

        Parameters:
        - column (str): The column to compare.
        - distribution (str): The distribution type to compare.
        - stats1 (ClimateExtreme): The first ClimateExtreme object.
        - stats2 (ClimateExtreme): The second ClimateExtreme object.
        - name1 (str): The label for the first set of results.
        - name2 (str): The label for the second set of results.

        Returns:
        - pd.DataFrame: A DataFrame with the parameters of both objects.
        """
        param1 = (
            stats1.fit_results[column]["parameters"]
            if stats1.fit_results[column]["distribution"] == distribution
            else None
        )
        param2 = (
            stats2.fit_results[column]["parameters"]
            if stats2.fit_results[column]["distribution"] == distribution
            else None
        )

        if param1 is None or param2 is None:
            raise ValueError(
                "The specified distribution was not found in one or both instances."
            )

        df1 = pd.DataFrame({name1: param1})
        df2 = pd.DataFrame({name2: param2})

        result = pd.concat([df1, df2], axis=1)

        result["percent_change"] = (result[name2] - result[name1]) / result[name1] * 100

        return result
