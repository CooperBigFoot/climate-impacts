import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from dataclasses import dataclass, field


@dataclass
class ClimateStats:
    """
    A class to calculate climate statistics.
    """

    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)

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

        if not np.isfinite(self.extreme).all():
            raise ValueError("Data contains non-finite values.")

        c, loc, scale = genextreme.fit(self.extreme)
        return c, loc, scale

    def plot_hist_vs_genextreme(
        self, column: str, c: float, loc: float, scale: float
    ) -> None:
        """
        Plot the histogram of the data against the Generalized Extreme Value distribution.

        Parameters:
        - column (str): The column to plot.
        - c (float): The shape parameter of the distribution.
        - loc (float): The location parameter of the distribution.
        - scale (float): The scale parameter of the distribution.
        """
        if self.extreme is None:
            raise ValueError(
                "You must fit the Generalized Extreme Value distribution first."
            )

        bins = int(np.sqrt(len(self.extreme)))

        plt.hist(self.extreme, bins=bins, density=True, alpha=0.6, color="g")

        x = np.linspace(self.extreme.min(), self.extreme.max(), 1000)
        y = genextreme.pdf(x, c, loc, scale)
        plt.plot(x, y, "r--", label="Fitted GEV")

        plt.title(f"{column} vs Generalized Extreme Value")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()
        plt.show()
