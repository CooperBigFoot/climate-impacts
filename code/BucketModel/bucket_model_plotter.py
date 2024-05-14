import seaborn as sns  # For styling the plots
import matplotlib.pyplot as plt  # For plotting
import scipy.stats as stats  # For the confidence interval
import numpy as np  # For the density plot

import pandas as pd  # For the data handling


def plot_water_balance(
    results: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    palette: list = ["#004E64", "#007A9A", "#00A5CF", "#9FFFCB", "#25A18E"],
    start_year: str = "1986",
    end_year: str = "2000",
    figsize: tuple[int, int] = (10, 6),
    fontsize: int = 12,
) -> None:
    """Plot the water balance of the model.

    Parameters:
    - results (pd.DataFrame): The results from the model run.
    - title (str): The title of the plot, if empty, no title will be shown.
    - output_destination (str): The path to the output file, if empty, the plot will not be saved.
    - palette (list): The color palette to use for the plot, default is ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB', '#25A18E'].
    - start_year (str): The start year of the plot, default is '1986'.
    - end_year (str): The end year of the plot, default is '2000'.
    - figsize (tuple): The size of the figure, default is (10, 6).
    - fontsize (int): The fontsize of the plot, default is 12.
    """
    # Some style settings
    BAR_WIDTH = 0.35
    sns.set_context("paper")
    sns.set_style("white")

    # Helper function to plot a single bar chart layer
    def plot_bar_layer(
        ax: plt.Axes,
        positions: np.ndarray,
        heights: np.ndarray,
        label: str,
        color: str,
        bottom_layer_heights: np.ndarray = None,
    ) -> None:
        """Plot a single layer of a bar chart.

        Parameters:
        - ax (plt.Axes): The axis to plot on.
        - positions (np.ndarray): The x-positions of the bars.
        - heights (np.ndarray): The heights of the bars.
        - label (str): The label of the layer.
        - color (str): The color of the layer.
        - bottom_layer_heights (np.ndarray): The heights of the bottom layer, default is None.
        """
        ax.bar(
            positions,
            heights,
            width=BAR_WIDTH,
            label=label,
            color=color,
            bottom=bottom_layer_heights,
        )

    # Prepare the data
    results_filtered = results.copy()
    results_filtered["Year"] = results_filtered.index.year
    results_filtered = results_filtered[start_year:end_year]
    yearly_totals = results_filtered.groupby("Year").sum()

    years = yearly_totals.index

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each component of the water balance
    plot_bar_layer(ax, years - BAR_WIDTH / 2, yearly_totals["Rain"], "Rain", palette[0])
    plot_bar_layer(
        ax,
        years - BAR_WIDTH / 2,
        yearly_totals["Snow"],
        "Snow",
        palette[1],
        bottom_layer_heights=yearly_totals["Rain"],
    )
    plot_bar_layer(
        ax, years + BAR_WIDTH / 2, yearly_totals["Q_s"], "Q$_{surface}$", palette[2]
    )
    plot_bar_layer(
        ax,
        years + BAR_WIDTH / 2,
        yearly_totals["Q_gw"],
        "Q$_{gw}$",
        palette[3],
        bottom_layer_heights=yearly_totals["Q_s"],
    )
    plot_bar_layer(
        ax,
        years + BAR_WIDTH / 2,
        yearly_totals["ET"],
        "ET",
        palette[4],
        bottom_layer_heights=yearly_totals["Q_s"] + yearly_totals["Q_gw"],
    )

    ax.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax.set_ylabel("Water depth [mm]", fontsize=fontsize)
    ax.legend(fontsize=fontsize, ncol=3, loc="best")
    plt.tight_layout()
    sns.despine()

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def plot_ECDF(
    results: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    color: str = "#007A9A",
    figsize: tuple[int, int] = (6, 6),
    fontsize: int = 12,
) -> None:
    """Plot the empirical cumulative distribution function (ECDF) of the simulated total runoff (Q) values.

    Parameters:
    - results (pd.DataFrame): The results from the model run.
    - title (str): The title of the plot, if empty, no title will be shown.
    - output_destination (str): The path to the output file, if empty, the plot will not be saved.
    - color (str): The color of the plot, default is '#007A9A'.
    - figsize (tuple): The size of the figure, default is (6, 6).
    - fontsize (int): The fontsize of the plot, default is 12.
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = results.copy()
    results_filtered["Total_Runoff"] = (
        results_filtered["Q_s"] + results_filtered["Q_gw"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the ECDF of the simulated total runoff
    sns.ecdfplot(
        data=results_filtered["Total_Runoff"],
        ax=ax,
        color=color,
        label="Simulated total runoff",
    )

    ax.set_xlabel("Total runoff [mm/d]", fontsize=fontsize)
    ax.set_ylabel("F cumulative", fontsize=fontsize)
    ax.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc="best")
    plt.tight_layout()
    sns.despine()
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def plot_KDE(
    results: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    color: str = "#007A9A",
    figsize: tuple[int, int] = (6, 6),
    fontsize: int = 12,
    fill: bool = True,
) -> None:
    """Plot the kernel density estimate (KDE) of the simulated total runoff (Q) values.

    Parameters:
    - results (pd.DataFrame): The results from the model run.
    - title (str): The title of the plot, if empty, no title will be shown.
    - output_destination (str): The path to the output file, if empty, the plot will not be saved.
    - color (str): The color of the plot, default is '#007A9A'.
    - figsize (tuple): The size of the figure, default is (6, 6).
    - fontsize (int): The fontsize of the plot, default is 12.
    - fill (bool): If True, the KDE will be filled, default is True.
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = results.copy()
    results_filtered["Total_Runoff"] = (
        results_filtered["Q_s"] + results_filtered["Q_gw"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the KDE of the simulated total runoff
    sns.kdeplot(
        data=results_filtered["Total_Runoff"],
        ax=ax,
        color=color,
        label="Simulated total runoff",
        fill=fill,
    )

    ax.set_xlabel("Total runoff [mm/d]", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc="best")
    plt.tight_layout()
    sns.despine()
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def plot_monthly_boxplot(
    results: pd.DataFrame,
    title: str = "",
    output_destination: str = "",
    figsize: tuple[int, int] = (12, 12),
    fontsize: int = 12,
    palette: list = ["#004E64", "#007A9A", "#00A5CF", "#9FFFCB"],
) -> None:
    """Plot the monthly boxplot of the simulated environmental variables.

    Variables:
    - Monthly Precipitation
    - Actual Monthly Evapotranspiration
    - Monthly Snowmelt
    - Monthly simulated Total Runoff

    Parameters:
    - results (pd.DataFrame): The results from the model run, make sure you have the following columns: 'Precip', 'ET', 'Snow_melt', 'Q_s', 'Q_gw'.
    - title (str): The title of the plot, if empty, no title will be shown.
    - output_destination (str): The path to the output file, if empty, the plot will not be saved.
    - figsize (tuple): The size of the figure, default is (12, 12).
    - fontsize (int): The fontsize of the plot, default is 12.
    - palette (list): The color palette to use for the plot, default is ['#004E64', '#007A9A', '#00A5CF', '#9FFFCB'].
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = results.copy()
    results_filtered["Total_Runoff"] = (
        results_filtered["Q_s"] + results_filtered["Q_gw"]
    )
    results_filtered["Month"] = results_filtered.index.month
    results_filtered["Year"] = results_filtered.index.year

    monthly_sums = results_filtered.groupby(["Year", "Month"]).sum().reset_index()

    months = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    monthly_sums["Month"] = monthly_sums["Month"].map(months)

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)

    ax_precip = plt.subplot2grid(layout, (0, 0))
    ax_et = plt.subplot2grid(layout, (0, 1))
    ax_snow_melt = plt.subplot2grid(layout, (1, 0))
    ax_runoff = plt.subplot2grid(layout, (1, 1))

    sns.boxplot(
        x="Month", y="Precip", data=monthly_sums, ax=ax_precip, color=palette[0]
    )
    sns.boxplot(x="Month", y="ET", data=monthly_sums, ax=ax_et, color=palette[1])
    sns.boxplot(
        x="Month", y="Snow_melt", data=monthly_sums, ax=ax_snow_melt, color=palette[2]
    )
    sns.boxplot(
        x="Month", y="Total_Runoff", data=monthly_sums, ax=ax_runoff, color=palette[3]
    )

    ax_precip.set_xlabel("")
    ax_precip.set_ylabel("Precipitation [mm/d]", fontsize=fontsize)
    ax_precip.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_precip.set_title("Monthly Precipitation", fontsize=fontsize)
    ax_precip.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_et.set_xlabel("")
    ax_et.set_ylabel("Actual ET [mm/d]", fontsize=fontsize)
    ax_et.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_et.set_title("Monthly Actual ET", fontsize=fontsize)
    ax_et.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_snow_melt.set_xlabel("")
    ax_snow_melt.set_ylabel("Snowmelt [mm/d]", fontsize=fontsize)
    ax_snow_melt.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_snow_melt.set_title("Monthly Snowmelt", fontsize=fontsize)
    ax_snow_melt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax_runoff.set_xlabel("")
    ax_runoff.set_ylabel("Total Runoff [mm/d]", fontsize=fontsize)
    ax_runoff.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax_runoff.set_title("Monthly Total Runoff", fontsize=fontsize)
    ax_runoff.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    sns.despine()

    if title:
        plt.suptitle(title, fontsize=fontsize)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def plot_timeseries(
    results: pd.DataFrame,
    start_year: str,
    end_year: str,
    monthly: bool = False,
    title: str = "",
    output_destination: str = "",
    figsize: tuple[int, int] = (10, 6),
    fontsize: int = 12,
    palette: list = ["#007A9A", "#25A18E"],
) -> None:
    """Plot the timeseries of the simulated total runoff (Q) values.

    Parameters:
    - results (pd.DataFrame): The results from the model run.
    - start_year (str): The start date of the plot.
    - end_year (str): The end date of the plot, inclusive.
    - monthly (bool): If True, the plot will be monthly, default is False (daily).
    - title (str): The title of the plot, if empty, no title will be shown.
    - output_destination (str): The path to the output file, if empty, the plot will not be saved.
    - figsize (tuple): The size of the figure, default is (10, 6).
    - fontsize (int): The fontsize of the plot, default is 12.
    - palette (list): The color palette to use for the plot, default is ['#007A9A', '#25A18E'].
    """
    sns.set_context("paper")

    # Prepare the data
    results_filtered = results.copy()
    results_filtered = results_filtered[start_year:end_year]
    results_filtered["Total_Runoff"] = (
        results_filtered["Q_s"] + results_filtered["Q_gw"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    if monthly:
        results_filtered = results_filtered.resample("ME").sum()

    sns.lineplot(
        data=results_filtered["Total_Runoff"],
        ax=ax,
        color=palette[0],
        label="Simulated total runoff",
        alpha=0.7,
    )

    ax.set_xlabel("")

    if monthly:
        ax.set_ylabel("Total runoff [mm/month]", fontsize=fontsize)
    else:
        ax.set_ylabel("Total runoff [mm/d]", fontsize=fontsize)

    ax.tick_params(which="both", length=10, width=2, labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc="best")
    plt.tight_layout()
    sns.despine()
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    if title:
        plt.title(title)

    if output_destination:
        fig.savefig(output_destination, dpi=300, bbox_inches="tight")


def group_by_month_with_ci(
    results_df: pd.DataFrame, n_simulations: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the mean and 95% confidence interval of the monthly data.

    Parameters:
    - results_df (pd.DataFrame): The results from running the BucketModel for multiple simulations.
    - n_simulations (int): The number of simulations in the generated ensemble, default is 50.

    Returns:
    - monthly_mean (pd.DataFrame): The mean of the monthly data.
    - ci (pd.DataFrame): The 95% confidence interval of the monthly data.
    """

    results_df = results_df.copy()
    results_df["month"] = results_df.index.month
    results_df["year"] = results_df.index.year

    monthly_data = (
        results_df.groupby(["Simulation", "year", "month"]).sum().reset_index()
    )

    monthly_mean = monthly_data.groupby("month").mean()

    monthly_std = monthly_data.groupby("month").std()

    ci = stats.t.ppf(0.975, n_simulations - 1) * (monthly_std / np.sqrt(n_simulations))

    return monthly_mean, ci


def plot_monthly_runoff_with_ci(
    results_monthly: pd.DataFrame, ci: pd.DataFrame, output_destination: str = ""
):
    """
    Plots mean monthly total runoff with 95% confidence interval and saves the plot.

    Parameters:
        results_monthly (pd.DataFrame): DataFrame containing the monthly results with mean values.
        ci (pd.DataFrame): DataFrame containing the confidence intervals for each month.
        output_destination (str): File path to save the plot.
    """
    results_monthly["total_runoff"] = results_monthly["Q_s"] + results_monthly["Q_gw"]
    ci_total_runoff = ci["Q_s"] + ci["Q_gw"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        results_monthly.index,
        results_monthly["total_runoff"],
        label="Total Runoff [mm/day]",
        color="b",
    )
    plt.fill_between(
        results_monthly.index,
        results_monthly["total_runoff"] - ci_total_runoff,
        results_monthly["total_runoff"] + ci_total_runoff,
        color="b",
        alpha=0.2,
        label="95% CI",
    )
    plt.xlabel("Month")
    plt.ylabel("Runoff [mm/day]")
    plt.title("Mean Monthly Total Runoff with 95% Confidence Interval")
    plt.legend()
    sns.despine()
    plt.grid(linestyle="-", alpha=0.7)

    if output_destination:
        plt.savefig(output_destination, bbox_inches="tight", dpi=300)
    plt.show()
