import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

COLOR_OBSERVED = "#189AB4"
COLOR_SIMULATED = "r"
FIGURE_SIZE = (6, 6)


def add_is_wet_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column to indicate whether a day is wet.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with the added column.
    """

    df["is_wet"] = df["Precipitation"] > 0.1
    return df


def group_data(df: pd.DataFrame, by_columns: list) -> pd.DataFrame:
    """Generic grouping of the dataframe by specified columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The grouped dataframe."""

    return df.groupby(by_columns).sum().reset_index()


def calculate_monthly_stats(
    df: pd.DataFrame, include_simulation: bool = False
) -> pd.DataFrame:
    """Calculate monthly statistics and adjust grouping if simulation data is included.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - include_simulation (bool): Whether to include simulation data in the grouping.

    Returns:
    - pd.DataFrame: The dataframe with monthly statistics.
    """

    group_cols = (
        ["Simulation", "Year", "Month"] if include_simulation else ["Year", "Month"]
    )
    df_stats = (
        df.groupby(group_cols)
        .agg(
            {
                "Precipitation": ["sum", "mean", "std", "max"],
                "T_max": ["mean", "std"],
                "T_min": ["mean", "std"],
                "T_avg": ["mean", "std"],
            }
        )
        .reset_index()
    )
    df_stats.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_stats.columns.values
    ]
    return df_stats


def plot_wet_days(
    observations: pd.DataFrame, simulation: pd.DataFrame, output_destination: str = None
) -> None:
    """Plot the number of wet days from observations and simulation.

    Parameters:
    - observations (pd.DataFrame): The observed data.
    - simulation (pd.DataFrame): The simulated data.
    - output_destination (str): File path to save the figure. If None, the plot will be displayed instead of saving.

    Returns:
    - None
    """

    obs_wet_days = add_is_wet_column(observations)
    sim_wet_days = add_is_wet_column(simulation)
    obs_grouped = group_data(obs_wet_days, ["Year", "Month"])
    sim_grouped = group_data(sim_wet_days, ["Simulation", "Year", "Month"])

    obs_median = obs_grouped.groupby("Month")["is_wet"].median()
    sim_median = sim_grouped.groupby("Month")["is_wet"].median()

    plt.figure(figsize=FIGURE_SIZE)
    sns.boxplot(
        data=obs_grouped, x="Month", y="is_wet", color=COLOR_OBSERVED, showfliers=False
    )
    sns.lineplot(
        data=sim_median, color=COLOR_SIMULATED, marker="o", label="Median Simulated"
    )
    sns.despine()
    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_OBSERVED, lw=5, label="Observed"),
        plt.Line2D([0], [0], color=COLOR_SIMULATED, lw=1.5, label="Median Simulated"),
    ]
    plt.legend(handles=legend_elements)
    plt.ylabel("Number of Wet Days")
    plt.xlabel("")
    if output_destination:
        plt.savefig(output_destination, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_ECDF(
    observations: pd.DataFrame,
    simulation: pd.DataFrame,
    column: str,
    xlabel: str,
    output_destination: str = None,
) -> None:
    """Plot the cumulative distribution function of the observations and simulation.

    Parameters:
    - observations (pd.DataFrame): The observed data.
    - simulation (pd.DataFrame): The simulated data.
    - column (str): The column to plot.
    - xlabel (str): The x-axis label.
    - output_destination (str): File path to save the figure. If None, the plot will be displayed instead of saving.

    Returns:
    - None
    """

    plt.figure(figsize=FIGURE_SIZE)
    sns.ecdfplot(observations[column], label="Observed", color=COLOR_OBSERVED)
    sns.ecdfplot(simulation[column], label="Simulated", color=COLOR_SIMULATED)
    plt.legend()
    plt.ylabel("Cumulative Probability")
    plt.xlabel(xlabel)
    plt.grid(linestyle="-", alpha=0.2, color="black")
    sns.despine()
    if output_destination:
        plt.savefig(output_destination, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_mean_and_std(
    observations: pd.DataFrame, simulation: pd.DataFrame, output_destination: str = None
) -> None:
    """Plot the mean and standard deviation of observations and simulation data for a specified column.

    Parameters:
    - observations (pd.DataFrame): The observed data.
    - simulation (pd.DataFrame): The simulated data.
    - output_destination (str): File path to save the figure. If None, the plot will be displayed instead of saving.

    Returns:
    - None
    """

    obs_stats_monthly = calculate_monthly_stats(observations)
    simul_stats_monthly = calculate_monthly_stats(simulation, include_simulation=True)

    obs_stats_monthly = obs_stats_monthly.groupby("Month_").mean().reset_index()
    simul_stats_monthly = simul_stats_monthly.groupby("Month_").mean().reset_index()

    plt.figure(figsize=FIGURE_SIZE)
    sns.scatterplot(
        x=obs_stats_monthly["Month_"],
        y=obs_stats_monthly["T_avg_mean"],
        label="Observed mean",
        color=COLOR_OBSERVED,
        marker="o",
        s=80,
    )
    sns.scatterplot(
        x=obs_stats_monthly["Month_"],
        y=obs_stats_monthly["T_avg_std"],
        label="Observed std",
        color=COLOR_OBSERVED,
        marker="^",
        s=80,
    )
    sns.lineplot(
        x=simul_stats_monthly["Month_"],
        y=simul_stats_monthly["T_avg_mean"],
        label="Simulated mean",
        color=COLOR_SIMULATED,
        linestyle="-",
    )
    sns.lineplot(
        x=simul_stats_monthly["Month_"],
        y=simul_stats_monthly["T_avg_std"],
        label="Simulated std",
        color=COLOR_SIMULATED,
        linestyle="--",
    )
    plt.legend()
    plt.ylabel("Temperature (°C)")
    plt.xlabel("")
    plt.grid(linestyle="-", alpha=0.2, color="black")
    sns.despine()
    if output_destination:
        plt.savefig(output_destination, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def plot_ddf(data: pd.DataFrame) -> None:
    """Plot the Depth-Duration-Frequency curve.

    Parameters:
    - data (pd.DataFrame): The input dataframe.

    Returns:
    - None
    """

    durations = [1, 3, 7]

    ams = {
        duration: data["P_mix"].rolling(window=duration).sum().resample("Y").max()
        for duration in durations
    }

    def fit_gumbel(data):
        params = stats.gumbel_r.fit(data)
        return params

    # Fit Gumbel distribution to each duration. Explore other distributions too
    params = {duration: fit_gumbel(ams[duration].dropna()) for duration in durations}

    return_periods = [2, 5, 10, 25, 50, 100]
    quantiles = {
        duration: {
            rp: stats.gumbel_r.ppf(1 - 1 / rp, *params[duration])
            for rp in return_periods
        }
        for duration in durations
    }

    plt.figure(figsize=(10, 6))

    for duration in durations:
        plt.plot(
            return_periods,
            [quantiles[duration][rp] for rp in return_periods],
            marker="o",
            label=f"{duration}-day duration",
        )

    plt.xlabel("Return Period (years)")
    plt.ylabel("Precipitation Depth (mm)")
    plt.title("Depth-Duration-Frequency Curve")
    plt.legend()
    plt.grid(linestyle="-", alpha=0.2, color="black")
    sns.despine()
    plt.show()
