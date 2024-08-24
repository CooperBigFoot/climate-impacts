import os
from functools import lru_cache
import pandas as pd
from typing import Dict, List, Tuple

from BucketModel.bucket_model import BucketModel
from BucketModel.bucket_model_plotter import group_by_month_with_ci
from BucketModel.data_processing import (
    preprocess_for_bucket_model,
    run_multiple_simulations,
)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# Constants
MODELS = [
    "CLMCOM-CCLM4-ECEARTH",
    "CLMCOM-CCLM4-HADGEM",
    "DMI-HIRHAM-ECEARTH",
    "MPICSC-REMO1-MPIESM",
    "SMHI-RCA-IPSL",
]


@lru_cache(maxsize=None)
def get_model_from_filename(filename: str) -> str:
    """
    Extract the model name from a filename.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The extracted model name, or None if not found.
    """
    return next((model for model in MODELS if model in filename), None)


def run_model_for_single_scenario(
    file_path: str, rcp: str, bucket_model: BucketModel
) -> Tuple[str, pd.DataFrame]:
    """
    Run the bucket model for a single climate scenario file.

    Args:
        file_path (str): Path to the climate scenario file.
        rcp (str): RCP scenario ('4.5' or '8.5').
        bucket_model (BucketModel): Instance of the BucketModel.

    Returns:
        Tuple[str, pd.DataFrame]: Model name and monthly mean results.
    """
    future_data = pd.read_csv(file_path)
    preprocessed_future_data = preprocess_for_bucket_model(future_data)

    model_results = run_multiple_simulations(
        preprocessed_simulated_data=preprocessed_future_data,
        bucket_model=bucket_model,
        n_simulations=50,
    )

    model_results["total_runoff"] = model_results["Q_s"] + model_results["Q_gw"]

    monthly_mean, _ = group_by_month_with_ci(model_results)

    model_name = get_model_from_filename(os.path.basename(file_path))
    return model_name, monthly_mean


def run_model_for_future_climate(
    future_data_folder: str, bucket_model: BucketModel
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the bucket model for future climate scenarios across different models and RCP scenarios.

    This function processes all climate scenario files in the given folder,
    runs the bucket model simulations, and computes monthly means for each scenario.

    Args:
        future_data_folder (str): Path to the folder containing climate scenario files.
        bucket_model (BucketModel): Instance of the BucketModel.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Results for each RCP scenario and model,
        organized as {RCP: {model_name: monthly_mean_results}}.
    """
    results = {"4.5": {}, "8.5": {}}

    for file in os.listdir(future_data_folder):
        file_path = os.path.join(future_data_folder, file)
        file_lower = file.lower()

        if file_lower.endswith("rcp4.5.csv"):
            model_name, monthly_mean = run_model_for_single_scenario(
                file_path, "4.5", bucket_model
            )
            results["4.5"][model_name] = monthly_mean
        elif file_lower.endswith("rcp8.5.csv"):
            model_name, monthly_mean = run_model_for_single_scenario(
                file_path, "8.5", bucket_model
            )
            results["8.5"][model_name] = monthly_mean

    return results


def plot_climate_scenarios(
    results: Dict[str, Dict[str, pd.DataFrame]], output_destination: str = None
):
    """
    Plot the results of climate scenarios for different models and RCP scenarios.

    Args:
        results (Dict[str, Dict[str, pd.DataFrame]]): Results from run_model_for_future_climate.
        output_destination (str, optional): Path to save the plot. If None, the plot is displayed instead.
    """
    sns.set_context("paper", font_scale=1.5)

    palette = {
        "CLMCOM-CCLM4-ECEARTH": "blue",
        "CLMCOM-CCLM4-HADGEM": "green",
        "DMI-HIRHAM-ECEARTH": "red",
        "MPICSC-REMO1-MPIESM": "purple",
        "SMHI-RCA-IPSL": "orange",
    }

    rcps = ["4.5", "8.5"]

    layout = (2, 2)
    fig = plt.figure(figsize=(12, 10))

    ax_precipitation = plt.subplot2grid(layout, (0, 0))
    ax_evaporation = plt.subplot2grid(layout, (0, 1))
    ax_snowmelt = plt.subplot2grid(layout, (1, 0))
    ax_total_runoff = plt.subplot2grid(layout, (1, 1))

    for rcp in rcps:
        linestyle = "--" if rcp == "4.5" else "-"
        for model, monthly_mean in results[rcp].items():
            ax_precipitation.plot(
                monthly_mean["Precip"],
                linestyle=linestyle,
                color=palette[model],
                alpha=0.7,
            )
            ax_evaporation.plot(
                monthly_mean["ET"], linestyle=linestyle, color=palette[model], alpha=0.7
            )
            ax_snowmelt.plot(
                monthly_mean["Snow_melt"],
                linestyle=linestyle,
                color=palette[model],
                alpha=0.7,
            )
            ax_total_runoff.plot(
                monthly_mean["total_runoff"],
                linestyle=linestyle,
                color=palette[model],
                alpha=0.7,
            )

    ax_precipitation.set_title("Precipitation")
    ax_precipitation.set_ylabel("Mean monthly precipitation [mm]")

    ax_evaporation.set_title("Evapotranspiration")
    ax_evaporation.set_ylabel("Mean monthly evapotranspiration [mm]")

    ax_snowmelt.set_title("Snowmelt")
    ax_snowmelt.set_ylabel("Mean monthly snowmelt [mm]")

    ax_total_runoff.set_title("Runoff")
    ax_total_runoff.set_ylabel("Mean monthly streamflow [mm]")

    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for ax in [ax_precipitation, ax_evaporation, ax_snowmelt, ax_total_runoff]:
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.grid(linestyle="--", alpha=0.7)

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="RCP 8.5"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="RCP 4.5"),
    ]
    for model, color in palette.items():
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=f"Model {model}")
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    sns.despine()

    if output_destination:
        plt.savefig(output_destination, dpi=300, bbox_inches="tight")
    else:
        plt.show()


# TODO: Add possibility to plot the results for the present climate as well
# TODO: Move to uncertainty_analysis.py (?). Make more sense to me tbh
def combine_climate_data(
    folder_path: str, bucket_model: BucketModel, n_simulations: int = 50
) -> pd.DataFrame:
    """
    Combine climate data from multiple CSV files into a single DataFrame.

    This function reads climate data from CSV files in the specified folder, runs multiple simulations,
    and combines the results into a single DataFrame with annual mean streamflow, climate model, and scenario.

    Args:
        folder_path (str): Path to the folder containing the climate data CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with columns for 'Simulation', 'Year', 'Streamflow', 'Climate_Model', and 'Scenario'.
    """
    combined_data = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for filename in csv_files:
        try:
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            future_data = preprocess_for_bucket_model(df)
            future_streamflow = run_multiple_simulations(
                future_data, bucket_model, n_simulations=n_simulations
            )
            future_streamflow["Streamflow"] = (
                future_streamflow["Q_s"] + future_streamflow["Q_gw"]
            )
            future_streamflow["Year"] = future_streamflow.index.year

            # Extract model and scenario from filename
            model, scenario = filename.rsplit("_", 2)[-2:]
            model = model.replace("-", "_")
            scenario = os.path.splitext(scenario)[0]

            annual_totals = (
                future_streamflow.groupby(["Simulation", "Year"])["Streamflow"]
                .sum()
                .reset_index()
            )

            annual_mean = (
                annual_totals.groupby(["Simulation"])["Streamflow"].mean().reset_index()
            )
            annual_mean["Climate_Model"] = model
            annual_mean["Scenario"] = scenario

            combined_data.append(annual_mean)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was processed
