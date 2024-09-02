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


from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def generate_color_palette() -> Dict[str, tuple]:
    """
    Generate a color palette for climate models and present climate.

    Returns:
        Dict[str, tuple]: A dictionary mapping model names to color tuples.
    """
    colors = sns.color_palette("husl", n_colors=6)
    return {
        "CLMCOM-CCLM4-ECEARTH": colors[0],
        "CLMCOM-CCLM4-HADGEM": colors[1],
        "DMI-HIRHAM-ECEARTH": colors[2],
        "MPICSC-REMO1-MPIESM": colors[3],
        "SMHI-RCA-IPSL": colors[4],
        "present": colors[5],
    }


def setup_subplot(ax: plt.Axes, title: str, ylabel: str) -> None:
    """
    Set up a subplot with common attributes.

    Args:
        ax (plt.Axes): The subplot axes to set up.
        title (str): The title for the subplot.
        ylabel (str): The label for the y-axis.
    """
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [
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
    )
    ax.grid(linestyle="--", alpha=0.7)


def plot_climate_variable(
    ax: plt.Axes,
    results: Dict[str, Dict[str, pd.DataFrame]],
    present_results: pd.DataFrame,
    variable: str,
    palette: Dict[str, tuple],
) -> None:
    """
    Plot a climate variable for different models and scenarios.

    Args:
        ax (plt.Axes): The subplot axes to plot on.
        results (Dict[str, Dict[str, pd.DataFrame]]): The climate scenario results.
        present_results (pd.DataFrame): The present climate results.
        variable (str): The name of the climate variable to plot.
        palette (Dict[str, tuple]): The color palette for different models.
    """
    for rcp in ["4.5", "8.5"]:
        linestyle = "--" if rcp == "4.5" else "-"
        for model, monthly_mean in results[rcp].items():
            ax.plot(
                monthly_mean[variable],
                linestyle=linestyle,
                color=palette[model],
                alpha=0.7,
            )
    ax.plot(
        present_results[variable],
        color=palette["present"],
        lw=2,
        linestyle="-",
        alpha=0.7,
    )


def create_legend(palette: Dict[str, tuple]) -> List[Line2D]:
    """
    Create legend elements for the plot.

    Args:
        palette (Dict[str, tuple]): The color palette for different models.

    Returns:
        List[Line2D]: A list of Line2D objects representing legend elements.
    """
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="RCP 8.5"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="RCP 4.5"),
        Line2D(
            [0],
            [0],
            color=palette["present"],
            lw=2,
            linestyle="-",
            label="Present climate",
        ),
    ]
    for model, color in palette.items():
        if model != "present":
            legend_elements.append(
                Line2D([0], [0], color=color, lw=2, label=f"Model {model}")
            )
    return legend_elements


def plot_climate_scenarios(
    results: Dict[str, Dict[str, pd.DataFrame]],
    present_results: pd.DataFrame,
    output_destination: str = None,
) -> None:
    """
    Plot the results of climate scenarios for different models and RCP scenarios.

    This function creates a 2x2 grid of subplots, each showing a different climate
    variable (precipitation, evapotranspiration, snowmelt, and runoff) for multiple
    climate models and RCP scenarios, as well as the present climate.

    Args:
        results (Dict[str, Dict[str, pd.DataFrame]]): A nested dictionary containing the results 
            from run_model_for_future_climate. The outer dictionary keys are RCP scenarios 
            ("4.5" and "8.5"), and the inner dictionary keys are model names. Values are pandas 
            DataFrames with monthly data.
        present_results (pd.DataFrame): A pandas DataFrame containing monthly mean results for 
            the present climate.
        output_destination (str, optional): If provided, the path where the plot will be saved. 
            If None, the plot will only be displayed.
    """
    sns.set_context("paper", font_scale=1.5)
    palette = generate_color_palette()

    fig, ((ax_precipitation, ax_evaporation), (ax_snowmelt, ax_total_runoff)) = (
        plt.subplots(2, 2, figsize=(12, 10))
    )

    plot_climate_variable(ax_precipitation, results, present_results, "Precip", palette)
    plot_climate_variable(ax_evaporation, results, present_results, "ET", palette)
    plot_climate_variable(ax_snowmelt, results, present_results, "Snow_melt", palette)
    plot_climate_variable(
        ax_total_runoff, results, present_results, "total_runoff", palette
    )

    setup_subplot(ax_precipitation, "Precipitation", "Mean monthly precipitation [mm]")
    setup_subplot(
        ax_evaporation, "Evapotranspiration", "Mean monthly evapotranspiration [mm]"
    )
    setup_subplot(ax_snowmelt, "Snowmelt", "Mean monthly snowmelt [mm]")
    setup_subplot(ax_total_runoff, "Runoff", "Mean monthly streamflow [mm]")

    legend_elements = create_legend(palette)
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    sns.despine()

    if output_destination:
        plt.savefig(output_destination, dpi=300, bbox_inches="tight")
    else:
        plt.show()
