import os
import numpy as np
import pandas as pd

DELTA_CHANGES = {
    "CLMCOM-CCLM4-ECEARTH": {
        "RCP4.5": {"Precipitation": 5.7, "Temperature": 1.9},
        "RCP8.5": {"Precipitation": 10.2, "Temperature": 3.8},
    },
    "CLMCOM-CCLM4-HADGEM": {
        "RCP4.5": {"Precipitation": 5.3, "Temperature": 3.0},
        "RCP8.5": {"Precipitation": 11.2, "Temperature": 5.3},
    },
    "DMI-HIRHAM-ECEARTH": {
        "RCP4.5": {"Precipitation": 1.1, "Temperature": 1.5},
        "RCP8.5": {"Precipitation": 4.0, "Temperature": 3.4},
    },
    "MPICSC-REMO1-MPIESM": {
        "RCP4.5": {"Precipitation": 3.0, "Temperature": 1.7},
        "RCP8.5": {"Precipitation": -1.9, "Temperature": 3.5},
    },
    "SMHI-RCA-IPSL": {
        "RCP4.5": {"Precipitation": -1.2, "Temperature": 2.9},
        "RCP8.5": {"Precipitation": -3.8, "Temperature": 5.3},
    },
}


def generate_future_climate(data: pd.DataFrame, name: str, output_folder: str) -> None:
    """
    Generate future climate data based on the delta change method.

    Args:
        data (pd.DataFrame): The DataFrame containing the simulated climate data.
        name (str): The prefix to use for the future climate data files.
        output_folder (str): The folder where the future climate data will be saved.

    Raises:
        TypeError: If the data is not a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    models = DELTA_CHANGES.keys()
    scenarios = ["RCP4.5", "RCP8.5"]

    print("Generating future climate data. Be patient...")

    for model in models:
        for scenario in scenarios:
            delta_change = DELTA_CHANGES[model][scenario]

            future_data = data.copy()
            future_data["Precipitation"] = future_data["Precipitation"] * (
                1 + delta_change["Precipitation"] / 100
            )

            future_data["T_max"] = future_data["T_max"] + delta_change["Temperature"]
            future_data["T_min"] = future_data["T_min"] + delta_change["Temperature"]
            future_data["T_avg"] = future_data["T_avg"] + delta_change["Temperature"]
            future_data["Year"] = future_data["Year"] + 90

            output_path = os.path.join(output_folder, f"{name}_{model}_{scenario}.csv")
            future_data.to_csv(output_path, index=False)

            print(f"Future climate data for {model} {scenario} saved to {output_path}")

    print("Future climate data generated successfully.")
