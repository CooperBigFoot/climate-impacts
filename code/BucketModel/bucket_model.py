import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import copy


@dataclass
class BucketModel:
    """
    A class to simulate hydrological processes using a simple bucket model. These processes include: Evapotranspiration, Surface Runoff, Groundwater Runoff, Snow Accumulation, Soil Storage, Groundwater Storage, Snow Mel, Rainfall and Snowfall.

    Args:
        k (float): Degree-day snowmelt parameter [mm/°C/day].
        S_max (float): Maximum soil water storage [mm].
        fr (float): Fraction of impermeable area at soil saturation [fractional value].
        rg (float): Mean residence time of water in groundwater [days].
        gauge_adj (float): Parameter to adjust for undercatch by rain gauge [fractional value].

    Attributes:
        S (float): Soil water content (initial condition) [mm].
        S_gw (float): Groundwater storage (initial condition) [mm].
        T_basin (float): Basin temperature [°C].
        T_max (float): Maximum temperature [°C].
        T_min (float): Minimum temperature [°C].
        Precip (float): Precipitation [mm].
        Rain (float): Rainfall [mm].
        Snow (float): Snowfall [mm].
        Snow_accum (float): Snow accumulation (cover) [mm].
        Snow_melt (float): Snow melt [mm].
        PET (float): Potential evapotranspiration [mm/day].
        ET (float): Evapotranspiration [mm/day].
        Q_s (float): Surface runoff [mm/day].
        Q_gw (float): Groundwater runoff [mm/day].
        Percol (float): Percolation [mm/day].
        Date (pd.Timestamp): Date [YYYY-MM-DD].

    Methods:
        set_catchment_properties(lapse_rate, station_elevation, basin_elevation, snowmelt_temp_threshold, latitude):
            Set the values of the constants.
        change_initial_conditions(S, S_gw):
            Change the initial conditions of the model (not implemented).
        adjust_temperature():
            Adjust the temperature based on lapse rate and elevations.
        gauge_adjustment():
            Adjust for undercatch by the rain gauge.
        partition_precipitation():
            Partition precipitation into rainfall and snowfall.
        compute_snow_melt():
            Compute snowmelt based on basin temperature.
        update_snow_accum():
            Update snow cover based on snowfall and snowmelt.
        compute_julianday():
            Compute the Julian day based on self.Date.
        compute_evapotranspiration():
            Compute evapotranspiration using the Hamon method.
        surface_runoff():
            Compute surface runoff.
        percolation(excess_water):
            Compute percolation.
        update_soil_moisture():
            Implement the water dynamics in the soil bucket.
        groundwater_runoff():
            Compute groundwater runoff with the linear reservoir concept.
        update_groundwater_storage():
            Update groundwater storage based on groundwater runoff.
        reset_variables():
            Reset the state variables to their initial values.
        run(data):
            Run the model with provided data.
        update_parameters(parameters):
            Update the model parameters.
        get_parameters():
            Return the model parameters.
        copy():
            Return a copy of the model.
    """

    k: float
    S_max: float
    fr: float
    rg: float
    gauge_adj: float

    S: float = field(default=10, init=False, repr=False)
    S_gw: float = field(default=100, init=False, repr=False)
    T_basin: float = field(default=0, init=False, repr=False)
    T_max: float = field(default=0, init=False, repr=False)
    T_min: float = field(default=0, init=False, repr=False)
    Precip: float = field(default=0, init=False, repr=False)
    Rain: float = field(default=0, init=False, repr=False)
    Snow: float = field(default=0, init=False, repr=False)
    Snow_accum: float = field(default=0, init=False, repr=False)
    Snow_melt: float = field(default=0, init=False, repr=False)
    PET: float = field(default=0, init=False, repr=False)
    ET: float = field(default=0, init=False, repr=False)
    Q_s: float = field(default=0, init=False, repr=False)
    Q_gw: float = field(default=0, init=False, repr=False)
    Percol: float = field(default=0, init=False, repr=False)
    Date: pd.Timestamp = field(
        default=pd.Timestamp("2000-08-14"), init=False, repr=False
    )

    LR: float = field(init=False, repr=False)
    H_STATION: float = field(init=False, repr=False)
    H_BASIN: float = field(init=False, repr=False)
    T_SM: float = field(init=False, repr=False)
    LAT: float = field(init=False, repr=False)

    def __post_init__(self):
        """
        Check the validity of the model parameters after initialization.
        """
        self.check_parameter_validity()

    def check_parameter_validity(self):
        """
        Check the validity of the model parameters.

        Args:
            ValueError: If any of the parameters are invalid.
        """
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.S_max <= 0:
            raise ValueError("S_max must be positive")
        if self.fr < 0 or self.fr > 1:
            raise ValueError("fr must be between 0 and 1")
        if self.rg < 1:
            raise ValueError("rg must be greater than 1")
        if self.gauge_adj < 0:
            raise ValueError("gauge_adj must be greater than or equal to 0")

    def set_catchment_properties(
        self,
        lapse_rate: float,
        station_elevation: float,
        basin_elevation: float,
        snowmelt_temp_threshold: float,
        latitude: float,
    ) -> None:
        """
        Set the values of the catchment properties.

        Args:
            lapse_rate (float): Lapse rate (°C/m).
            station_elevation (float): Station elevation (m.a.s.l).
            basin_elevation (float): Basin elevation (m.a.s.l).
            snowmelt_temp_threshold (float): Snowmelt temperature threshold (°C).
            latitude (float): Latitude in degrees.
        """
        self.LR = lapse_rate
        self.H_STATION = station_elevation
        self.H_BASIN = basin_elevation
        self.T_SM = snowmelt_temp_threshold
        self.LAT = latitude

    def change_initial_conditions(self, S: float = None, S_gw: float = None) -> None:
        """
        Change the initial conditions of the model.

        Args:
            S (float, optional): New initial soil water content (mm). Must be between 0 and S_max.
            S_gw (float, optional): New initial groundwater storage (mm). Must be non-negative.


        Raises:
            ValueError: If any of the provided values are outside their valid ranges.
        """
        if S is not None:
            if 0 <= S <= self.S_max:
                self.S = S
            else:
                raise ValueError(
                    f"Initial soil water content must be between 0 and {self.S_max} mm."
                )

        if S_gw is not None:
            if S_gw >= 0:
                self.S_gw = S_gw
            else:
                raise ValueError("Initial groundwater storage must be non-negative.")

        print("Initial conditions updated successfully.")
        print(f"Current initial conditions: S = {self.S} mm, S_gw = {self.S_gw} mm")

    def adjust_temperature(self) -> None:
        """
        Adjust the temperature based on lapse rate and elevations.

        Process:
            Compute the mean basin temperature.
            Adjust the temperature based on the lapse rate and elevation differences.
        """
        T = (self.T_max + self.T_min) / 2
        DELTA_H = self.H_STATION - self.H_BASIN
        LR_DELTA_H = self.LR * DELTA_H
        self.T_basin = T + LR_DELTA_H
        self.T_max += LR_DELTA_H
        self.T_min += LR_DELTA_H

    def gauge_adjustment(self) -> None:
        """
        Adjust for undercatch by the rain gauge.

        Process:
            Multiply precipitation by (1 + gauge adjustment parameter).
        """
        self.Precip = self.Precip * (1 + self.gauge_adj)

    def partition_precipitation(self) -> None:
        """
        Partition precipitation into rainfall and snowfall based on temperature thresholds.

        Process:
            If minimum temperature is above freezing, all precipitation is rainfall.
            If maximum temperature is below freezing, all precipitation is snowfall.
            Otherwise, partition based on temperature range.
        """
        if self.T_min > 0:
            self.Rain = self.Precip
            self.Snow = 0
        elif self.T_max <= 0:
            self.Snow = self.Precip
            self.Rain = 0
        else:
            rain_fraction = self.T_max / (self.T_max - self.T_min)
            self.Rain = self.Precip * rain_fraction
            self.Snow = self.Precip - self.Rain

    def compute_snow_melt(self) -> None:
        """
        Compute snowmelt based on basin temperature.

        Process:
            If basin temperature is below the snowmelt threshold, no snowmelt occurs.
            If basin temperature is above the snowmelt threshold, melt occurs as long as there is snow cover.
        """
        if self.T_basin <= self.T_SM:
            self.Snow_melt = 0
        else:
            self.Snow_melt = min(self.k * (self.T_basin - self.T_SM), self.Snow_accum)

    def update_snow_accum(self) -> None:
        """
        Update snow cover based on snowfall and snowmelt.

        Process:
            Add snowfall to snow accumulation.
            Subtract snowmelt from snow accumulation.
        """
        self.Snow_accum += self.Snow - self.Snow_melt

    def compute_julianday(self) -> int:
        """
        Compute the Julian day based on self.Date.

        Returns:
            int: Julian day (1 for January 1, ... , 365 or 366 for December 31).
        """
        return self.Date.timetuple().tm_yday

    def _Hamon_PET(self) -> float:
        """
        Calculate potential evapotranspiration using Hamon (1961).

        Reference:
            Hamon (1961): https://ascelibrary.org/doi/10.1061/JYCEAJ.0000599

        Returns:
            float: Potential evapotranspiration (mm/day).
        """
        J = self.compute_julianday()
        phi = np.radians(self.LAT)
        delta = 0.4093 * np.sin((2 * np.pi / 365) * J - 1.405)
        omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
        Nt = 24 * omega_s / np.pi
        a, b, c = 0.6108, 17.27, 237.3
        es = a * np.exp(b * self.T_basin / (self.T_basin + c))
        E = 2.1 * (Nt**2) * es / (self.T_basin + 273.3)
        return E

    def compute_evapotranspiration(self) -> None:
        """
        Compute evapotranspiration using the Hamon method.

        Process:
            Calculate potential evapotranspiration (PET) using the Hamon method.
            Compute actual evapotranspiration (ET) based on relative soil moisture.
        """
        self.PET = self._Hamon_PET()
        rel_soil_moisture = self.S / self.S_max
        self.ET = self.PET * rel_soil_moisture

    def surface_runoff(self) -> None:
        """
        Compute surface runoff.

        Process:
            Surface runoff is the fraction of rainfall and snowmelt based on impermeable area.
        """
        self.Q_s = (self.Rain + self.Snow_melt) * self.fr

    def percolation(self, excess_water: float) -> None:
        """
        Compute percolation.

        Args:
            excess_water (float): The excess water that percolates into the groundwater.
        """
        self.Percol = excess_water

    def update_soil_moisture(self) -> None:
        """
        Implement the water dynamics in the soil bucket.

        Process:
            Compute potential soil water content.
            If potential soil water content exceeds the maximum storage, compute surface runoff.
            Subtract surface runoff from potential soil water content.
            If potential soil water content still exceeds the maximum storage, compute percolation.
            Update soil water content.
        """
        potential_soil_water_content = self.S + self.Rain + self.Snow_melt - self.ET
        if potential_soil_water_content > self.S_max:
            self.surface_runoff()
        potential_soil_water_content -= self.Q_s
        if potential_soil_water_content > self.S_max:
            self.S = self.S_max
            water_excess = potential_soil_water_content - self.S_max
            self.percolation(water_excess)
        else:
            self.S = max(potential_soil_water_content, 0)

    def groundwater_runoff(self) -> None:
        """
        Compute groundwater runoff with the linear reservoir concept.

        Note:
            The minimal value of rg is 1.
        """
        self.Q_gw = self.S_gw / self.rg

    def update_groundwater_storage(self) -> None:
        """
        Update groundwater storage based on groundwater runoff.

        Process:
            Add percolation to groundwater storage.
            Subtract groundwater runoff from groundwater storage.
            Ensure groundwater storage is non-negative.
        """
        self.S_gw += self.Percol - self.Q_gw
        if self.S_gw < 0:
            self.S_gw = 0

    def reset_variables(self) -> None:
        """
        Reset the state variables to their initial values.

        Variables reset:
            Precip, Rain, Snow, Snow_melt, PET, ET, Q_s, Q_gw, Percol.
        """
        self.Precip = 0
        self.Rain = 0
        self.Snow = 0
        self.Snow_melt = 0
        self.PET = 0
        self.ET = 0
        self.Q_s = 0
        self.Q_gw = 0
        self.Percol = 0

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the model with the provided data.

        Args:
            data (pd.DataFrame): DataFrame with index 'date' and columns 'P_mix', 'T_max', 'T_min'.

        Returns:
            pd.DataFrame: DataFrame with the simulation results.
        """
        intermediate_results = {
            "ET": [],
            "Q_s": [],
            "Q_gw": [],
            "Snow_accum": [],
            "S": [],
            "S_gw": [],
            "Snow_melt": [],
            "Rain": [],
            "Snow": [],
            "Precip": [],
        }
        for index, row in data.iterrows():
            self.reset_variables()
            self.Date = index
            self.Precip = row["P_mix"]
            self.T_max = row["T_max"]
            self.T_min = row["T_min"]
            self.gauge_adjustment()
            self.adjust_temperature()
            self.partition_precipitation()
            self.compute_snow_melt()
            self.update_snow_accum()
            self.compute_evapotranspiration()
            self.update_soil_moisture()
            self.groundwater_runoff()
            self.update_groundwater_storage()
            for key in intermediate_results:
                intermediate_results[key].append(getattr(self, key))
        results_df = pd.DataFrame(intermediate_results, index=data.index)
        return results_df

    def update_parameters(self, parameters: dict) -> None:
        """
        Update the model parameters.

        Args:
            parameters (dict): A dictionary containing the parameters to update.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

        self.check_parameter_validity()  # Validate after updating all parameters

    def get_parameters(self) -> dict:
        """
        Return the model parameters.

        Returns:
            dict: A dictionary containing the model parameters.
        """
        parameters = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.init
        }
        return parameters

    def copy(self) -> "BucketModel":
        """
        Return a copy of the model.

        Returns:
            BucketModel: A deep copy of the model.
        """
        return copy.deepcopy(self)
