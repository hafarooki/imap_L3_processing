from __future__ import annotations

from dataclasses import dataclass, field

import lmfit
import numpy as np
import scipy.optimize
import spiceypy
import uncertainties
from imap_processing.swapi.l2 import swapi_l2
from lmfit import Parameters
from numpy import ndarray
from scipy.linalg import inv
from uncertainties import ufloat
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, \
    HE_PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS, ONE_AU_IN_KM, \
    METERS_PER_KILOMETER, CENTIMETERS_PER_METER, ONE_SECOND_IN_NANOSECONDS, BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
from imap_l3_processing.swapi.l3a.science.speed_calculation import calculate_combined_sweeps, calculate_sw_speed
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable, \
    InstrumentResponseLookupTableCollection
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
import numdifftools as ndt


def calculate_pickup_ion_values(instrument_response_lookup_table, geometric_factor_calibration_table,
                                energy: np.ndarray[float], count_rates: uarray, center_of_epoch: int,
                                sw_velocity_vector: ndarray,
                                density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
                                efficiency_table: EfficiencyCalibrationTable, hydrogen_inflow_vector: InflowVector,
                                helium_inflow_vector: InflowVector) -> FittingParameters:
    ephemeris_time = spiceypy.unitim(center_of_epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
    sw_velocity = np.linalg.norm(sw_velocity_vector)

    energy_labels = range(62, 0, -1)
    lower_energy_cutoff = 1.25 * calculate_pui_energy_cutoff(PROTON_MASS_KG, ephemeris_time, sw_velocity_vector,
                                                             hydrogen_inflow_vector)
    upper_energy_cutoff = 1.2 * calculate_pui_energy_cutoff(HE_PUI_PARTICLE_MASS_KG, ephemeris_time, sw_velocity_vector,
                                                            helium_inflow_vector)
    sweep_count = len(count_rates)
    average_count_rates, energies = calculate_combined_sweeps(count_rates, energy)

    extracted_energy_labels, extracted_energies, extracted_count_rates = extract_pui_energy_bins(energy_labels,
                                                                                                 energies,
                                                                                                 average_count_rates,
                                                                                                 lower_energy_cutoff,
                                                                                                 upper_energy_cutoff)
    model_count_rate_calculator = ModelCountRateCalculator(instrument_response_lookup_table,
                                                           geometric_factor_calibration_table, sw_velocity_vector,
                                                           density_of_neutral_helium_lookup_table, efficiency_table,
                                                           helium_inflow_vector)
    indices = list(zip(extracted_energy_labels, extracted_energies))

    def make_parameters(cooling_index, ionization_rate, cutoff_speed, background_count_rate) -> Parameters:
        params = Parameters()
        params.add('cooling_index', value=cooling_index, min=1.0, max=5.0)
        params.add('ionization_rate', value=ionization_rate, min=0.6e-9, max=8.0e-7)
        params.add('cutoff_speed', value=cutoff_speed, min=sw_velocity * .8, max=sw_velocity * 1.2)
        params.add('background_count_rate', value=background_count_rate, min=0, max=0.2)
        return params

    params = make_parameters(1.50, 1e-7, sw_velocity, 0.1)

    def map_to_internal(value, param):
        return np.arcsin(2 * (value - param.min) / (param.max - param.min) - 1)

    def map_param_values_to_internal_values(ci, ir, cs, bcr):
        return [
            map_to_internal(ci, params['cooling_index']),
            map_to_internal(ir, params['ionization_rate']),
            map_to_internal(cs, params['cutoff_speed']),
            map_to_internal(bcr, params['background_count_rate']),
        ]

    initial_simplex=np.array([
        map_param_values_to_internal_values(1.5, 1e-7, sw_velocity, 0.1),
        map_param_values_to_internal_values(5.0, 1e-7, sw_velocity, 0.1),
        map_param_values_to_internal_values(1.5, 2.1e-7, sw_velocity, 0.1),
        map_param_values_to_internal_values(1.5, 1e-7, sw_velocity * 1.2, 0.1),
        map_param_values_to_internal_values(1.5, 1e-7, sw_velocity, 0.2),
    ])
    minimizer = lmfit.Minimizer(
        calc_chi_squared_lm_fit, params,
                                fcn_args=(extracted_count_rates, indices, model_count_rate_calculator, ephemeris_time,
                                sweep_count),
                                scale_covar=False,
                                options=dict(initial_simplex=initial_simplex),
                                )
    result = minimizer.minimize(method="nelder")
    flags = SwapiL3Flags.NONE
    if result.redchi > 10:
        flags |= SwapiL3Flags.BAD_FIT

    param_vals = result.uvars
    if result.uvars is None:
        flags |= SwapiL3Flags.PUI_FIT_MISSING_UNCERTAINTY
        Hfun = ndt.Hessian(minimizer.penalty, step=1.e-4)
        try:
            hessian_ndt = Hfun(result.x)
            cov_x = inv(hessian_ndt) * 2.0
            scaled_cov = minimizer._int2ext_cov_x(cov_x, result.x)
            uncertainties = np.sqrt(np.diag(scaled_cov))
            param_vals = {}

            nominal_value_by_param_name = result.params.valuesdict()
            for var_name, uncertainty in zip(result.var_names, uncertainties):
                param_vals[var_name] = ufloat(nominal_value_by_param_name[var_name], uncertainty)
        except:
            param_vals = {k: ufloat(v, np.nan) for k, v in result.params.valuesdict().items()}


    return FittingParameters(param_vals["cooling_index"], param_vals["ionization_rate"], param_vals["cutoff_speed"],
                             param_vals["background_count_rate"], flags)


def calculate_helium_pui_density(epoch: int,
                                 sw_velocity_vector: ndarray,
                                 density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
                                 fitting_params: FittingParameters, helium_inflow_vector: InflowVector) -> float:
    @uncertainties.wrap
    def calculate(cooling_index: float,
                  ionization_rate: float,
                  cutoff_speed: float,
                  background_count_rate: float):
        fitting_params = FittingParameters(
            cooling_index, ionization_rate, cutoff_speed, background_count_rate
        )
        ephemeris_time = spiceypy.unitim(epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
        model = build_forward_model(fitting_params, ephemeris_time, sw_velocity_vector,
                                    density_of_neutral_helium_lookup_table, helium_inflow_vector)
        lower_discontinuity = (density_of_neutral_helium_lookup_table.get_minimum_distance() / (
                model.distance_km / ONE_AU_IN_KM)) ** (
                                      1 / fitting_params.cooling_index) * fitting_params.cutoff_speed
        points = (0, lower_discontinuity, fitting_params.cutoff_speed)

        results = scipy.integrate.quad(lambda v: model.f(v) * v * v, 0, fitting_params.cutoff_speed, limit=100,
                                       points=points)
        return 4 * np.pi * results[0] / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3

    return calculate(fitting_params.cooling_index,
                     fitting_params.ionization_rate,
                     fitting_params.cutoff_speed,
                     fitting_params.background_count_rate)


def calculate_helium_pui_temperature(epoch: int,
                                     sw_velocity_vector: ndarray,
                                     density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
                                     fitting_params: FittingParameters, helium_inflow_vector: InflowVector) -> float:
    @uncertainties.wrap
    def calculate(cooling_index: float,
                  ionization_rate: float,
                  cutoff_speed: float,
                  background_count_rate: float):
        fitting_params = FittingParameters(
            cooling_index, ionization_rate, cutoff_speed, background_count_rate
        )
        ephemeris_time = spiceypy.unitim(epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
        model = build_forward_model(fitting_params, ephemeris_time, sw_velocity_vector,
                                    density_of_neutral_helium_lookup_table, helium_inflow_vector)
        lower_discontinuity = (density_of_neutral_helium_lookup_table.get_minimum_distance() / (
                model.distance_km / ONE_AU_IN_KM)) ** (
                                      1 / fitting_params.cooling_index) * fitting_params.cutoff_speed
        points = (0, lower_discontinuity, fitting_params.cutoff_speed)

        numerator = scipy.integrate.quad(lambda v: model.f(v) * v ** 4, 0, fitting_params.cutoff_speed, points=points,
                                         limit=100)
        denominator = scipy.integrate.quad(lambda v: model.f(v) * v ** 2, 0, fitting_params.cutoff_speed, points=points,
                                           limit=100)
        return HE_PUI_PARTICLE_MASS_KG / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN) * \
            numerator[0] / denominator[0] * \
            METERS_PER_KILOMETER ** 2

    return calculate(fitting_params.cooling_index,
                     fitting_params.ionization_rate,
                     fitting_params.cutoff_speed,
                     fitting_params.background_count_rate)


@dataclass
class FittingParameters:
    cooling_index: float
    ionization_rate: float
    cutoff_speed: float
    background_count_rate: float
    flags: int = SwapiL3Flags.NONE


@dataclass
class ForwardModel:
    fitting_params: FittingParameters
    ephemeris_time: float
    solar_wind_speed_inertial_frame: float
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable
    distance_km: float
    psi: float

    def f(self, pickup_ion_speed):
        w = pickup_ion_speed / self.fitting_params.cutoff_speed
        radius_in_au = self.distance_km / ONE_AU_IN_KM
        neutral_helium_density_per_cm3 = self.density_of_neutral_helium_lookup_table.density(
            self.psi, radius_in_au * w ** self.fitting_params.cooling_index)
        neutral_helium_density_per_km3 = neutral_helium_density_per_cm3 * (
                CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
        term1 = self.fitting_params.cooling_index / (4 * np.pi)
        term2 = (self.fitting_params.ionization_rate * ONE_AU_IN_KM ** 2) / (
                self.distance_km * self.solar_wind_speed_inertial_frame * self.fitting_params.cutoff_speed ** 3)
        term3 = w ** (self.fitting_params.cooling_index - 3)
        term4 = neutral_helium_density_per_km3
        term5 = np.heaviside(1 - w, 0.5)
        return term1 * term2 * term3 * term4 * term5


def build_forward_model(fitting_params: FittingParameters, ephemeris_time: float, solar_wind_vector: ndarray,
                        density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
                        helium_inflow_vector: InflowVector) -> ForwardModel:
    solar_wind_vector_eclipj2000_frame = convert_velocity_relative_to_imap(solar_wind_vector,
                                                                           ephemeris_time,
                                                                           "IMAP_DPS",
                                                                           "ECLIPJ2000")
    imap_position_eclip2000_frame_state = spiceypy.spkezr(
        "IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")[0][0:3]
    distance_km, longitude, latitude = spiceypy.reclat(imap_position_eclip2000_frame_state)
    psi = np.rad2deg(longitude) - helium_inflow_vector.longitude_deg_eclipj2000

    return ForwardModel(fitting_params, ephemeris_time,
                        np.linalg.norm(solar_wind_vector_eclipj2000_frame),
                        density_of_neutral_helium_lookup_table, distance_km, psi)


@dataclass
class ModelCountRateCalculator:
    response_lookup_table_collection: InstrumentResponseLookupTableCollection
    geometric_table: GeometricFactorCalibrationTable
    solar_wind_vector: np.ndarray
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable
    efficiency_table: EfficiencyCalibrationTable
    helium_inflow_vector: InflowVector
    _speed_grid_cache: dict = field(default_factory=dict)

    def get_speed_grid(self, response_lookup_table: InstrumentResponseLookupTable, ephemeris_time: float):

        key = (id(response_lookup_table), ephemeris_time)
        cached = self._speed_grid_cache.get(key)
        if cached is not None:
            return cached

        speed_inst = calculate_sw_speed(HE_PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS,
                                        response_lookup_table.energy)

        pui_vector_instrument_frame = calculate_pui_velocity_vector(speed_inst, response_lookup_table.elevation,
                                                                    response_lookup_table.azimuth)
        pui_vector_eclipj2000_frame = convert_velocity_relative_to_imap(pui_vector_instrument_frame,
                                                                        ephemeris_time,
                                                                        "IMAP_SWAPI", "ECLIPJ2000")

        solar_wind_vector_eclipj2000_frame = convert_velocity_relative_to_imap(self.solar_wind_vector,
                                                                               ephemeris_time,
                                                                               "IMAP_DPS",
                                                                               "ECLIPJ2000")

        pui_vector_solar_wind_frame = pui_vector_eclipj2000_frame - solar_wind_vector_eclipj2000_frame
        speed = np.linalg.norm(pui_vector_solar_wind_frame, axis=-1)
        self._speed_grid_cache[key] = speed
        return speed

    def model_count_rate(self, indices_and_energy_centers: list[tuple[int, float]],
                         fitting_params: FittingParameters, ephemeris_time: float) -> np.ndarray:
        forward_model = build_forward_model(fitting_params, ephemeris_time, self.solar_wind_vector,
                                            self.density_of_neutral_helium_lookup_table,
                                            self.helium_inflow_vector)
        model_count_rates = []
        for energy_bin_index, energy_bin_center in indices_and_energy_centers:
            model_count_rates.append(
                self.model_one_count_rate(energy_bin_index, energy_bin_center, forward_model)
            )
        return np.array(model_count_rates)

    def model_one_count_rate(self, energy_bin_index, energy_bin_center, forward_model) -> float:
        response_lookup_table = self.response_lookup_table_collection.get_table_for_energy_bin(energy_bin_index)
        speed_grid = self.get_speed_grid(response_lookup_table, forward_model.ephemeris_time)
        integral = model_count_rate_integral(response_lookup_table, forward_model, speed_grid)
        efficiency = self.efficiency_table.get_alpha_efficiency_for(forward_model.ephemeris_time)
        geometric_factor = self.geometric_table.lookup_geometric_factor(energy_bin_center)
        return efficiency * (geometric_factor / 2) * integral + forward_model.fitting_params.background_count_rate


def calc_chi_squared_lm_fit(params: Parameters, observed_count_rates: np.ndarray,
                            indices_and_energy_centers: list[tuple[int, float]], calculator: ModelCountRateCalculator,
                            ephemeris_time: float, sweep_count: int):
    parvals = params.valuesdict()

    cooling_index = parvals["cooling_index"]
    ionization_rate = parvals["ionization_rate"]
    cutoff_speed = parvals["cutoff_speed"]
    background_count_rate = parvals["background_count_rate"]

    fit_params = FittingParameters(cooling_index, ionization_rate, cutoff_speed, background_count_rate)
    modeled_rates = calculator.model_count_rate(indices_and_energy_centers, fit_params, ephemeris_time)

    modeled_counts = modeled_rates * sweep_count * swapi_l2.SWAPI_LIVETIME
    observed_counts = observed_count_rates * sweep_count * swapi_l2.SWAPI_LIVETIME
    result = np.sqrt(2 * (modeled_counts - observed_counts + observed_counts * np.log(
        observed_counts / modeled_counts)))
    return result


def model_count_rate_integral(response_lookup_table: InstrumentResponseLookupTable, forward_model: ForwardModel,
                              speed_grid):
    count_rates = forward_model.f(speed_grid)

    integrals = response_lookup_table.integral_factor * count_rates

    return np.sum(integrals)


def convert_velocity_to_reference_frame(velocity: ndarray, ephemeris_time: float, from_frame: str,
                                        to_frame: str) -> ndarray:
    rotation_matrix = spiceypy.sxform(from_frame, to_frame, ephemeris_time)

    state = velocity[..., np.newaxis]

    state_in_target_frame = np.matmul(rotation_matrix[3:6, 3:6], state)
    return state_in_target_frame[..., 0]


def convert_velocity_relative_to_imap(velocity, ephemeris_time, from_frame, to_frame):
    velocity_in_target_frame_relative_to_imap = convert_velocity_to_reference_frame(velocity, ephemeris_time,
                                                                                    from_frame, to_frame)
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, to_frame, "NONE", "SUN")[0][3:6]

    return velocity_in_target_frame_relative_to_imap + imap_velocity


def calculate_velocity_vector(sw_speed: ndarray, elevation: ndarray, azimuth: ndarray) -> np.ndarray:
    elevation_radians = np.deg2rad(elevation)
    azimuth_radians = np.deg2rad(azimuth)
    z = sw_speed * np.sin(elevation_radians)
    xy_radius = sw_speed * np.cos(elevation_radians)
    x = xy_radius * np.cos(azimuth_radians)
    y = xy_radius * np.sin(azimuth_radians)
    return np.transpose([x, y, z])


def calculate_pui_velocity_vector(speed: ndarray, elevation: ndarray, azimuth: ndarray) -> np.ndarray:
    y_axis_azimuth = 90
    return calculate_velocity_vector(-speed, elevation, y_axis_azimuth - azimuth)


def calculate_pui_energy_cutoff(particle_mass: float, ephemeris_time: float, sw_velocity_in_imap_frame,
                                particle_inflow_vector: InflowVector):
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")[0][
                    3:6]
    solar_wind_velocity = convert_velocity_relative_to_imap(
        sw_velocity_in_imap_frame, ephemeris_time, "IMAP_DPS", "ECLIPJ2000")
    particle_velocity = spiceypy.latrec(-particle_inflow_vector.speed_km_per_s,
                                        particle_inflow_vector.longitude_deg_eclipj2000,
                                        particle_inflow_vector.latitude_deg_eclipj2000)

    particle_velocity_cutoff_vector = solar_wind_velocity - particle_velocity - imap_velocity
    particle_speed_cutoff = np.linalg.norm(particle_velocity_cutoff_vector)
    return (0.5 * (particle_mass / PROTON_CHARGE_COULOMBS) * (2 * particle_speed_cutoff * METERS_PER_KILOMETER) ** 2)


def extract_pui_energy_bins(energy_bin_labels, energies, observed_count_rates, lower_energy_cutoff,
                            upper_energy_cutoff):
    extracted_energy_bins = []
    count_rates = []
    extracted_energy_bin_labels = []

    for label, energy, count_rate in zip(energy_bin_labels, energies, observed_count_rates):
        if energy > lower_energy_cutoff and energy < upper_energy_cutoff and count_rate > 0:
            extracted_energy_bins.append(energy)
            count_rates.append(count_rate)
            extracted_energy_bin_labels.append(label)

    return np.array(extracted_energy_bin_labels), np.array(extracted_energy_bins), np.array(count_rates)


def calculate_solar_wind_velocity_vector(speeds: ndarray, deflection_angle: ndarray, clock_angle: ndarray) -> ndarray:
    elevation_angle = 90 - deflection_angle
    return calculate_velocity_vector(-speeds, elevation_angle, clock_angle)


def calculate_ten_minute_velocities(speeds: ndarray, deflection_angle: ndarray, clock_angle: ndarray,
                                    quality_flags: list[SwapiL3Flags]) -> (ndarray, ndarray):
    velocity_vector = calculate_solar_wind_velocity_vector(speeds, deflection_angle, clock_angle)
    left_slice = 0
    chunked_velocities = []
    chunked_quality_flags = []
    while left_slice < len(velocity_vector):
        ten_min_slice = slice(left_slice, left_slice + 10)
        ten_min_quality_flag = np.bitwise_or.reduce(quality_flags[ten_min_slice])

        chunked_velocities.append(np.mean(velocity_vector[ten_min_slice], axis=0))
        chunked_quality_flags.append(ten_min_quality_flag)

        left_slice += 10

    return np.array(chunked_velocities), np.array(chunked_quality_flags)
