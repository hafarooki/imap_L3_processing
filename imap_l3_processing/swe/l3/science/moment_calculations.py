import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Tuple

import numpy as np
import spiceypy

from imap_l3_processing.constants import ELECTRON_MASS_KG, \
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN, METERS_PER_KILOMETER, \
    CENTIMETERS_PER_METER, PROTON_CHARGE_COULOMBS, GRAMS_PER_KILOGRAM
from imap_l3_processing.pitch_angles import calculate_unit_vector
from imap_l3_processing.predicted_ephemeris_tracker import PredictedEphemerisTracker

ELECTRON_MASS_OVER_BOLTZMANN_IN_CGS_UNITS = ELECTRON_MASS_KG / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * 1e-4
NUMBER_OF_DETECTORS = 7
NUMBER_OF_SPIN_SECTORS = 30
ZMK = 3.2971e-12

ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR = np.sqrt(2 * PROTON_CHARGE_COULOMBS / ELECTRON_MASS_KG) * 100

logger = logging.getLogger(__name__)

@dataclass
class Moments:
    alpha: float
    beta: float
    t_parallel: float
    t_perpendicular: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    density: float
    aoo: float
    ao: float

    @classmethod
    def construct_all_fill(cls):
        return cls(alpha=np.nan,
                   beta=np.nan,
                   t_parallel=np.nan,
                   t_perpendicular=np.nan,
                   velocity_x=np.nan,
                   velocity_y=np.nan,
                   velocity_z=np.nan,
                   density=np.nan,
                   aoo=np.nan,
                   ao=np.nan,
                   )


@dataclass
class MomentFitResults:
    moments: Moments
    chisq: float
    number_of_points: int
    regress_result: np.ndarray


@dataclass
class HaloCorrectionParameters:
    spacecraft_potential: float
    core_halo_breakpoint: float


@dataclass
class IntegrateOutputs:
    density: np.float64
    velocity: np.ndarray
    temperature: np.ndarray
    heat_flux: np.ndarray
    base_energy: np.float64


@dataclass
class ScaleDensityOutput:
    density: np.float64
    velocity: np.ndarray
    temperature: np.ndarray
    cdelnv: np.ndarray
    cdelt: np.ndarray


def core_fit_moments_retrying_on_failure(corrected_energy_bins: np.ndarray,
                                         velocity_vectors: np.ndarray,
                                         phase_space_density: np.ndarray,
                                         weights: np.ndarray,
                                         energy_start: int,
                                         energy_end: int,
                                         density_history: Union[np.ndarray, list[float]]) -> Optional[MomentFitResults]:
    return _fit_moments_retrying_on_failure(
        corrected_energy_bins,
        velocity_vectors,
        phase_space_density,
        weights,
        energy_start,
        energy_end,
        density_history,
        1.85
    )


def halo_fit_moments_retrying_on_failure(corrected_energy_bins: np.ndarray, velocity_vectors: np.ndarray,
                                         phase_space_density: np.ndarray,
                                         weights: np.ndarray,
                                         energy_start: int,
                                         energy_end: int,
                                         density_history: Union[np.ndarray, list[float]],
                                         spacecraft_potential: Union[np.float64, float],
                                         core_halo_breakpoint: Union[np.float64, float]) -> Optional[MomentFitResults]:
    return _fit_moments_retrying_on_failure(
        corrected_energy_bins,
        velocity_vectors,
        phase_space_density,
        weights,
        energy_start,
        energy_end,
        density_history,
        1.35,
        halo_correction_parameters=HaloCorrectionParameters(spacecraft_potential,
                                                            core_halo_breakpoint)
    )


def _fit_moments_retrying_on_failure(corrected_energy_bins: np.ndarray,
                                     velocity_vectors: np.ndarray,
                                     phase_space_density: np.ndarray,
                                     weights: np.ndarray,
                                     energy_start: int,
                                     energy_end: int,
                                     density_history: Union[np.ndarray, list[float]],
                                     history_scalar: float,
                                     halo_correction_parameters: Optional[HaloCorrectionParameters] = None) -> Optional[
    MomentFitResults]:
    filtered_velocity_vectors, filtered_weights, filtered_yreg = filter_and_flatten_regress_parameters(
        corrected_energy_bins,
        velocity_vectors,
        phase_space_density,
        weights,
        energy_start,
        energy_end)

    fit_function, chi_squared = regress(filtered_velocity_vectors, filtered_weights, filtered_yreg)
    moment = calculate_fit_temperature_density_velocity(fit_function)
    average_density = np.average(density_history) * history_scalar

    if halo_correction_parameters is not None:
        moment.density = halotrunc(moment, halo_correction_parameters.core_halo_breakpoint,
                                   halo_correction_parameters.spacecraft_potential)

    results = MomentFitResults(moments=moment, chisq=chi_squared,
                               number_of_points=energy_end - energy_start,
                               regress_result=fit_function
                               )
    if moment.density is not None and 0 < moment.density < average_density:
        return results
    elif energy_end - energy_start < 4:
        if moment.density is not None and moment.density > 0:
            return results
        else:
            return None
    else:
        return _fit_moments_retrying_on_failure(
            corrected_energy_bins,
            velocity_vectors,
            phase_space_density,
            weights,
            energy_start,
            energy_end - 1,
            density_history,
            history_scalar,
            halo_correction_parameters
        )


def regress(velocity_vectors: np.ndarray, weight: np.ndarray, yreg: np.ndarray) -> \
        np.ndarray:
    fit_function = np.zeros((len(velocity_vectors), 9))

    velocity_xs = velocity_vectors[:, 0]
    velocity_ys = velocity_vectors[:, 1]
    velocity_zs = velocity_vectors[:, 2]

    fit_function[:, 0] = -1 * ZMK * (velocity_xs ** 2)
    fit_function[:, 1] = -1 * ZMK * (velocity_ys ** 2)
    fit_function[:, 2] = -1 * ZMK * (velocity_zs ** 2)
    fit_function[:, 3] = -1 * ZMK * 2 * velocity_xs * velocity_ys
    fit_function[:, 4] = -1 * ZMK * 2 * velocity_xs * velocity_zs
    fit_function[:, 5] = -1 * ZMK * 2 * velocity_ys * velocity_zs
    fit_function[:, 6] = -1 * ZMK * 2 * velocity_xs
    fit_function[:, 7] = -1 * ZMK * 2 * velocity_ys
    fit_function[:, 8] = -1 * ZMK * 2 * velocity_zs

    r = np.zeros(9)
    a = np.zeros(10)
    sa = np.zeros(9)

    xmean = np.zeros(9)
    chisq = 0

    array = np.zeros((9, 9))

    wt = 1 / (weight * weight)
    sum = np.sum(wt)

    if sum == 0.0:
        return

    ymean = np.average(yreg, weights=wt)
    for i in range(9):
        xmean[i] = np.average(fit_function[:, i], weights=wt)

    wt /= sum

    yy = yreg - ymean

    for j in range(9):
        xx = fit_function[:, j] - xmean[j]
        r[j] = np.sum(wt * xx * yy)
        for k in range(j, 9):
            array[j][k] = np.sum(wt * xx * (fit_function[:, k] - xmean[k]))

    for j in range(9):
        for k in range(9):
            array[k][j] = array[j][k]

    # TODO the C code has an error case here, the following line can throw LinAlgError which may or may not be what the C code handles
    invarray = np.linalg.inv(array)

    ao = ymean

    yfit = np.zeros(len(fit_function))
    for j in range(9):
        a[j] = np.sum(r * invarray[j, :])
        ao -= a[j] * xmean[j]
        yfit += a[j] * fit_function[:, j]

    a[9] = ao
    for i in range(len(velocity_vectors)):
        yfit[i] += ao
        chisq += wt[i] * (yfit[i] - yreg[i]) * (yfit[i] - yreg[i])

    freen = len(velocity_vectors) - 9 - 1
    if freen > 0:
        chisq *= sum / freen
    elif freen == 0:
        chisq = 0.0

    for j in range(9):
        for k in range(9):
            invarray[j][k] /= sum
            sa[j] -= xmean[k] * invarray[j][k]

    sao = 1 / sum
    for j in range(9):
        for k in range(9):
            sao += xmean[j] * xmean[k] * invarray[j][k]

    return a, chisq


def calculate_fit_temperature_density_velocity(parameters: np.ndarray[float]):
    moments = Moments(alpha=0,
                      beta=0,
                      t_parallel=0,
                      t_perpendicular=0,
                      velocity_x=0,
                      velocity_y=0,
                      velocity_z=0,
                      density=0,
                      ao=0,
                      aoo=0
                      )

    moments.ao = parameters[9]
    if moments.ao == 0:
        return moments
    moments.alpha = np.atan2(parameters[5], parameters[4])
    moments.beta = np.atan2(parameters[3], parameters[4] * np.sin(moments.alpha))

    if moments.beta == 0.0:
        moments.t_perpendicular = 1.0 / parameters[0]
        moments.t_parallel = 1.0 / parameters[2]
    else:
        moments.t_perpendicular = 1.0 / (parameters[0] - parameters[3] * parameters[4] / parameters[5])
        moments.t_parallel = 1.0 / (parameters[0] + parameters[1] + parameters[2] - 2.0 / moments.t_perpendicular)

    c11 = parameters[1] * parameters[2] - parameters[5] * parameters[5]
    c12 = parameters[5] * parameters[4] - parameters[3] * parameters[2]
    c13 = parameters[3] * parameters[5] - parameters[1] * parameters[4]
    c22 = parameters[0] * parameters[2] - parameters[4] * parameters[4]
    c23 = parameters[4] * parameters[3] - parameters[0] * parameters[5]
    c33 = parameters[0] * parameters[1] - parameters[3] * parameters[3]

    d = parameters[0] * c11 + parameters[3] * c12 + parameters[4] * c13

    if d == 0:
        moments.velocity_x = 0
        moments.velocity_y = 0
        moments.velocity_z = 0
    else:
        moments.velocity_x = -1 * (parameters[6] * c11 + parameters[7] * c12 + parameters[8] * c13) / d
        moments.velocity_y = -1 * (parameters[6] * c12 + parameters[7] * c22 + parameters[8] * c23) / d
        moments.velocity_z = -1 * (parameters[6] * c13 + parameters[7] * c23 + parameters[8] * c33) / d

    fact = parameters[0] * moments.velocity_x ** 2 + parameters[1] * moments.velocity_y ** 2 + parameters[
        2] * moments.velocity_z ** 2

    moments.aoo = fact + parameters[3] * moments.velocity_x * moments.velocity_y + parameters[4] \
                  * moments.velocity_x * moments.velocity_z + \
                  parameters[5] * moments.velocity_y * moments.velocity_z

    moments.velocity_x *= 1e-5
    moments.velocity_y *= 1e-5
    moments.velocity_z *= 1e-5

    if (moments.t_parallel < 0) or (moments.aoo < 0) or (moments.aoo > 1e13):
        moments.density = 0.0
    else:
        moments.density = (moments.t_perpendicular * np.sqrt(moments.t_parallel) /
                           ((ZMK / math.pi) * np.sqrt(ZMK / math.pi))) * np.exp(
            moments.ao + ZMK * moments.aoo)
    return moments


LIMIT = np.array([32, 64, 128, 256, 1024, 2048, 3072, 5120, 9216, 17408, 33792])
SIGMA2 = np.array([0, 0.25, 1.25, 5.25, 21.25, 85.25, 341.25, 1365.25, 5461.25, 21845.25, 87381.25])
MINIMUM_WEIGHT = 0.8165
MAX_VARIANCE = 349525.25


def compute_maxwellian_weight_factors(count_rates: np.ndarray, acquisition_durations: np.ndarray) -> \
        np.ndarray[float]:
    correction = 1.0 - 1e-9 * LIMIT / acquisition_durations[:, :, np.newaxis]
    correction[correction < 0.1] = 0.1
    xlimits_per_measurement = LIMIT / correction

    counts = count_rates * acquisition_durations[:, :, np.newaxis]
    weights = np.empty_like(count_rates, dtype=np.float64)

    for (energy_i, spin_i, declination_i), corrected_count in np.ndenumerate(counts):
        if corrected_count <= 1.5:
            weights[energy_i, spin_i, declination_i] = MINIMUM_WEIGHT
        else:
            variance = MAX_VARIANCE
            xlimits = xlimits_per_measurement[energy_i, spin_i]
            for xlimit, sigma in zip(xlimits, SIGMA2):
                if corrected_count < xlimit:
                    variance = sigma
                    break
            weights[energy_i, spin_i, declination_i] = np.sqrt(variance + corrected_count) / corrected_count

    return weights


def filter_and_flatten_regress_parameters(corrected_energy_bins: np.ndarray,
                                          velocity_vectors: np.ndarray,
                                          phase_space_density: np.ndarray,
                                          weights: np.ndarray,
                                          start_index: int,
                                          end_index: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = np.full_like(phase_space_density, fill_value=False, dtype=bool)
    valid_mask[start_index:end_index] = True
    valid_mask[corrected_energy_bins <= 0] = False
    valid_mask[phase_space_density <= 0] = False

    filtered_phase_space_density = phase_space_density[valid_mask]
    yreg = np.zeros_like(filtered_phase_space_density)
    yreg[filtered_phase_space_density > 1e-35] = np.log(
        filtered_phase_space_density[filtered_phase_space_density > 1e-35])
    yreg[filtered_phase_space_density <= 1e-35] = -80.6

    return velocity_vectors[valid_mask], weights[valid_mask], yreg


def get_dps_to_rtn_rotation_matrix(epoch: datetime) -> np.ndarray:
    return spiceypy.pxform("IMAP_DPS", "IMAP_RTN", spiceypy.datetime2et(epoch))


def apply_rotation_matrix(rotation_matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return rotation_matrix @ vector


def rotate_rtn_vectors_to_dps(epochs: np.ndarray, vectors_rtn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    result = np.full_like(vectors_rtn, np.nan, dtype=float)
    predicted_ephemeris_flags = np.zeros_like(epochs, dtype=bool)
    for i, epoch in enumerate(epochs):
        tracker = PredictedEphemerisTracker()
        try:
            et_time = spiceypy.datetime2et(epoch)
            rotation_matrix = tracker.run(spiceypy.pxform,"IMAP_RTN", "IMAP_DPS", et_time)
        except spiceypy.SpiceyError as e:
            logger.info(f"Failed to rotate RTN→DPS at epoch {epoch}: {e}")
            continue
        result[i] = rotation_matrix @ vectors_rtn[i]

        predicted_ephemeris_flags[i] = tracker.used_predict
    return result, predicted_ephemeris_flags


def rotate_temperature(rotation_matrix: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
    sin_dec = np.sin(beta)
    x = sin_dec * np.cos(alpha)
    y = sin_dec * np.sin(alpha)
    z = np.cos(beta)

    rtn_temperature = apply_rotation_matrix(rotation_matrix, np.array([x, y, z]))

    theta = np.asin(rtn_temperature[2])
    phi = np.atan2(rtn_temperature[1], rtn_temperature[0])

    return theta, phi


def rotate_vector_to_rtn_spherical_coordinates(rotation_matrix: np.ndarray, heat_flux: np.ndarray) -> tuple[
    float, float, float]:
    r, t, n = apply_rotation_matrix(rotation_matrix, heat_flux)
    magnitude = np.linalg.norm(heat_flux, axis=-1)
    rt = np.sqrt(r * r + t * t)
    theta = np.arctan2(n, rt)
    phi = np.arctan2(t, r)
    return magnitude, theta, phi


def compute_density_scale(core_electron_energy_range: float, speed: float, temperature: float) -> float:
    energy_multplier = 11600.0
    density_multiplier = 0.88623
    ev_speed = 593.097

    velocity_delta = ev_speed * math.sqrt(core_electron_energy_range)

    velocity_plus = velocity_delta + speed
    velocity_minus = velocity_delta - speed

    energy_plus = (velocity_plus / ev_speed) ** 2
    energy_minus = (velocity_minus / ev_speed) ** 2

    scalep2 = energy_multplier * energy_plus / temperature
    scalep = math.sqrt(scalep2)

    scalem2 = energy_multplier * energy_minus / temperature
    scalem = math.sqrt(scalem2)

    escalep = math.erfc(scalep) if scalep < 9 else 0
    escalem = math.erfc(scalem) if scalem < 9 else 0

    dscalep = density_multiplier / (scalep * np.exp(-scalep2) + density_multiplier * escalep)
    dscalem = density_multiplier / (scalem * np.exp(-scalem2) + density_multiplier * escalem)

    return 0.5 * (dscalep + dscalem)


def halotrunc(moments: Moments, core_halo_breakpoint: float, spacecraft_potential: float) -> Optional[np.float64]:
    htmax = moments.t_parallel
    htmin = moments.t_perpendicular

    speed = math.sqrt(moments.velocity_x ** 2 + moments.velocity_y ** 2 + moments.velocity_z ** 2)

    halod = moments.density

    if htmax > 1.0e8 or htmin > 1e8:
        return None

    temp = (htmax + 2 * htmin) / 3

    if htmax <= 1e4 or htmin <= 1e4:
        dscale = 1
    elif core_halo_breakpoint - spacecraft_potential > 5 and temp < 1.0e7:
        dscale = compute_density_scale(core_halo_breakpoint - spacecraft_potential, speed, temp)
    else:
        dscale = 1

    halod = halod / np.clip(dscale, 1, 5)

    return halod


def integrate(istart, iend, energy: np.ndarray, sintheta: np.ndarray,
              costheta: np.ndarray, deltheta: np.ndarray, fv: np.ndarray, phi: np.ndarray,
              spacecraft_potential: float, cdelnv: np.ndarray, cdelt: np.ndarray) -> Optional[
    IntegrateOutputs]:
    sumn = 0
    sumvx = 0
    sumvy = 0
    sumvz = 0
    base = 1000
    delphi = np.full(7, 2 * np.pi / 30)
    delv = np.empty((len(energy)))
    fv_with_axis_order_energy_cem_spin = np.moveaxis(fv, 1, 2)
    for i in range(istart, iend + 1):
        if energy[i] > 0:
            v2mid = (ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR ** 2) * energy[i]
            v3mid = v2mid * np.sqrt(v2mid)
        else:
            v2mid = 0
            v3mid = 0
        ehigh = 1.175 * energy[i]
        if i < len(energy) - 1:
            ehigh = np.sqrt((energy[i + 1] + spacecraft_potential) * (energy[i] + spacecraft_potential))
        elow = np.sqrt((energy[i] + spacecraft_potential) * (energy[i - 1] + spacecraft_potential))

        delv[i] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * (np.sqrt(ehigh) - np.sqrt(elow))
        if elow < base:
            base = elow
        for j in range(7):
            for k in range(28):  # why 28 and not 30?
                if energy[i] > 0:
                    delta = delv[i] * deltheta[j] * delphi[j]
                    fact = sintheta[j] * fv_with_axis_order_energy_cem_spin[i, j, k]
                    sumn += delta * v2mid * fact
                    sumvx += delta * v3mid * fact * sintheta[j] * np.cos(np.deg2rad(phi[i, k]))
                    sumvy += delta * v3mid * fact * sintheta[j] * np.sin(np.deg2rad(phi[i, k]))
                    sumvz += delta * v3mid * fact * costheta[j]

    totden = sumn + cdelnv[0]
    if totden <= 0:
        return None

    base -= spacecraft_potential
    CM_PER_KM = METERS_PER_KILOMETER * CENTIMETERS_PER_METER
    KM_PER_CM = 1 / CM_PER_KM
    output_velocities = np.array([
        (-KM_PER_CM * sumvx + cdelnv[1]) / totden,
        (-KM_PER_CM * sumvy + cdelnv[2]) / totden,
        (-KM_PER_CM * sumvz + cdelnv[3]) / totden,
    ])

    sumtxx = 0
    sumtxy = 0
    sumtxz = 0
    sumtyy = 0
    sumtyz = 0
    sumtzz = 0
    sumqx = 0
    sumqy = 0
    sumqz = 0
    for i in range(istart, iend + 1):
        for j in range(7):
            v2mid = (ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR ** 2) * energy[i]
            for k in range(28):  # again why 28?
                if energy[i] > 0:
                    angx = sintheta[j] * np.cos(np.deg2rad(phi[i, k]))
                    vx = -ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[i]) * angx - CM_PER_KM * \
                         output_velocities[0]

                    angy = sintheta[j] * np.sin(np.deg2rad(phi[i, k]))
                    vy = -ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[i]) * angy - CM_PER_KM * \
                         output_velocities[1]

                    angz = costheta[j]
                    vz = -ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[i]) * angz - CM_PER_KM * \
                         output_velocities[2]

                    vmag2 = vx * vx + vy * vy + vz * vz

                    delta = delv[i] * deltheta[j] * delphi[j]
                    fact = v2mid * sintheta[j] * fv_with_axis_order_energy_cem_spin[i, j, k]
                    sumtxx += delta * fact * vx * vx
                    sumtxy += delta * fact * vx * vy
                    sumtxz += delta * fact * vx * vz
                    sumtyy += delta * fact * vy * vy
                    sumtyz += delta * fact * vy * vz
                    sumtzz += delta * fact * vz * vz
                    sumqx += delta * fact * vmag2 * vx
                    sumqy += delta * fact * vmag2 * vy
                    sumqz += delta * fact * vmag2 * vz

    TEMPERATURE_SCALING_FACTOR_TO_UNDO_IN_EIGEN = 1e-4
    temperature = (np.array([sumtxx, sumtxy, sumtyy, sumtxz, sumtyz, sumtzz]) *
                   TEMPERATURE_SCALING_FACTOR_TO_UNDO_IN_EIGEN * ELECTRON_MASS_OVER_BOLTZMANN_IN_CGS_UNITS + cdelt) / totden

    heat_flux = np.array([sumqx, sumqy, sumqz]) * 500 * ELECTRON_MASS_KG * GRAMS_PER_KILOGRAM

    return IntegrateOutputs(totden, output_velocities, temperature, heat_flux, base)


def scale_core_density(core_density: float,
                       core_velocity: np.ndarray, core_temp: np.ndarray,
                       core_moment_fit: Moments, ifit: int, energy: np.ndarray,
                       spacecraft_potential: float, cosin_p: np.ndarray,
                       aperture_field_of_view: list,
                       phi: np.ndarray,
                       regress_outputs: np.ndarray,
                       base_energy: float) -> ScaleDensityOutput:
    zmk = ELECTRON_MASS_KG / (2 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * 1e4)
    MAX_SPIN_SECTOR_INDEX = 28  # why 28 and not 30?

    number_of_energies = 1
    assert 0 < ifit < len(energy) - 1, "ifit must be in middle of energies"
    delv = np.zeros((2, NUMBER_OF_DETECTORS))
    velocity_in_sc_frame = np.zeros((2, NUMBER_OF_DETECTORS))
    for j in range(NUMBER_OF_DETECTORS):
        ehigh = np.sqrt((energy[ifit + 1] + spacecraft_potential) *
                        (energy[ifit] + spacecraft_potential))
        elow = np.sqrt((energy[ifit] + spacecraft_potential) *
                       (energy[ifit - 1] + spacecraft_potential))
        delv[0, j] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * (np.sqrt(ehigh) - np.sqrt(elow))
        velocity_in_sc_frame[0, j] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[ifit])
        base_energy = min(base_energy, elow)
    base_energy -= spacecraft_potential
    if base_energy > 0:
        number_of_energies = 2
        velocity_in_sc_frame[1, :] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(0.5 * base_energy)
        delv[1, :] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(base_energy)

    factor = core_moment_fit.density * zmk / (np.pi * core_moment_fit.t_perpendicular) * np.sqrt(
        zmk / (np.pi * core_moment_fit.t_parallel))

    cos_theta = cosin_p[np.newaxis, :, np.newaxis]
    sin_theta = np.sin(np.arccos(cos_theta))
    delta_theta = np.array(aperture_field_of_view)[np.newaxis, :, np.newaxis]
    cos_phi = np.cos(np.deg2rad(phi[0]))
    sin_phi = np.sin(np.deg2rad(phi[0]))
    delta_phi = 2 * np.pi / NUMBER_OF_SPIN_SECTORS

    velocity_3_dim = velocity_in_sc_frame[:, :, np.newaxis]
    vx = -velocity_3_dim * sin_theta * cos_phi
    vy = -velocity_3_dim * sin_theta * sin_phi
    vz = -velocity_3_dim * cos_theta

    fun = -zmk * np.stack(np.broadcast_arrays(vx * vx, vy * vy, vz * vz, vx * vy, vx * vz, vy * vz, vx, vy, vz),
                          axis=-1)
    exponent = -zmk * core_moment_fit.aoo + np.dot(fun, regress_outputs[:9])
    delt = delv[:, :, np.newaxis] * delta_theta * delta_phi
    common_factor = np.square(velocity_3_dim) * delt * sin_theta * factor * np.exp(exponent)

    def integrate(array):
        return np.sum(array[:number_of_energies, :NUMBER_OF_DETECTORS, :MAX_SPIN_SECTOR_INDEX])

    sumint = integrate(common_factor)
    sumvx = integrate(vx * common_factor)
    sumvy = integrate(vy * common_factor)
    sumvz = integrate(vz * common_factor)

    sumtxx = integrate(vx * vx * common_factor)
    sumtxy = integrate(vx * vy * common_factor)
    sumtxz = integrate(vx * vz * common_factor)
    sumtyy = integrate(vy * vy * common_factor)
    sumtyz = integrate(vy * vz * common_factor)
    sumtzz = integrate(vz * vz * common_factor)

    sum_velocities = np.array([sumvx, sumvy, sumvz])
    sum_temperatures = np.array([sumtxx, sumtxy, sumtyy, sumtxz, sumtyz, sumtzz])

    corrected_density = core_density + sumint

    delta_v = -1e-5 * sum_velocities

    corrected_core_velocity = core_velocity.copy()
    corrected_core_velocity *= core_density / corrected_density
    corrected_core_velocity += delta_v / corrected_density

    cdelt = sum_temperatures * 1e-4 * ELECTRON_MASS_OVER_BOLTZMANN_IN_CGS_UNITS

    corrected_core_temp = core_temp.copy()
    corrected_core_temp *= core_density / corrected_density
    corrected_core_temp += cdelt / corrected_density

    cdelnv = np.append(sumint, delta_v)

    return ScaleDensityOutput(density=corrected_density, velocity=corrected_core_velocity,
                              temperature=corrected_core_temp, cdelnv=cdelnv, cdelt=cdelt)


def scale_halo_density(halo_density: float,
                       halo_velocity: np.ndarray, halo_temp: np.ndarray,
                       halo_moment_fit: Moments,
                       spacecraft_potential: float,
                       core_halo_break: float,
                       cosin_p: np.ndarray,
                       aperture_field_of_view: list,
                       phi: np.ndarray,
                       regress_outputs: np.ndarray,
                       base_energy: float) -> ScaleDensityOutput:
    zmk = ELECTRON_MASS_KG / (2 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * 1e4)
    MAX_SPIN_SECTOR_INDEX = 28  # why 28 and not 30?

    hchbreak = core_halo_break - spacecraft_potential
    scval = abs(hchbreak - base_energy)
    min_energy = min(base_energy, hchbreak)
    vsch = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(min_energy + 0.5 * scval)
    deltav = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(scval)
    factor = halo_moment_fit.density * zmk / (np.pi * halo_moment_fit.t_parallel) * np.sqrt(
        zmk / (np.pi * halo_moment_fit.t_perpendicular))

    cos_theta = cosin_p[:, np.newaxis]
    sin_theta = np.sin(np.arccos(cos_theta))
    delta_theta = np.array(aperture_field_of_view)[:, np.newaxis]
    cos_phi = np.cos(np.deg2rad(phi[0]))
    sin_phi = np.sin(np.deg2rad(phi[0]))
    delta_phi = 2 * np.pi / NUMBER_OF_SPIN_SECTORS

    vx = -vsch * sin_theta * cos_phi
    vy = -vsch * sin_theta * sin_phi
    vz = -vsch * cos_theta

    fun = -zmk * np.stack(np.broadcast_arrays(vx * vx, vy * vy, vz * vz, vx * vy, vx * vz, vy * vz, vx, vy, vz),
                          axis=-1)
    exponent = np.dot(fun, regress_outputs[:9])
    delt = deltav * delta_theta * delta_phi
    common_factor = np.square(vsch) * delt * sin_theta * factor * np.exp(exponent)

    def integrate(array):
        return np.sum(array[:NUMBER_OF_DETECTORS, :MAX_SPIN_SECTOR_INDEX])

    sumint = integrate(common_factor)
    sumvx = integrate(vx * common_factor)
    sumvy = integrate(vy * common_factor)
    sumvz = integrate(vz * common_factor)

    sumtxx = integrate(vx * vx * common_factor)
    sumtxy = integrate(vx * vy * common_factor)
    sumtxz = integrate(vx * vz * common_factor)
    sumtyy = integrate(vy * vy * common_factor)
    sumtyz = integrate(vy * vz * common_factor)
    sumtzz = integrate(vz * vz * common_factor)

    sum_velocities = np.array([sumvx, sumvy, sumvz])
    sum_temperatures = np.array([sumtxx, sumtxy, sumtyy, sumtxz, sumtyz, sumtzz])

    if base_energy > hchbreak:
        delta_density = sumint
        delta_velocity = sum_velocities
        delta_temperature = sum_temperatures
    else:
        delta_density = -sumint
        delta_velocity = -sum_velocities
        delta_temperature = -sum_temperatures

    corrected_density = halo_density + delta_density

    corrected_halo_velocity = halo_velocity.copy()
    corrected_halo_velocity *= halo_density / corrected_density
    corrected_halo_velocity += delta_velocity * (-1e-5) / corrected_density

    corrected_halo_temp = halo_temp.copy()
    corrected_halo_temp *= halo_density / corrected_density
    corrected_halo_temp += delta_temperature * 1e-4 * ELECTRON_MASS_OVER_BOLTZMANN_IN_CGS_UNITS / corrected_density

    return ScaleDensityOutput(density=corrected_density, velocity=corrected_halo_velocity,
                              temperature=corrected_halo_temp, cdelnv=None, cdelt=None)


def calculate_primary_eigenvector(temperature_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric_temperature_tensor = np.zeros(shape=(3, 3))

    symmetric_temperature_tensor[0][0] = temperature_tensor[0]
    symmetric_temperature_tensor[0][1] = symmetric_temperature_tensor[1][0] = temperature_tensor[1]
    symmetric_temperature_tensor[1][1] = temperature_tensor[2]
    symmetric_temperature_tensor[0][2] = symmetric_temperature_tensor[2][0] = temperature_tensor[3]
    symmetric_temperature_tensor[1][2] = symmetric_temperature_tensor[2][1] = temperature_tensor[4]
    symmetric_temperature_tensor[2][2] = temperature_tensor[5]

    nan_array = np.full(3, np.nan)
    try:
        eigen_values, eigen_vectors = np.linalg.eigh(symmetric_temperature_tensor)
    except np.linalg.LinAlgError:
        return nan_array, nan_array

    if np.any(eigen_values < 0) or np.all(eigen_values == 0) or np.any(np.isnan(eigen_values)):
        return nan_array, nan_array

    eigen_values_mean = np.mean(eigen_values)

    diff = 0
    ipar = 0
    for i in range(len(eigen_values)):
        if math.fabs(eigen_values[i] - eigen_values_mean) > diff:
            ipar = i
            diff = math.fabs(eigen_values[i] - eigen_values_mean)

    primary_evec = eigen_vectors[:, ipar]
    TEMPERATURE_SCALING_FACTOR = 1e4
    parallel_temperature = TEMPERATURE_SCALING_FACTOR * eigen_values[ipar]
    other_evals = [v for i, v in enumerate(eigen_values) if i != ipar]
    max_eval = max(other_evals)
    min_eval = min(other_evals)
    perpendicular_temperature = TEMPERATURE_SCALING_FACTOR * np.sqrt(min_eval * max_eval)
    if min_eval == 0:
        gyro = 1
    else:
        gyro = max(other_evals) / min_eval
    return primary_evec, np.array([parallel_temperature, perpendicular_temperature, gyro])


def rotation_matrix_builder(mag_vector: np.ndarray) -> np.ndarray(shape=(3, 3)):
    normalized_mag_vector = mag_vector / np.linalg.norm(mag_vector)
    smallest_index = np.argmin(np.abs(normalized_mag_vector))
    reference_vector = np.zeros(3)
    reference_vector[smallest_index] = 1
    row_1 = calculate_unit_vector(np.cross(normalized_mag_vector, reference_vector))
    row_2 = calculate_unit_vector(np.cross(normalized_mag_vector, row_1))
    return np.stack((normalized_mag_vector, row_1, row_2), axis=0)


def rotate_temperature_tensor_to_mag(temperature_tensor: np.ndarray, mag_vector: np.ndarray) -> tuple[
    float, float, float]:
    symmetric_temperature_tensor = np.zeros(shape=(3, 3))

    symmetric_temperature_tensor[0][0] = temperature_tensor[0]
    symmetric_temperature_tensor[0][1] = symmetric_temperature_tensor[1][0] = temperature_tensor[1]
    symmetric_temperature_tensor[1][1] = temperature_tensor[2]
    symmetric_temperature_tensor[0][2] = symmetric_temperature_tensor[2][0] = temperature_tensor[3]
    symmetric_temperature_tensor[1][2] = symmetric_temperature_tensor[2][1] = temperature_tensor[4]
    symmetric_temperature_tensor[2][2] = temperature_tensor[5]

    rotation_matrix = rotation_matrix_builder(mag_vector)

    rotated_tensor = rotation_matrix @ symmetric_temperature_tensor @ rotation_matrix.T

    t_parallel = rotated_tensor[0][0] * 1e4

    t_perpendicular_1 = (rotated_tensor[1][1] + rotated_tensor[2][2]) * 1e4 / 2
    t_perpendicular_2 = 0

    if rotated_tensor[1][1] > 0 and rotated_tensor[2][2] > 0:
        t_perpendicular_2 = rotated_tensor[1][1] / rotated_tensor[2][2]
        if t_perpendicular_2 < 1:
            t_perpendicular_2 = 1 / t_perpendicular_2

    return t_parallel, t_perpendicular_1, t_perpendicular_2
