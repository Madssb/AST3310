"""
Author: Mads S. Balto
Description:
Models radiative and convective energy transport of a Sun-like star.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from project1 import EnergyProduction
import os
import matplotlib.pyplot as plt
from cross_section import cross_section
from tabulate import tabulate
import random

STEFAN_BOLTZMANN_CONSTANT = 5.6704e-8  # W / (m^2 K^4)
GRAVITATIONAL_CONSTANT = 6.6742e-11  # N m^2 / kg^2
BOLTZMANN_CONSTANT = 1.3806e-23  # m^2 kg / (s^2 K)
ATOMIC_MASS_UNIT = 1.6605e-27  # kg
ADIABATIC_TEMPERATURE_GRAD = 2 / 5
DELTA = 1
LUMINOSITY_SUN = 3.846e26  # kg m^2 /s^3  (W)
MASS_SUN = 1.989e30  # kg
RADIUS_SUN = 6.96e8  # m
AVERAGE_MASS_DENSITY_SUN = 1.408e3  # kg/m^3
SURFACE_TEMPERATURE = 5770 #K
SPEED_OF_LIGHT = 2.9979e8  # m/s
SURFACE_DENSITY = AVERAGE_MASS_DENSITY_SUN*1.42e-7

def mean_molecular_mass():
    """
    Computes the mean molecular mass.

    Returns:
        mean molecular mass.
    """
    free_particles = {"hydrogen_1": 2, "helium_3": 3, "helium_4": 3,
                      "lithium_7": 4, "beryllium_7": 5, "nitrogen_14": 8}
    mass_fraction = {"hydrogen_1": 0.7, "helium_3": 1e-10, "helium_4": 0.29,
                     "lithium_7": 1e-7, "beryllium_7": 1e-7,
                     "nitrogen_14": 1e-11}
    nucleons = {"hydrogen_1": 1, "helium_3": 3, "helium_4": 4, "lithium_7": 7,
                "beryllium_7": 7, "nitrogen_14": 14}
    isotopes = free_particles.keys()
    sum = 0
    for isotope in isotopes:
        sum += free_particles[isotope]/nucleons[isotope]*mass_fraction[isotope]
    return 1/sum


MEAN_MOLECULAR_MASS = mean_molecular_mass()


def specific_heat_capacity():
    """
    Computes the specific heat capacity.

    Returns:
        (float) Specific heat capacity.
    """
    return 5 * BOLTZMANN_CONSTANT / 2 / MEAN_MOLECULAR_MASS / ATOMIC_MASS_UNIT


SPECIFIC_HEAT_CAPACITY = specific_heat_capacity()


def radial_coordinate_differential(radial_position, mass_density):
    """
    Computes the differential of the radial coordinate with respect to mass,
    for a point at a given radial position within a sun-like star.

    Args:
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].


    Returns:
        float: The differential of the radial coordinate with respect to mass,
            measured in units of distance per unit mass.
    """
    return 1 / (4 * np.pi * radial_position**2 * mass_density)


def log_10_opacity(mass_density, temperature):
    """
    Computes opacity as functions of R and
    temperature using data from "opacity.txt".

    Args:
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].

    Returns:
        opacity (float) Opacity [m^2/kg] of sun-like star, as a function
        of temperature and pressure, per interpolation of data
        from 'opacity.txt'.
    """
    data = np.genfromtxt("opacity.txt")
    LOG_10_RS = data[0][1:]
    LOG_10_TEMPERATURES = data[1:, 0]
    LOG_10_OPACITIES = data[1:, 1:]
    instance = RectBivariateSpline(
        LOG_10_TEMPERATURES, LOG_10_RS, LOG_10_OPACITIES)

    LOG_10_TEMPERATURE = np.log10(temperature)
    MASS_DENSITY_CGS = mass_density/1000
    R = MASS_DENSITY_CGS/(temperature/1e6)**3
    LOG_10_R = np.log10(R)
    LOG_10_OPACITY = instance.ev(LOG_10_TEMPERATURE, LOG_10_R)
    OUT_OF_BOUNDS = (LOG_10_TEMPERATURES[0] > LOG_10_TEMPERATURE.any()
                     or LOG_10_TEMPERATURES[-1] < LOG_10_TEMPERATURE.any()
                     or LOG_10_RS[0] > LOG_10_R.any()
                     or LOG_10_RS[-1] < LOG_10_R.any())
    if OUT_OF_BOUNDS:
        #print("Warning: entered out of bounds.")
        pass
    assert np.isfinite(mass_density).any()  
    assert np.isfinite(temperature).any()
    assert np.isfinite(MASS_DENSITY_CGS).any()
    assert np.isfinite(R).any()
    try:
        assert np.isfinite(LOG_10_R).any()
    except AssertionError:
        print(f"{R=:.4g}")
        print(f"{LOG_10_R=:.4g}")
        raise ValueError
    assert np.isfinite(LOG_10_OPACITY).any()
    return LOG_10_OPACITY


def opacity(temperature, mass_density, sanity_check=False):
    LOG_10_OPACITY = log_10_opacity(mass_density, temperature)
    OPACITY_CGS = 10**LOG_10_OPACITY
    OPACITY_SI = OPACITY_CGS/10
    if sanity_check:
        return 3.98
    assert np.isfinite(LOG_10_OPACITY).any()
    assert np.isfinite(OPACITY_CGS).any()
    assert np.isfinite(OPACITY_SI).any()
    return OPACITY_SI


def check_relative_error(actual, approx, tol):
    """
    Checks if the relative error between the actual and approximate values is
    within the tolerance.

    Args:
        actual: The actual value.
        approx: The approximate value.
        tol: The tolerance value.

    Returns:
        None. Raises an AssertionError if the relative error is outside the
        tolerance.
    """
    error = abs(actual - approx)
    rel_error = error / abs(actual)
    assert rel_error < tol, f"{rel_error=} is larger than tolerance {tol}"


def sanity_check_opacity():
    """
    Verifies that opacity() works as intended.

    Returns:
        None.
    """
    LOG_10_TEMPERATURES = np.asarray([3.750, 3.755, 3.755, 3.755, 3.755,
                                      3.770, 3.780, 3.795, 3.770, 3.775,
                                      3.780, 3.795, 3.800])
    TEMPERATURES = 10**(LOG_10_TEMPERATURES)  # K
    LOG_10_R = np.asfarray([-6.00, -5.95, -5.80, -5.70,
                            -5.55, -5.95, -5.95, -5.95,
                            -5.80, -5.75, -5.70, -5.55,
                            -5.50])
    R = 10**LOG_10_R
    MASS_DENSITIES_CGS = R*(TEMPERATURES/1e6)**3
    MASS_DENSITIES_SI = MASS_DENSITIES_CGS*1e3
    LOG_10_OPACITIES = log_10_opacity(MASS_DENSITIES_SI, TEMPERATURES)
    OPACITIES = opacity(TEMPERATURES, MASS_DENSITIES_SI)
    EXPECTED_OPACITIES = [2.84e-3, 3.11e-3, 2.68e-3, 2.46e-3, 2.12e-3,
                          4.70e-3, 6.25e-3, 9.45e-3, 4.05e-3, 4.43e-3,
                          4.94e-3, 6.89e-3, 7.69e-3]
    table = np.stack([LOG_10_TEMPERATURES, LOG_10_R,
                     LOG_10_OPACITIES, OPACITIES], axis=-1)
    headers = ["log10 T", "log10 R", "log 10 opacity", "opacity"]
    print(tabulate(table, headers=headers))
    for index, opacity_ in enumerate(OPACITIES):
        check_relative_error(EXPECTED_OPACITIES[index], opacity_, tol=6e-2)


def radiative_pressure(temperature):
    """
    Computes the radiative pressure:

    Args:
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].

    Returns:
        (float): Radiative pressure.
    """
    RADIATION_DENSITY_CONSTANT = 4*STEFAN_BOLTZMANN_CONSTANT/SPEED_OF_LIGHT/3
    return RADIATION_DENSITY_CONSTANT*temperature**4


def pressure_to_mass_density(temperature, pressure):
    """
    Compute the mass density.

    Args:
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        pressure (float): The pressure in units of Pa. [N/m^2]

    Returns:
        (float) mass density in units of kg/m^3
    """
    RADIATIVE_PRESSURE = radiative_pressure(temperature)
    GAS_PRESSURE = pressure - RADIATIVE_PRESSURE
    return (
        ATOMIC_MASS_UNIT*MEAN_MOLECULAR_MASS/BOLTZMANN_CONSTANT/temperature
        * np.abs(GAS_PRESSURE)
    )


def mass_density_to_pressure(temperature, mass_density):
    """
    Compute the pressure given some temperature and mass density.

    Args:
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        mass_density (float): The mass density [kg/m^3]
    returns:
        (float) pressure [?]
    """
    RADIATIVE_PRESSURE = radiative_pressure(temperature)
    GAS_PRESSURE = (
        BOLTZMANN_CONSTANT*temperature/MEAN_MOLECULAR_MASS/ATOMIC_MASS_UNIT
        * mass_density
    )
    return RADIATIVE_PRESSURE + GAS_PRESSURE


def pressure_differential(mass, radial_position):
    """
    Computes the differential of pressure with respect to mass.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].

    Returns:
        (float): Differential of pressure with respect to mass.
    """
    try:
        assert mass.any() > 0
        assert radial_position.any() > 0
    except AttributeError:
        assert mass > 0
        assert radial_position > 0
    return -GRAVITATIONAL_CONSTANT * mass / 4 / np.pi / radial_position**4


def total_energy_production_rate(mass_density, temperature):
    """
    Computes the total energy production rate. (dL/dm)
    Args:
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].

    returns:
        (float): The total energy production rate. []
    """
    ENERGY_PRODUCTION_INSTANCE = EnergyProduction(mass_density, temperature)
    return ENERGY_PRODUCTION_INSTANCE.true_total_energy_production_rate_v2()


def gravitational_acceleration(mass, radial_position):
    """"
    Computes gravitational acceleration

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].

    Returns:
        (float): Gravitational acceleration [m/s^2]
    """
    return mass * GRAVITATIONAL_CONSTANT / radial_position**2


def pressure_scale_height(mass, radial_position, temperature):
    """
    Computes pressure scale height.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].

    Returns:
        (float): Pressure scale height. 
    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    return (
        BOLTZMANN_CONSTANT*temperature/ATOMIC_MASS_UNIT /
        MEAN_MOLECULAR_MASS/GRAVITATIONAL_ACCELERATION
    )


def stable_temperature_grad(mass, radial_position, mass_density,
                            temperature, luminosity, sanity_check=False):
    """
    Compute the temperature gradient nescesary for all the energy to be
    carried by radiation.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 

    Returns:
        (float): The temperature gradient nescesary for all the energy to be
            carried by radiation.
    """
    OPACITY = opacity(temperature, mass_density, sanity_check)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    return(
        3
        * luminosity
        * OPACITY
        * mass_density
        * PRESSURE_SCALE_HEIGHT
        / 64
        / np.pi
        / radial_position**2
        / STEFAN_BOLTZMANN_CONSTANT
        / temperature**4
    )


def u(mass, radial_position, mass_density, temperature, sanity_check=False):
    """
    Compute unnamed but useful quantity U.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 

    Returns:
        (float): unnamed but useful quantity U.
    """
    OPACITY = opacity(temperature, mass_density, sanity_check=sanity_check)
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    assert np.isfinite(OPACITY).any()
    assert np.isfinite(GRAVITATIONAL_ACCELERATION).any()
    assert np.isfinite(PRESSURE_SCALE_HEIGHT).any()
    return (
        64
        * STEFAN_BOLTZMANN_CONSTANT
        * temperature**3
        * np.sqrt(PRESSURE_SCALE_HEIGHT / GRAVITATIONAL_ACCELERATION / DELTA)
        / 3
        / OPACITY
        / mass_density**2
        / SPECIFIC_HEAT_CAPACITY
    )


def xi(mass, radial_position, mass_density, temperature, luminosity,
       sanity_check=False):
    """
    computes xi.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): xi.
    """

    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    GEOMETRIC_FACTOR = 4 / DELTA / PRESSURE_SCALE_HEIGHT
    STABLE_TEMPERATURE_GRAD = stable_temperature_grad(
        mass, radial_position, mass_density, temperature, luminosity,
        sanity_check=sanity_check)
    U = u(mass, radial_position, mass_density,
          temperature, sanity_check=sanity_check)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    coeffs = np.asarray(
        [
            np.ones_like(mass),
            U / MIXING_LENGTH**2,
            U**2 * GEOMETRIC_FACTOR / MIXING_LENGTH**3,
            U
            / MIXING_LENGTH**2
            * (ADIABATIC_TEMPERATURE_GRAD - STABLE_TEMPERATURE_GRAD),
        ]
    )
    xis = np.empty_like(PRESSURE_SCALE_HEIGHT)
    if isinstance(mass, np.ndarray):
        for i in range(len(PRESSURE_SCALE_HEIGHT)):
            roots = np.roots(coeffs[:, i])
            index = np.where(roots.imag == 0)[0][0]
            XI = roots[index].real
            xis[i] = XI
        return xis
    roots = np.roots(coeffs)
    index = np.where(roots.imag == 0)[0][0]
    XI = roots[index].real
    return XI


def temperature_grad(mass, radial_position, mass_density, temperature,
                     luminosity, sanity_check=False):
    """
    Computes the temperature gradient.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 

    Returns:
        (float): Temperature gradient.
    """
    STABLE_TEMPERATURE_GRAD = stable_temperature_grad(mass,
                                                      radial_position,
                                                      mass_density,
                                                      temperature,
                                                      luminosity)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    GEOMETRIC_FACTOR = 4 / DELTA / PRESSURE_SCALE_HEIGHT
    U = u(mass, radial_position, mass_density,
          temperature, sanity_check=sanity_check)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    XI = xi(mass, radial_position, mass_density, temperature,
            luminosity, sanity_check=sanity_check)
    if isinstance(mass, np.ndarray):
        temperature_grads = np.empty_like(mass)
        for i in range(len(mass)):
            CONVECTIVELY_UNSTABLE = (STABLE_TEMPERATURE_GRAD[i] 
                                     > ADIABATIC_TEMPERATURE_GRAD)
            if not CONVECTIVELY_UNSTABLE:
                temperature_grads[i] = STABLE_TEMPERATURE_GRAD[i]
            else:
                temperature_grads[i] = (
                    XI[i]**2 + U[i]*GEOMETRIC_FACTOR[i]*XI[i]/MIXING_LENGTH[i]
                    + ADIABATIC_TEMPERATURE_GRAD)
        return temperature_grads
    CONVECTIVELY_UNSTABLE = STABLE_TEMPERATURE_GRAD > ADIABATIC_TEMPERATURE_GRAD
    if not CONVECTIVELY_UNSTABLE:
        return STABLE_TEMPERATURE_GRAD
    return (XI**2 + U*GEOMETRIC_FACTOR*XI/MIXING_LENGTH 
            + ADIABATIC_TEMPERATURE_GRAD)


def parcel_velocity(mass, radial_position, mass_density, temperature,
                    luminosity, sanity_check=False):
    """
    Compute the velocity of parcel.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): parcel velocity [m/s]
    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    PARCEL_DISTANCE_MOVED = PRESSURE_SCALE_HEIGHT/2
    XI = xi(mass, radial_position, mass_density, temperature, luminosity,
            sanity_check=sanity_check)
    return (np.sqrt(GRAVITATIONAL_ACCELERATION*DELTA/PRESSURE_SCALE_HEIGHT)
            * XI*PARCEL_DISTANCE_MOVED)


def convective_flux(mass, radial_position, mass_density, temperature,
                    luminosity, sanity_check=False):
    """
    Computes convective flux.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 

    Returns:
        (float): The convective flux.
    """
    STABLE_TEMPERATURE_GRAD = stable_temperature_grad(mass,
                                                      radial_position,
                                                      mass_density,
                                                      temperature,
                                                      luminosity)
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    XI = xi(mass, radial_position, mass_density,
            temperature, luminosity, sanity_check=sanity_check) 
    if isinstance(mass,np.ndarray):
        #booleans and array is a yucky combination
        convective_fluxes = np.empty_like(mass)
        for i in range(len(mass)):
            convective_fluxes[i] = (mass_density[i]*SPECIFIC_HEAT_CAPACITY*temperature[i]
                    * np.sqrt(GRAVITATIONAL_ACCELERATION[i]*DELTA/PRESSURE_SCALE_HEIGHT[i]**3)
                    * (MIXING_LENGTH[i]/2)**2*XI[i]**3)
            CONVECTIVELY_UNSTABLE = (STABLE_TEMPERATURE_GRAD[i] 
                                > ADIABATIC_TEMPERATURE_GRAD)
            if not CONVECTIVELY_UNSTABLE:
                convective_fluxes[i] = 0
        return convective_fluxes
    return (mass_density*SPECIFIC_HEAT_CAPACITY*temperature
            * np.sqrt(GRAVITATIONAL_ACCELERATION*DELTA/PRESSURE_SCALE_HEIGHT**3)
            * (MIXING_LENGTH/2)**2*XI**3)


def radiative_flux(mass, radial_position, mass_density, temperature,
                   luminosity, sanity_check=False):
    """
    Computes radiative flux.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): Radiative flux.
    """
    OPACITY = opacity(temperature, mass_density, sanity_check=sanity_check)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    TEMPERATURE_GRAD = temperature_grad(
        mass, radial_position, mass_density, temperature, luminosity,
        sanity_check=sanity_check)
    return (16*STEFAN_BOLTZMANN_CONSTANT*temperature**4*TEMPERATURE_GRAD/3
            /OPACITY/mass_density/PRESSURE_SCALE_HEIGHT)


def append_line_to_data_file(data, filename):
    """
    Appends data to new line in specified file
    """
    data = np.reshape(data, (1, len(data)))
    with open(filename, 'a') as f:
        np.savetxt(f, data, delimiter=',', fmt='%f')


def initialize_file(filename):
    """
    Clears the contents of the file specified by filename, if it exists, or
    creates the file if it doesn't exist.

    Args:
        Filename (str): name of file to be initialized.

    Returns:
        None. 
    """
    if os.path.exists(filename):
        with open(filename, 'w') as f:
            f.truncate(0)


def temperature_diff(mass, radial_position, mass_density, temperature,
                             pressure, luminosity, dpdm, sanity_check=False):
    """
    computes the differential of temperature with respect to mass.

    Args:
        mass (float): mass inside sphere of radius radial_position, with mass
            density of sun-like star [kg].
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        pressure (float): Local pressure inside at radial_position,
            inside sun-like star.
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        dpdm (float): The differential of pressure with respect to mass.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 
    Returns:
        (float): The differential of temperature with respect to mass.
    """
    TEMPERATURE_GRAD = temperature_grad(
        mass, radial_position, mass_density, temperature, luminosity,
        sanity_check=sanity_check)
    return TEMPERATURE_GRAD*temperature/pressure*dpdm


def compute_temperature_diff_radiative_only(
    radial_position, mass_density, temperature, luminosity, sanity_check=False
):
    """
    Computes the differential of temperature with respect to mass for
        convectively stable mass shell.

    Args:
        radial_position (float): distance from center of sun-like star [m].
        mass_density (float): local mass density at radial_position, inside
            sun-like star [kg/m^3].
        temperature (float): local temperature at radial_position, inside
            sun-like star [K].
        luminosity (float): luminosity emitted by mass inside sphere of radius
            radial_position, with mass density of sun-like star.sphere with
            radius equal to radial position, consisting of sun-like star.
        sanity_check (Bool): False if not used for sanity check,
            true if used for sanity check. 

    Returns:
        (float): the differential of temperature with respect to mass, for
            convectively stable mass shell.
    """
    OPACITY = opacity(temperature, mass_density, sanity_check=sanity_check)
    return (
        -3
        * OPACITY
        * luminosity
        / 256
        / np.pi**2
        / STEFAN_BOLTZMANN_CONSTANT
        / radial_position**4
        / temperature**3
    )


def sanity_check_gradients():
    """
    Verifies quantity computing functions by comparing with expected values.

    Returns:
        None.
    """
    TEMPERATURE = 0.9e6  # K
    MASS_DENSITY = 55.9  # kg/m^3
    PRESSURE = mass_density_to_pressure(TEMPERATURE, MASS_DENSITY)
    RADIAL_POSITION = 0.84*RADIUS_SUN
    MASS = 0.99*MASS_SUN
    LUMINOSITY = LUMINOSITY_SUN
    STABLE_TEMPERATURE_GRAD = stable_temperature_grad(MASS,
                                                      RADIAL_POSITION,
                                                      MASS_DENSITY,
                                                      TEMPERATURE,
                                                      LUMINOSITY,
                                                      sanity_check=True)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        MASS, RADIAL_POSITION, TEMPERATURE)
    U = u(MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, sanity_check=True)
    XI = xi(MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, LUMINOSITY,
            sanity_check=True)
    TEMPERATURE_GRAD = temperature_grad(MASS, RADIAL_POSITION,
                                        MASS_DENSITY, TEMPERATURE,
                                        LUMINOSITY,
                                        sanity_check=True)
    PARCEL_VELOCITY = parcel_velocity(MASS, RADIAL_POSITION, MASS_DENSITY,
                                      TEMPERATURE, LUMINOSITY,
                                      sanity_check=True)
    RADIATIVE_FLUX = radiative_flux(
        MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, LUMINOSITY,
        sanity_check=True)
    CONVECTIVE_FLUX = convective_flux(
        MASS, RADIAL_POSITION, MASS_DENSITY, TEMPERATURE, LUMINOSITY,
        sanity_check=True)
    FLUX_SUM = RADIATIVE_FLUX + CONVECTIVE_FLUX
    CONVECTIVE_FLUX_RATIO = CONVECTIVE_FLUX/FLUX_SUM
    RADIATIVE_FLUX_RATIO = RADIATIVE_FLUX/FLUX_SUM
    check_relative_error(0.6, MEAN_MOLECULAR_MASS, 4e-2)
    check_relative_error(32.4e6, PRESSURE_SCALE_HEIGHT, 3e-2)
    check_relative_error(3.26, STABLE_TEMPERATURE_GRAD, 4e-2)
    check_relative_error(5.94e5, U, 2e-2)
    check_relative_error(1.173e-3, XI, 2e-2)
    check_relative_error(0.400, TEMPERATURE_GRAD, 1e-2)
    check_relative_error(65.50, PARCEL_VELOCITY, 2e-3)
    check_relative_error(0.88, CONVECTIVE_FLUX_RATIO, 5e-2)
    check_relative_error(0.12, RADIATIVE_FLUX_RATIO, 6e-2)
    print("success")


def energy_production_branches(mass_density, temperature):
    instance = EnergyProduction(mass_density, temperature)
    TOT_EPR_PP1 = instance.energy_production_rate_pp_1()
    TOT_EPR_PP2 = instance.energy_production_rate_pp_2()
    TOT_EPR_PP3 = instance.energy_production_rate_pp_3()
    TOT_EPR_CNO = instance.energy_production_rate_cno()
    return TOT_EPR_PP1, TOT_EPR_PP2, TOT_EPR_PP3, TOT_EPR_CNO


class EnergyTransport:
    """
    Models central parts of a Sun-like star.
    """

    def __init__(self, temperature=SURFACE_TEMPERATURE,
                 luminosity=LUMINOSITY_SUN,
                 mass=MASS_SUN,
                 radius=RADIUS_SUN,
                 mass_density=SURFACE_DENSITY,
                 star_name="main"):
        """
        Initializes instance of EnergyTransport. 

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
            radius (float): Radius of sun-like star [m].
            temperature (float): Surface temperature for sun-like star [K].
            mass (float): Mass of sun-like star [kg].
            luminosity (float): Luminosity of sun-like star [W].
            mass_density (float): mass density at surface of sun-like star [kg/m^3].
        """
        self.star_name = star_name
        self.filename = "data/" + self.star_name + "_data.txt"
        self.radial_position = radius
        self.mass = mass
        self.mass_density = mass_density
        self.pressure = mass_density_to_pressure(temperature, mass_density)
        self.luminosity = luminosity
        self.temperature = temperature
        self.luminosity = luminosity
        self.init_radial_position = radius
        self.init_mass = mass
        self.init_mass_density = mass_density
        self.init_temperature = temperature
        self.init_pressure = self.pressure
        self.init_luminosity =luminosity

    def advance(self):
        """
        Evolves stellar parameter quantities 1 step with Euler's method, with
        variable step length.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        STABLE_TEMPERATURE_GRAD = stable_temperature_grad(self.mass,
                                                          self.radial_position,
                                                          self.mass_density,
                                                          self.temperature,
                                                          self.luminosity)
        CONVECTIVELY_UNSTABLE = (STABLE_TEMPERATURE_GRAD 
                                 > ADIABATIC_TEMPERATURE_GRAD)
        permitted_change = 0.01
        drdm = radial_coordinate_differential(self.radial_position,
                                              self.mass_density)
        dpdm = pressure_differential(self.mass, self.radial_position)
        dldm = total_energy_production_rate(self.mass_density,
                                            self.temperature)
        dtdm = compute_temperature_diff_radiative_only(self.radial_position,
                                                       self.mass_density,
                                                       self.temperature,
                                                       self.luminosity)
        if CONVECTIVELY_UNSTABLE:
            #print("convectively unstable")
            dtdm = temperature_diff(self.temperature, self.mass_density,
                                    self.mass, self.radial_position,
                                    self.pressure, self.luminosity, dpdm)
        else:
            #print("convectively stable")
            pass
        mass_differentials = np.asarray([
            permitted_change*self.radial_position/drdm,
            permitted_change*self.pressure/dpdm,
            permitted_change*self.luminosity/dldm,
            permitted_change*self.temperature/dtdm])
        index = np.argmin(np.abs(mass_differentials))
        try:
            step_size_old = step_size
        except UnboundLocalError:
            step_size_old = 0
        step_size = np.abs(mass_differentials[index])
        self.mass = self.mass - step_size
        self.radial_position = self.radial_position - step_size*drdm
        self.pressure = self.pressure - step_size*dpdm
        self.luminosity = self.luminosity - step_size*dldm
        self.temperature = self.temperature - step_size*dtdm
        self.mass_density = pressure_to_mass_density(
            self.temperature, self.pressure)
        self.step_size_difference = step_size - step_size_old

    def compute_and_store_to_file(self):
        """
        numerically integrates the stellar parameter quantities with Euler's
        method, and stores values in file with name filename using CSV.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        initialize_file(self.filename)
        parameters = np.asarray([self.mass, self.radial_position, self.luminosity,
                                self.temperature, self.mass_density, self.pressure])
        append_line_to_data_file(parameters, self.filename)
        while((self.mass > self.init_mass*5e-2 and
              self.radial_position > self.init_radial_position*5e-2 and
              self.luminosity > self.init_luminosity*5e-2) or
              self.step_size_difference > 1e-5):
            self.advance()
            parameters = np.asarray([self.mass, self.radial_position, self.luminosity,
                                    self.temperature, self.mass_density, self.pressure])
            append_line_to_data_file(parameters, self.filename)

    def read_file(self):
        """
        Reads numerical integral results stored in .txt, stores in
        numpy arrays.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        data = np.loadtxt(self.filename, delimiter=",")
        self.masses = data[:, 0]
        self.radial_positions = data[:, 1]
        self.luminosities = data[:, 2]
        self.temperatures = data[:, 3]
        self.mass_densities = data[:, 4]
        self.pressures = data[:, 5]
        self.rel_radial_positions = self.radial_positions/RADIUS_SUN
        self.rel_masses = self.masses/MASS_SUN
        self.rel_temperatures = self.temperatures
        self.rel_mass_densities = self.mass_densities/SURFACE_DENSITY
        self.rel_pressures = self.pressures/self.init_pressure
        self.rel_luminosities = self.luminosities/LUMINOSITY_SUN

    def plot_parameters(self):
        """
        Visualizes quantities solved for, i.e. radial position,
        luminosity, mass density, temperature, and mass.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        fig, ax = plt.subplots(2, 2)  # set figure size
        ax[0, 0].plot(self.rel_radial_positions,
                    self.rel_masses, label="radius")
        ax[0, 0].set_xlabel(r"Radius [$r/R_\odot$]")
        ax[0, 0].set_ylabel(r"Mass $m/M_\odot$")

        ax[0, 1].plot(self.rel_radial_positions,
                    self.rel_luminosities, label="luminosity")
        ax[0, 1].set_xlabel(r"Radius [$r/R_\odot$]")
        ax[0, 1].set_ylabel(r"Luminosity [$L/L_\odot$]")

        ax[1, 0].plot(self.rel_radial_positions,
                    self.rel_temperatures, label="temperature")
        ax[1, 0].set_xlabel(r"Radius [$r/R_\odot$]")
        ax[1, 0].set_ylabel(r"Temperature [K]")

        ax[1, 1].plot(self.rel_radial_positions,
                    self.rel_mass_densities, label="mass density")
        ax[1, 1].set_xlabel(r"Radius [$r/R_\odot$]")
        ax[1, 1].set_ylabel(r"Mass density [$\rho / \bar{\rho}_\odot$]")
        ax[1, 1].set_yscale("log")
        fig.suptitle(self.star_name)
        fig.tight_layout()
        fig.savefig("figs/" + self.star_name + "_parameters.pdf", dpi=300) # set dpi for high-quality image


    def plot_opacity(self):
        """
        Visualizes the opacity. Uninteresting beyond the scope
        of debugging.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        fig, ax = plt.subplots()
        OPACITIES = opacity(self.temperatures, self.mass_densities)
        MAX_OPACITY = np.amax(OPACITIES)
        ax.plot(self.rel_radial_positions, OPACITIES/MAX_OPACITY)
        ax.set_xlabel(r"Radius [$r/R_\odot$]")
        ax.set_ylabel(r"Opacity [$\kappa/\kappa_\mathrm{max}$]")
        ax.set_yscale("log")
        fig.suptitle(self.star_name)
        fig.tight_layout()
        #fig.savefig(self.star_name + "_opacity.pdf")
        

    def plot_gradients(self):
        """
        Visualizes the temperature gradients of interest.

        Args:
            self (EnergyTransport): Instance of EnergyTransport..
        """
        fig, ax = plt.subplots()
        TEMPERATURE_GRADS = temperature_grad(self.masses,
                                             self.radial_positions,
                                             self.mass_densities,
                                             self.temperatures,
                                             self.luminosities)
        STABLE_TEMPEARTURE_GRADS = stable_temperature_grad(self.masses,
                                                           self.radial_positions,
                                                           self.mass_densities,
                                                           self.temperatures,
                                                           self.luminosities)
        ADIABATIC_TEMPERATURE_GRADS = (np.ones_like(self.masses)
                                       * ADIABATIC_TEMPERATURE_GRAD)
        ax.plot(self.rel_radial_positions, TEMPERATURE_GRADS,
                label=r"$\nabla^*$")
        ax.plot(self.rel_radial_positions, STABLE_TEMPEARTURE_GRADS,
                label=r"$\nabla_\mathrm{stable}$", linestyle="--")
        ax.plot(self.rel_radial_positions, ADIABATIC_TEMPERATURE_GRADS,
                label=r"$\nabla_\mathrm{AD}$")
        ax.legend()
        ax.set_xlabel(r"radius [$r/R_\odot$]")
        ax.set_ylabel(r"$\nabla$")
        ax.set_yscale("log")
        fig.suptitle(self.star_name)
        fig.tight_layout()
        fig.savefig("figs/" + self.star_name + "_gradients.pdf")

    def plot_cross_section(self):
        """
        Visualizes the cross section for the star, highlighting the various
        zones in terms of convective and radiative zones, inside and outside
        of the core.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.

        Returns:
            None.
        """
        self.convective_fluxes = convective_flux(self.masses,
                                                 self.radial_positions,
                                                 self.mass_densities,
                                                 self.temperatures,
                                                 self.luminosities)
        cross_section(self.radial_positions, self.luminosities,
                      self.convective_fluxes, self.star_name, show_every=100)

    def plot_fluxes(self):
        """
        Visualizes the convective flux and the radiative flux.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        RADIATIVE_FLUX = radiative_flux(self.masses, self.radial_positions,
                                        self.mass_densities, self.temperatures,
                                        self.luminosities)
        CONVECTIVE_FLUX = convective_flux(self.masses, self.radial_positions,
                                          self.mass_densities,
                                          self.temperatures,
                                          self.luminosities)
        TOTAL_FLUX = CONVECTIVE_FLUX + RADIATIVE_FLUX
        REL_RADIATIVE_FLUX = RADIATIVE_FLUX/TOTAL_FLUX
        REL_CONVECTIVE_FLUX = CONVECTIVE_FLUX/TOTAL_FLUX
        fig, ax = plt.subplots()
        ax.plot(self.rel_radial_positions, REL_RADIATIVE_FLUX,
                label=r"$F_\mathrm{rad}/F_\mathrm{total}$")
        ax.plot(self.rel_radial_positions, REL_CONVECTIVE_FLUX,
                label=r"$F_\mathrm{con}/F_\mathrm{total}$")
        ax.legend()
        ax.set_xlabel(r"radius [$r/R_\odot$]")
        ax.set_ylabel(r"Normalized flux [$F_\mathrm{total}$]")
        fig.tight_layout()
        fig.savefig("figs/" + self.star_name + "_fluxes.pdf")
        print("success?")


    def plot_energy_production_branches(self):
        """
        Visualizes the energy production rates for some sun-like star,
        as a function of radial position.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        """
        EPR_PP1 = np.empty_like(self.masses)
        EPR_PP2 = np.copy(EPR_PP1)
        EPR_PP3 = np.copy(EPR_PP1)
        EPR_CNO = np.copy(EPR_PP1)
        EPR_SUM = np.copy(EPR_PP1)
        for i in range(len(self.masses)):
            EPR_PP1[i], EPR_PP2[i], EPR_PP3[i], EPR_CNO[i] = (
            energy_production_branches(self.mass_densities[i],
                                       self.temperatures[i]))
        fig, ax = plt.subplots()
        TEPR = np.sum([EPR_PP1, EPR_PP2, EPR_PP3, EPR_CNO], axis=0)
        ax.plot(self.rel_radial_positions, EPR_PP1/TEPR, label="PP1")
        ax.plot(self.rel_radial_positions, EPR_PP2/TEPR, label="PP2")
        ax.plot(self.rel_radial_positions, EPR_PP3/TEPR, label="PP3")
        ax.plot(self.rel_radial_positions, EPR_CNO/TEPR, label="CNO")
        ax.legend()
        ax.set_xlabel(r"radius [$r/R_\odot$]")
        ax.set_ylabel(r"Relative energy production")
        fig.savefig("figs/" + self.star_name + "_energy_production_rates.pdf")


def main_star():
    """
    instantiates EnergyTransport for a star with values of the solar surface,
    numerically integrates r, L, P and T with respect to mass, and visualizes
    these, temperature gradients, and a cross section for the star.
    saves figs except for the cross section as pdfs.
    """
    instance = EnergyTransport(star_name="main")
    #instance.compute_and_store_to_file()
    instance.read_file()
    #instance.plot_parameters()
    #instance.plot_gradients()
    instance.plot_cross_section()
    #instance.plot_fluxes()
    #instance.plot_energy_production_branches()

def large_radius_star():
    instance = EnergyTransport(radius=5*RADIUS_SUN,
                               star_name="large_radius")
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()


def small_radius_star():
    instance = EnergyTransport(temperature=5*SURFACE_TEMPERATURE,
                               star_name="small_radius")
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()


def large_temperature_star():
    instance = EnergyTransport(temperature=5*SURFACE_TEMPERATURE,
                               star_name="large_temperature")
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()


def small_temperature_star():
    instance = EnergyTransport(star_name="small_temperature",
                               temperature=SURFACE_TEMPERATURE/5)
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()


def x100_density_star():    
    instance = EnergyTransport(mass_density=SURFACE_DENSITY*1e2,
                               star_name="100x_mass_density")
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()

def x100_pressure_star():
    SURFACE_PRESSURE = mass_density_to_pressure(SURFACE_TEMPERATURE,
                                                SURFACE_DENSITY)
    PRESSURE = SURFACE_PRESSURE*100
    MASS_DENSITY = pressure_to_mass_density(SURFACE_TEMPERATURE, PRESSURE)
    instance = EnergyTransport(mass_density=MASS_DENSITY,
                               star_name="100x_pressure")
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()


def desperate_star_model():
    instance = EnergyTransport(mass_density=100*SURFACE_DENSITY,
                               mass=0.95*MASS_SUN,
                               radius=RADIUS_SUN,
                               temperature=0.8*SURFACE_TEMPERATURE, 
                               star_name="model_sun")
    instance.compute_and_store_to_file()
    instance.read_file()
    instance.plot_parameters()
    instance.plot_gradients()
    instance.plot_cross_section()


def main():
    #sanity_check_opacity()
    #sanity_check_gradients()
    #main_star()
    #large_radius_star()
    #small_radius_star()
    #large_temperature_star()
    #small_temperature_star()
    #x100_density_star()
    #x100_pressure_star()
    desperate_star_model()
    plt.show()

if __name__ == "__main__":
    main()  