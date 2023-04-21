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
import sys

STEFAN_BOLTZMANN_CONSTANT = 5.6704e-8  # W / (m^2 K^4)
GRAVITATIONAL_CONSTANT = 6.6742e-11  # N m^2 / kg^2
BOLTZMANN_CONSTANT = 1.3806e-23  # m^2 kg / (s^2 K)
ATOMIC_MASS_UNIT = 1.6605e-27  # kg
ADIABATIC_TEMPERATURE_GRADIENT = 2 / 5
DELTA = 1
LUMINOSITY_SUN = 3.846e26  # kg m^2 /s^3  (W)
MASS_SUN = 1.989e30  # kg
RADIUS_SUN = 6.96e8  # m
AVERAGE_MASS_DENSITY_SUN = 1.408e3  # kg/m^3


def mean_molecular_mass():
    """
    Computes the mean molecular mass.

    Returns:
        mean molecular mass.
    """
    free_particles = {"hydrogen_1": 1, "helium_3": 2, "helium_4": 2, "lithium_7": 3,
                      "beryllium_7": 4, "nitrogen_14": 7}
    mass_fraction = {"hydrogen_1": 0.7, "helium_3": 1e-10, "helium_4": 0.29,
                     "lithium_7": 1e-7, "beryllium_7": 1e-7, "nitrogen_14": 1e-11}
    nucleons = {"hydrogen_1": 1, "helium_3": 3, "helium_4": 4, "lithium_7": 7,
                "beryllium_7": 4, "nitrogen_14": 14}
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


def rmo_to_mass_density(temperature, rosseland_mean_opacity):
    """
    Calculates the mass density given some temperature and some Rosseland mean
    opacity.

    Args:
        temperature (float): The temperature in units of Kelvin.
        rosseland_mean_opacity (float): The Rosseland mean opacity in units of
            cm^2/g.

    Returns:
        mass_density (float): The mass density in units of g/cm^3
    """
    return rosseland_mean_opacity * (temperature * 1e-6) ** 3


def radial_coordinate_differential(radial_position, mass_density):
    """
    Computes the differential of the radial coordinate with respect to mass,
    for a point at a given radial position within a sun-like star.

    Args:
        radial_position (float): The radial position of the point within the
        star, measured in units of distance from the star's center.
        mass_density (float): The local mass density at the point within the
        star, measured in units of mass per unit volume.

    Returns:
        float: The differential of the radial coordinate with respect to mass,
            measured in units of distance per unit mass.
    """
    return 1 / (4 * np.pi * radial_position**2 * mass_density)


def opacity(temperature, mass_density):
    """
    Computes opacity as functions of the rmo and
    temperature using data from "opacity.txt". Interpolates opacity as
    functions of temperature and mass density.

    Args:
        temperature (float) Temperature in Kelvin.
        mass_density (float) Mass density in kg/m^3

    Returns:
        opacity (float) Opacity [m^2/kg] of sun-like star, as a function
        of temperature and pressure, per interpolation of data
        from 'opacity.txt'.
    """
    data = np.genfromtxt("opacity.txt")
    LOG_10_ROSSELAND_MEAN_OPACITY = data[0][1:]  # g/cm^3
    ROSSELAND_MEAN_OPACITY = 10 ** (LOG_10_ROSSELAND_MEAN_OPACITY)
    LOG_10_TEMPERATURES = data[1:, 0]  # K
    TEMPERATURES = 10 ** (LOG_10_TEMPERATURES)
    LOG_10_OPACITY = data[1:, 1:]  # g/cm^2
    OPACITY = 10 ** (LOG_10_OPACITY)
    mass_densities = np.empty_like(LOG_10_ROSSELAND_MEAN_OPACITY)
    for i, rmo in enumerate(ROSSELAND_MEAN_OPACITY):
        mass_densities[i] = rmo_to_mass_density(TEMPERATURES[i], rmo)
    try:
        spline = RectBivariateSpline(
            TEMPERATURES, mass_densities, OPACITY, bbox=[min(TEMPERATURES), max(
                TEMPERATURES), min(mass_densities), max(mass_densities)]
        )
    except ValueError:
        spline = RectBivariateSpline(
            TEMPERATURES, mass_densities, OPACITY, bbox=[
                -np.inf, np.inf, -np.inf, np.inf]
        )
        print(
            """
WARNING:
opacity accessed with parameter(s) exceeding interpolation bounds.
"""
        )
    opacity = float(spline(temperature, mass_density))
    return opacity


def pressure_to_mass_density(temperature, pressure):
    """
    Compute the mass density given some temperature and pressure.
    Args:
        temperature (float): The temperature [K].
        pressure (float): The pressure in units of Pa. [N/m^2]

    Returns:
        (float) mass density in units of kg/m^3
    """
    return (
        ATOMIC_MASS_UNIT*MEAN_MOLECULAR_MASS/BOLTZMANN_CONSTANT/temperature
        * pressure
    )


def mass_density_to_pressure(temperature, mass_density):
    """
    Compute the pressure given some temperature and mass density.

    Args:
        temperature (float): The temperature [K]
        mass_density (float): The mass density [kg/m^3]
    returns:
        (float) pressure [?]
    """
    return (
        BOLTZMANN_CONSTANT*temperature/MEAN_MOLECULAR_MASS/ATOMIC_MASS_UNIT
        * mass_density
    )


def pressure_differential(mass, radial_position):
    """
    Computes the differential of pressure with respect to mass.

    Args:
        mass (float): Mass inside volume V.
        radial_position (float): The radial position relative to center of a
            sun-like star. [m]

    Returns:
        (float): Differential of pressure with respect to mass.
    """
    return -GRAVITATIONAL_CONSTANT * mass / 4 / np.pi / radial_position**4


def total_energy_production_rate(temperature, mass_density):
    """
    Computes the total energy production rate. (dL/dm)
    Args:
        temperature (float): Temperature [K].
        mass_density (float): Mass density [kg/m^3],

    returns:
        (float): The total energy production rate. []
    """
    ENERGY_PRODUCTION_INSTANCE = EnergyProduction(mass_density, temperature)
    return ENERGY_PRODUCTION_INSTANCE.true_total_energy_production_rate()


def gravitational_acceleration(mass, radial_position):
    """"
    Computes gravitational acceleration

    Returns:
        (float): Gravitational acceleration [m/s^2]
    """
    return mass * GRAVITATIONAL_CONSTANT / radial_position**2


def pressure_scale_height(mass, radial_position, temperature):
    """
    Computes pressure scale height.
    Args:

    Returns:
        (float): Pressure scale height. 
    """
    GRAVITATIONAL_ACCLERATION = gravitational_acceleration(
        mass, radial_position)
    return (
        BOLTZMANN_CONSTANT*temperature/ATOMIC_MASS_UNIT /
        MEAN_MOLECULAR_MASS/GRAVITATIONAL_ACCLERATION
    )


def xi(mass, radial_position, mass_density, temperature, pressure, luminosity):
    """
    computes xi, useful quantity for computing the differential of temperature
    with respect to mass, aswell as convective flux. sequentially computes
    the following quantities for computing xi:

    1. geometric factor,

    2. unnamed but useful quantity U,

    3. temperature gradient needed for all the energy to be carried
    by radiation,
    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    GEOMETRIC_FACTOR = 4 / DELTA / PRESSURE_SCALE_HEIGHT
    OPACITY = opacity(temperature, mass_density)
    u = (
        64
        * STEFAN_BOLTZMANN_CONSTANT
        * temperature**3
        * np.sqrt(PRESSURE_SCALE_HEIGHT / GRAVITATIONAL_ACCELERATION / DELTA)
        / 3
        / OPACITY
        / mass_density**2
        / SPECIFIC_HEAT_CAPACITY
    )
    TEMPERATURE_GRADIENT_STABLE = (
        3
        * luminosity
        * OPACITY
        * mass_density
        * PRESSURE_SCALE_HEIGHT
        / 4
        / np.pi
        / radial_position**2
        / 16
        / STEFAN_BOLTZMANN_CONSTANT
        / temperature**4
    )
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    coeffs = np.asarray(
        [
            1,
            u / MIXING_LENGTH**2,
            u**2 * GEOMETRIC_FACTOR / MIXING_LENGTH**3,
            u
            / MIXING_LENGTH**2
            * (TEMPERATURE_GRADIENT_STABLE - ADIABATIC_TEMPERATURE_GRADIENT),
        ]
    )
    roots = np.roots(coeffs)
    real_root_idx = np.where(np.isreal(roots))[0][0]
    xi = roots[real_root_idx].real
    return xi


def temperature_differential(mass, radial_position, mass_density, temperature,
                             pressure, luminosity, dpdm):
    """
    computes the differential of temperature with respect to mass, by
    sequentially computing the following quantities:



    8. temperature gradient,

    9. differential of temperature with respect to mass

    Args:
        temperature (float): Temperature [K].
        mass_density (float): mass density [kg/m^3].
        mass (float): mass inside volume V [kg].
        radial_position (float): radial position relative to center of a
            sun-like star. [m].
        pressure (float): pressure [kg/(m s^2)].
        luminosity (float): Luminosity produced by a spherical volume with
            radius equal to radial position for a sun-like star [W].
        dpdm: (float): the differential of pressure with respect to mass.

    Returns:
        (float): differential of temperature with respect to mass].
    """
    XI = xi(mass, radial_position, mass_density,
            temperature, pressure, luminosity)
    actual_temperature_gradient = XI**2 - ADIABATIC_TEMPERATURE_GRADIENT
    return actual_temperature_gradient*temperature/pressure*dpdm


def convective_flux(mass, radial_position, mass_density, temperature, pressure, luminosity):
    """
    Computes convective flux.

    Args:
        mass (float): Mass inside volume for sphere with radius equal to
            radial position, centered at center of sun-like star [kg].
        temperature (float): Temperature [K].
    
    Returns:

    """
    GRAVITATIONAL_ACCELERATION = gravitational_acceleration(
        mass, radial_position)
    PRESSURE_SCALE_HEIGHT = pressure_scale_height(
        mass, radial_position, temperature)
    MIXING_LENGTH = 1 * PRESSURE_SCALE_HEIGHT
    XI = xi(mass, radial_position, mass_density,
            temperature, pressure, luminosity)
    return (mass_density*SPECIFIC_HEAT_CAPACITY*temperature
            * np.sqrt(GRAVITATIONAL_ACCELERATION*DELTA/PRESSURE_SCALE_HEIGHT)
            * (MIXING_LENGTH/2)**2*XI**3)


def append_line_to_data_file(data, filename="stellar_parameters.txt"):
    """
    Appends data to new line in specified file
    """
    data = np.reshape(data, (1, len(data)))
    with open(filename, 'a') as f:
        np.savetxt(f, data, delimiter=',', fmt='%f')


def initialize_file(filename="stellar_parameters.txt"):
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


def compute_temperature_differential_radiative_only(
    temperature, mass_density, luminosity, radial_position
):
    """
    Computes the differential of temperature with respect to mass, with
    radiative only.
    """
    OPACITY = opacity(temperature, mass_density)
    return (
        3
        * OPACITY
        * luminosity
        / 256
        / np.pi**2
        / STEFAN_BOLTZMANN_CONSTANT
        / radial_position**4
        / temperature**3
    )


class EnergyTransport:
    """
    Models central parts of a Sun-like star.
    """

    def __init__(self, init_temperature=5770, init_luminosity=LUMINOSITY_SUN,
                 init_mass=MASS_SUN, init_radial_position=RADIUS_SUN,
                 init_mass_density=1.42e-7*AVERAGE_MASS_DENSITY_SUN,
                 filename="stellar_parameters.txt"):
        """
        Initializes instance of EnergyTransport. 
        """
        self.filename = filename
        init_pressure = mass_density_to_pressure(init_temperature,
                                                 init_mass_density)
        self.init_radial_position = init_radial_position
        self.init_mass = init_mass
        init_convective_flux = convective_flux(init_mass,
                                               init_radial_position,
                                               init_mass_density,
                                               init_temperature,
                                               init_pressure,
                                               init_luminosity)
        self.parameters = np.asarray([init_mass, init_radial_position,
                                      init_luminosity, init_temperature,
                                      init_mass_density, init_pressure,
                                      init_convective_flux])

    def advance(self):
        """
        Evolves stellar parameter quantities 1 step with Euler's method, with
        variable step length.

        Args:
            self (EnergyTransport): Instance of EnergyTransport.
        
        
        """
        mass = self.parameters[0]
        radial_position = self.parameters[1]
        luminosity = self.parameters[2]
        temperature = self.parameters[3]
        mass_density = self.parameters[4]
        pressure = self.parameters[5]
        permitted_change = 0.1
        drdm = radial_coordinate_differential(radial_position, mass_density)
        dpdm = pressure_differential(mass, radial_position)
        dldm = total_energy_production_rate(temperature, mass_density)
        dtdm = temperature_differential(temperature, mass_density,
                                        mass, radial_position,
                                        pressure, luminosity, dpdm)
        mass_differentials = np.asarray([
            permitted_change*radial_position/drdm,
            permitted_change*pressure/dpdm,
            permitted_change*luminosity/dldm,
            permitted_change*temperature/dtdm])
        index = np.argmin(np.abs(mass_differentials))
        step_size = np.abs(mass_differentials[index])
        print(f"{step_size=}")
        mass = mass - step_size
        radial_position = radial_position - step_size*drdm
        pressure = pressure - step_size*dpdm
        luminosity = luminosity - step_size*dldm
        temperature = temperature - step_size*dtdm
        mass_density = pressure_to_mass_density(temperature, pressure)
        CONVECTIVE_FLUX = convective_flux(mass, radial_position, mass_density,
                                          temperature, pressure, luminosity)
        self.parameters = np.asarray([mass, radial_position, luminosity,
                                      temperature, mass_density, pressure,
                                      CONVECTIVE_FLUX])

    def compute_and_store_to_file(self):
        """
        numerically integrates the stellar parameter quantities with Euler's
        method, and stores values in file with name filename using CSV.
        """
        initialize_file(self.filename)
        append_line_to_data_file(self.parameters, self.filename)
        tolerance = 1e-5
        mass = self.parameters[0]
        while mass > tolerance:
            self.advance()
            append_line_to_data_file(self.parameters, self.filename)
        print("success")

    def read_file(self):
        """
        Reads numerical integral results stored in .txt, stores in
        numpy arrays.
        """
        data = np.loadtxt(self.filename, delimiter=",")
        self.masses = data[:, 0]
        self.radial_positions = data[:, 1]
        self.luminosities = data[:, 2]
        self.temperatures = data[:, 3]
        self.mass_densities = data[:, 4]
        self.pressures = data[:, 5]
        self.convective_fluxes = data[:, 6]

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.masses, self.radial_positions, label="radius")
        ax.set_xlabel("mass [kg]")
        ax.set_ylabel("radial position [m]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.show()

    def plot_cross_section(self):
        cross_section(self.radial_positions, self.luminosities,
                      self.convective_fluxes)


instance = EnergyTransport(filename="stellar_parameters_0dot1.txt")
instance.compute_and_store_to_file()
instance.read_file()
instance.plot_cross_section()
instance.plot()
